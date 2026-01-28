import argparse
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


def _t_critical_95(df: int) -> float:
    # Prefer exact t critical if scipy exists; otherwise fall back to 1.96.
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, df=max(1, df)))
    except Exception:
        return 1.96


def mean_ci95(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n <= 1:
        return (float(np.mean(x)) if n == 1 else float("nan")), 0.0
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    se = s / math.sqrt(n)
    tcrit = _t_critical_95(n - 1)
    return m, tcrit * se


def _read_pairs_from_csv(csv_path: str, baseline_ablation: str, chado_ablation: str):
    df = pd.read_csv(csv_path)

    # Expected format for your LOEO CSV:
    # fold,ablation,test_acc,test_wf1
    required = {"fold", "ablation"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV missing required columns: {required}. Found: {list(df.columns)}")

    # Accept either test_acc/test_wf1 or acc/wf1
    if "test_acc" in df.columns:
        acc_col = "test_acc"
    elif "acc" in df.columns:
        acc_col = "acc"
    else:
        raise ValueError("CSV must contain test_acc or acc.")

    if "test_wf1" in df.columns:
        wf1_col = "test_wf1"
    elif "wf1" in df.columns:
        wf1_col = "wf1"
    else:
        raise ValueError("CSV must contain test_wf1 or wf1.")

    b = df[df["ablation"].astype(str) == str(baseline_ablation)].copy()
    c = df[df["ablation"].astype(str) == str(chado_ablation)].copy()

    if len(b) == 0 or len(c) == 0:
        raise ValueError(
            f"No rows found for baseline_ablation={baseline_ablation} or chado_ablation={chado_ablation}."
        )

    # Pair by fold (LOEO folds)
    merged = pd.merge(
        b[["fold", acc_col, wf1_col]].rename(columns={acc_col: "base_acc", wf1_col: "base_wf1"}),
        c[["fold", acc_col, wf1_col]].rename(columns={acc_col: "chado_acc", wf1_col: "chado_wf1"}),
        on="fold",
        how="inner",
    ).sort_values("fold")

    if len(merged) == 0:
        raise ValueError("No paired folds after merge. Ensure both ablations have the same fold ids.")

    # If your CSV stores metrics as percentages already, keep them.
    # If they are in [0,1], convert to [%].
    for col in ["base_acc", "chado_acc", "base_wf1", "chado_wf1"]:
        if merged[col].max() <= 1.5:
            merged[col] = merged[col] * 100.0

    return merged


def _pastel_cycle(n: int, cmap_name: str = "tab20"):
    """
    Soft multi-color paired lines similar to the sample image.
    tab20 produces many distinct but still soft colors at alpha<0.4.
    """
    cmap = mpl.cm.get_cmap(cmap_name)
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    out = [colors[i % len(colors)] for i in range(n)]
    return out


def _style_rcparams():
    # Close to ICML-like figure defaults (matplotlib only, no seaborn).
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
        "axes.linewidth": 1.2,
        "figure.dpi": 150,
        "savefig.dpi": 150,
    })


def _violin_pair_panel(ax, base_vals, chado_vals, ylabel, title, cmap_lines="tab20"):
    # Colors to match your sample
    violin_fill = "#9ecae1"          # light blue
    violin_alpha = 0.55
    mean_ci_color = "#ff7f0e"        # orange

    # Violin plots (same fill for baseline & CHADO)
    parts = ax.violinplot([base_vals, chado_vals], positions=[1, 2], widths=0.70, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(violin_fill)
        pc.set_edgecolor("none")
        pc.set_alpha(violin_alpha)

    # Paired lines (multi pastel colors)
    n = min(len(base_vals), len(chado_vals))
    line_colors = _pastel_cycle(n, cmap_name=cmap_lines)
    for i in range(n):
        ax.plot(
            [1, 2],
            [base_vals[i], chado_vals[i]],
            color=line_colors[i],
            alpha=0.30,
            linewidth=1.3,
            zorder=2,
        )

    # Mean + 95% CI (orange)
    bm, bci = mean_ci95(base_vals)
    cm, cci = mean_ci95(chado_vals)

    ax.errorbar(
        [1, 2],
        [bm, cm],
        yerr=[bci, cci],
        fmt="o",
        color=mean_ci_color,
        ecolor=mean_ci_color,
        elinewidth=3.0,
        capsize=7,
        markersize=9,
        zorder=5,
    )

    # Baseline/CHADO horizontal mean bars (subtle blue like sample’s “caps” feel)
    ax.hlines(bm, 1 - 0.18, 1 + 0.18, colors="#1f77b4", linewidth=2.5, zorder=4)
    ax.hlines(cm, 2 - 0.18, 2 + 0.18, colors="#1f77b4", linewidth=2.5, zorder=4)

    ax.set_xlim(0.6, 2.4)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Baseline", "CHADO"])
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)

    # Light grid (as in sample)
    ax.grid(True, axis="both", linestyle="-", linewidth=0.8, alpha=0.35)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="LOEO summary CSV (e.g., fold,ablation,test_acc,test_wf1).")
    parser.add_argument("--out_png", required=True)
    parser.add_argument("--dataset_name", default="CMU-MOSEI")
    parser.add_argument("--baseline_ablation", required=True, help="e.g., T")
    parser.add_argument("--chado_ablation", required=True, help="e.g., TAV")
    parser.add_argument("--cmap_lines", default="tab20", help="Line colormap for paired lines (tab20 recommended).")
    parser.add_argument("--fig_w", type=float, default=14.0)
    parser.add_argument("--fig_h", type=float, default=4.6)
    args = parser.parse_args()

    _style_rcparams()

    paired = _read_pairs_from_csv(args.csv, args.baseline_ablation, args.chado_ablation)

    base_acc = paired["base_acc"].to_numpy()
    chado_acc = paired["chado_acc"].to_numpy()
    base_wf1 = paired["base_wf1"].to_numpy()
    chado_wf1 = paired["chado_wf1"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(args.fig_w, args.fig_h), constrained_layout=True)

    _violin_pair_panel(
        axes[0],
        base_acc,
        chado_acc,
        ylabel="Accuracy (%)",
        title=f"{args.dataset_name} Accuracy: Violin + Mean 95% CI",
        cmap_lines=args.cmap_lines,
    )
    _violin_pair_panel(
        axes[1],
        base_wf1,
        chado_wf1,
        ylabel="WF1 (%)",
        title=f"{args.dataset_name} WF1: Violin + Mean 95% CI",
        cmap_lines=args.cmap_lines,
    )

    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    fig.savefig(args.out_png, bbox_inches="tight")
    print(f"[SAVED] {args.out_png}")
    print(f"[PAIRS] folds used: {paired['fold'].tolist()}")
    print(f"[BASE] ablation={args.baseline_ablation}  [CHADO] ablation={args.chado_ablation}")


if __name__ == "__main__":
    main()
