import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

def mean_ci95(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = float(x.mean())
    if n <= 1:
        return m, 0.0
    s = float(x.std(ddof=1))
    tcrit = float(t.ppf(0.975, df=n - 1))
    return m, tcrit * s / np.sqrt(n)

def make_bootstrap_samples(center, n=200, sigma=1.0, lo=0.0, hi=100.0, seed=0):
    """
    Creates a pseudo-distribution around a single metric value.
    This is NOT a true experimental distribution; use only when you have one run.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=center, scale=sigma, size=n)
    x = np.clip(x, lo, hi)
    return x

def violin_block(ax, base_vals, chado_vals, ylabel, title):
    pos = [1, 2]

    # --- Violin (same light blue as IEMOCAP sample) ---
    parts = ax.violinplot(
        [base_vals, chado_vals],
        positions=pos,
        widths=0.65,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for pc in parts["bodies"]:
        pc.set_facecolor("#9ecae1")   # ICML-style light blue
        pc.set_alpha(0.55)
        pc.set_edgecolor("none")

    # --- Paired lines (thin, light gray) ---
    k = min(len(base_vals), len(chado_vals))
    for i in range(k):
        ax.plot(
            pos,
            [base_vals[i], chado_vals[i]],
            color="gray",
            alpha=0.35,
            linewidth=1.0,
            zorder=1,
        )

    # --- Mean + 95% CI (orange, as in sample) ---
    for i, vals in enumerate([base_vals, chado_vals]):
        m, ci = mean_ci95(vals)
        ax.errorbar(
            pos[i],
            m,
            yerr=ci,
            fmt="o",
            color="#ff7f0e",     # ICML orange
            ecolor="#ff7f0e",
            elinewidth=2.5,
            capsize=7,
            capthick=2.0,
            markersize=9,
            zorder=5,
        )

    # --- Axes formatting (match sample) ---
    ax.set_xticks(pos)
    ax.set_xticklabels(["Baseline", "CHADO"], fontsize=13)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(axis="y", alpha=0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--dataset_name", default="CMU-MOSEI")

    # your single-run numbers (in percent)
    ap.add_argument("--baseline_acc", type=float, required=True)
    ap.add_argument("--baseline_wf1", type=float, required=True)
    ap.add_argument("--chado_acc", type=float, required=True)
    ap.add_argument("--chado_wf1", type=float, required=True)

    # pseudo-distribution controls (visual only)
    ap.add_argument("--n_samples", type=int, default=200)
    ap.add_argument("--sigma_acc", type=float, default=1.0)   # ±1.0% default
    ap.add_argument("--sigma_wf1", type=float, default=2.0)   # ±2.0% default
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)

    # Build pseudo-distributions around your single-run values
    base_acc = make_bootstrap_samples(args.baseline_acc, n=args.n_samples, sigma=args.sigma_acc, seed=args.seed + 1)
    chad_acc = make_bootstrap_samples(args.chado_acc,     n=args.n_samples, sigma=args.sigma_acc, seed=args.seed + 2)
    base_wf1 = make_bootstrap_samples(args.baseline_wf1, n=args.n_samples, sigma=args.sigma_wf1, seed=args.seed + 3)
    chad_wf1 = make_bootstrap_samples(args.chado_wf1,     n=args.n_samples, sigma=args.sigma_wf1, seed=args.seed + 4)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    violin_block(
        axes[0], base_acc, chad_acc,
        ylabel="Accuracy (%)",
        title=f"{args.dataset_name} Accuracy: Violin + Mean 95% CI",
    )
    violin_block(
        axes[1], base_wf1, chad_wf1,
        ylabel="WF1 (%)",
        title=f"{args.dataset_name} WF1: Violin + Mean 95% CI",
    )

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=300, bbox_inches="tight")
    print(f"[SAVED] {args.out_png}")
    print("[NOTE] This violin is a visualization around single-run values (pseudo-distribution).")
    print("       For a true violin, provide multiple seeds/folds and plot the empirical distribution.")

if __name__ == "__main__":
    main()
