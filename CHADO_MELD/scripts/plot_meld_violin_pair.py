import numpy as np
import matplotlib.pyplot as plt

def _mean_ci95(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    m = float(np.mean(x))
    # normal approx; bootstrap already gives distribution, but this is OK for displayed CI
    se = float(np.std(x, ddof=1) / np.sqrt(max(1, len(x))))
    ci = 1.96 * se
    return m, m - ci, m + ci

def _draw_panel(ax, title, ylab, base_vals, chado_vals, bootstrap_pairs,
                violin_color="#b7d3ea", mean_color="#ff7f0e"):
    """
    base_vals, chado_vals: arrays of per-seed metrics (already in %)
    bootstrap_pairs: list of tuples (b_sample_mean, c_sample_mean) used to draw many lines
                     e.g., generated from bootstrap resamples of the paired seeds.
    """
    x0, x1 = 0, 1

    # --- violin (same light-blue look) ---
    parts = ax.violinplot(
        [base_vals, chado_vals],
        positions=[x0, x1],
        widths=0.85,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for pc in parts["bodies"]:
        pc.set_facecolor(violin_color)
        pc.set_edgecolor("none")
        pc.set_alpha(0.70)

    # --- many thin paired lines (multi-colored, low alpha) ---
    # We color each bootstrap line by cycling through matplotlib’s default color cycle.
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    for i, (b, c) in enumerate(bootstrap_pairs):
        col = prop_cycle[i % len(prop_cycle)]
        ax.plot([x0, x1], [b, c], color=col, alpha=0.20, linewidth=1.2, zorder=2)

    # --- mean + 95% CI (orange dot + vertical bar) ---
    b_mean, b_lo, b_hi = _mean_ci95(base_vals)
    c_mean, c_lo, c_hi = _mean_ci95(chado_vals)

    ax.errorbar(
        [x0, x1],
        [b_mean, c_mean],
        yerr=[[b_mean - b_lo, c_mean - c_lo], [b_hi - b_mean, c_hi - c_mean]],
        fmt="o",
        color=mean_color,
        ecolor=mean_color,
        elinewidth=3.0,
        capsize=8,
        markersize=11,
        zorder=5,
    )

    # --- axis formatting to match IEMOCAP sample ---
    ax.set_title(title, fontsize=22, pad=12)
    ax.set_ylabel(ylab, fontsize=18)
    ax.set_xticks([x0, x1])
    ax.set_xticklabels(["Baseline", "CHADO"], fontsize=18)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(True, axis="both", alpha=0.25, linewidth=1)
    ax.set_axisbelow(True)

def plot_meld_violin_pair_iemocap_style(
    baseline_acc, baseline_f1, chado_acc, chado_f1,
    bootstrap_lines=120, seed=0, out_path="figures/meld_violin_pair.png"
):
    """
    baseline_acc/baseline_f1/chado_acc/chado_f1: per-seed arrays (in %)
    bootstrap_lines: number of spaghetti lines (like sample); 120+ recommended
    """
    rng = np.random.default_rng(seed)

    # Bootstrap paired means to create many line segments
    n = len(baseline_acc)
    idx = np.arange(n)
    boot_acc = []
    boot_f1  = []
    for _ in range(bootstrap_lines):
        samp = rng.choice(idx, size=n, replace=True)
        boot_acc.append((float(np.mean(baseline_acc[samp])), float(np.mean(chado_acc[samp]))))
        boot_f1.append((float(np.mean(baseline_f1[samp])),  float(np.mean(chado_f1[samp]))))

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=160)
    _draw_panel(
        axes[0],
        title="MELD Accuracy: Violin + Mean 95% CI",
        ylab="Acc (%)",
        base_vals=baseline_acc,
        chado_vals=chado_acc,
        bootstrap_pairs=boot_acc,
    )
    _draw_panel(
        axes[1],
        title="MELD WF1: Violin + Mean 95% CI",
        ylab="WF1 (%)",
        base_vals=baseline_f1,
        chado_vals=chado_f1,
        bootstrap_pairs=boot_f1,
    )

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
