import os
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Utilities
# ----------------------------
def bootstrap_ci_mean(x: np.ndarray, n_boot: int = 20000, ci: float = 0.95, seed: int = 123):
    """
    Bootstrap CI for the mean.
    Returns: (mean, lo, hi)
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    n = len(x)
    boot_means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        samp = rng.choice(x, size=n, replace=True)
        boot_means[i] = samp.mean()
    lo = np.quantile(boot_means, (1 - ci) / 2)
    hi = np.quantile(boot_means, 1 - (1 - ci) / 2)
    return x.mean(), lo, hi


def violin_with_ci(ax, data_groups, labels, title, ylabel, show_paired_lines=False, paired_a=None, paired_b=None):
    """
    data_groups: list of arrays, one per group (e.g., [baseline, chado])
    labels: list of group labels
    """
    positions = np.arange(1, len(data_groups) + 1)

    parts = ax.violinplot(
        data_groups,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )

    # Style violins
    for pc in parts["bodies"]:
        pc.set_alpha(0.25)
        pc.set_linewidth(1.2)

    # Overlay mean + 95% CI
    for i, x in enumerate(data_groups):
        mean, lo, hi = bootstrap_ci_mean(x, n_boot=20000, ci=0.95, seed=100 + i)
        ax.errorbar(
            positions[i], mean,
            yerr=[[mean - lo], [hi - mean]],
            fmt="o", capsize=6, linewidth=2
        )
        # small horizontal line at mean
        ax.hlines(mean, positions[i] - 0.15, positions[i] + 0.15, linewidth=2)

    # Optional paired lines (if exactly 2 groups and paired arrays provided)
    if show_paired_lines and paired_a is not None and paired_b is not None:
        a = np.asarray(paired_a, dtype=float)
        b = np.asarray(paired_b, dtype=float)
        assert len(a) == len(b), "paired arrays must have same length"
        # jitter slightly to reduce overlap
        ja = positions[0] + np.linspace(-0.03, 0.03, len(a))
        jb = positions[1] + np.linspace(-0.03, 0.03, len(b))
        for k in range(len(a)):
            ax.plot([ja[k], jb[k]], [a[k], b[k]], alpha=0.35, linewidth=1.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)


# ----------------------------
# EDIT HERE: paste your real seed metrics
# ----------------------------
seeds = np.array([42, 43, 44, 45, 46])

# Accuracy (%)
baseline_acc = np.array([73.2, 72.9, 73.5, 73.1, 72.8])
chado_acc    = np.array([78.1, 77.8, 78.4, 78.0, 77.9])

# WF1 (%)
baseline_wf1 = np.array([52.0, 51.7, 52.3, 51.9, 51.6])
chado_wf1    = np.array([58.1, 57.8, 58.4, 58.0, 57.9])

# ----------------------------
# Plot
# ----------------------------
os.makedirs("outputs/plots", exist_ok=True)

fig = plt.figure(figsize=(10, 4.5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

violin_with_ci(
    ax1,
    data_groups=[baseline_acc, chado_acc],
    labels=["Baseline", "CHADO"],
    title="CMU-MOSEI Accuracy: Violin + Mean 95% CI",
    ylabel="Accuracy (%)",
    show_paired_lines=True,
    paired_a=baseline_acc,
    paired_b=chado_acc
)

violin_with_ci(
    ax2,
    data_groups=[baseline_wf1, chado_wf1],
    labels=["Baseline", "CHADO"],
    title="CMU-MOSEI WF1: Violin + Mean 95% CI",
    ylabel="WF1 (%)",
    show_paired_lines=True,
    paired_a=baseline_wf1,
    paired_b=chado_wf1
)

plt.tight_layout()
plt.savefig("outputs/plots/violin_ci_acc_wf1.png", dpi=300)
plt.savefig("outputs/plots/violin_ci_acc_wf1.pdf", dpi=300)
plt.show()

print("Saved:",
      "outputs/plots/violin_ci_acc_wf1.png",
      "and outputs/plots/violin_ci_acc_wf1.pdf")
