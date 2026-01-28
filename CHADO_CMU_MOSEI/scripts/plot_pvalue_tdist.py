import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, ttest_rel

# =========================
# PUT YOUR REAL SEED RESULTS HERE
# =========================
baseline = np.array([73.2, 72.9, 73.5, 73.1, 72.8])  # baseline accuracy per seed
chado    = np.array([78.1, 77.8, 78.4, 78.0, 77.9])  # CHADO accuracy per seed

# Paired t-test
t_stat, p_val = ttest_rel(chado, baseline)
n = len(baseline)
df = n - 1

# Build t-distribution curve
x = np.linspace(-6, 6, 2000)
pdf = t.pdf(x, df)

plt.figure(figsize=(8, 4.5))
plt.plot(x, pdf, linewidth=2, label=f"t-distribution (df={df})")

# Shade p-value region (two-tailed)
t_abs = abs(t_stat)

# Left tail
x_left = x[x <= -t_abs]
plt.fill_between(x_left, t.pdf(x_left, df), alpha=0.3, label="p-value region (tails)")

# Right tail
x_right = x[x >= t_abs]
plt.fill_between(x_right, t.pdf(x_right, df), alpha=0.3)

# Observed t lines
plt.axvline(t_stat, linestyle="--", linewidth=2, label=f"observed t = {t_stat:.3f}")
plt.axvline(-t_abs, linestyle=":", linewidth=1)
plt.axvline(t_abs, linestyle=":", linewidth=1)

plt.title(f"Paired t-test p-value visualization (p={p_val:.4g})", fontsize=12)
plt.xlabel("t")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

import os
os.makedirs("outputs/plots", exist_ok=True)
plt.savefig("outputs/plots/pvalue_tdist_accuracy.pdf", dpi=300)
plt.savefig("outputs/plots/pvalue_tdist_accuracy.png", dpi=300)
plt.show()
