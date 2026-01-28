import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon, t
import os

# ======================================================
# 1. INSERT YOUR REAL RESULTS HERE (ACCURACY or WF1)
# ======================================================

seeds = np.array([42, 43, 44, 45, 46])

baseline = np.array([73.2, 72.9, 73.5, 73.1, 72.8])   # Baseline TAV
chado    = np.array([78.1, 77.8, 78.4, 78.0, 77.9])   # CHADO TAV

metric_name = "Accuracy (%)"

# ======================================================
# 2. BASIC STATISTICS
# ======================================================

n = len(seeds)
diff = chado - baseline

mean_base = baseline.mean()
std_base  = baseline.std(ddof=1)

mean_ch   = chado.mean()
std_ch    = chado.std(ddof=1)

# 95% CI
alpha = 0.05
tcrit = t.ppf(1 - alpha/2, df=n-1)

ci_base = tcrit * std_base / np.sqrt(n)
ci_ch   = tcrit * std_ch / np.sqrt(n)

# ======================================================
# 3. STATISTICAL TESTS
# ======================================================

t_stat, p_ttest = ttest_rel(chado, baseline)
w_stat, p_wilcoxon = wilcoxon(chado, baseline)

# Cohen's d (paired)
cohens_d = diff.mean() / diff.std(ddof=1)

# ======================================================
# 4. PRINT RESULTS (PAPER READY)
# ======================================================

print("\n===== Statistical Summary =====")
print(f"Metric: {metric_name}")
print(f"Seeds: {seeds.tolist()}")

print("\nBaseline:")
print(f"  Mean ± Std = {mean_base:.2f} ± {std_base:.2f}")
print(f"  95% CI     = [{mean_base-ci_base:.2f}, {mean_base+ci_base:.2f}]")

print("\nCHADO:")
print(f"  Mean ± Std = {mean_ch:.2f} ± {std_ch:.2f}")
print(f"  95% CI     = [{mean_ch-ci_ch:.2f}, {mean_ch+ci_ch:.2f}]")

print("\nPaired Statistical Tests:")
print(f"  Paired t-test: t = {t_stat:.3f}, p = {p_ttest:.6f}")
print(f"  Wilcoxon test: W = {w_stat:.3f}, p = {p_wilcoxon:.6f}")

print("\nEffect Size:")
print(f"  Cohen's d (paired) = {cohens_d:.3f}")

# ======================================================
# 5. PLOTS (BEST FOR PAPERS)
# ======================================================

os.makedirs("outputs/plots", exist_ok=True)

# ---------- Plot A: Mean ± 95% CI (BEST) ----------
plt.figure(figsize=(5.5,4))
means = [mean_base, mean_ch]
cis   = [ci_base, ci_ch]

plt.bar(["Baseline", "CHADO"], means, yerr=cis,
        capsize=6, color=["red", "green"], alpha=0.75)

plt.ylabel(metric_name)
plt.title(f"{metric_name} (Mean ± 95% CI)")
plt.text(0.5, max(means)+0.3, f"p < 0.001", ha="center", fontsize=11)

plt.tight_layout()
plt.savefig("outputs/plots/mean_ci_barplot.png", dpi=300)
plt.savefig("outputs/plots/mean_ci_barplot.pdf", dpi=300)
plt.show()

# ---------- Plot B: Seed-wise paired plot ----------
plt.figure(figsize=(6.5,4))
plt.plot(seeds, baseline, marker="o", label="Baseline", color="red")
plt.plot(seeds, chado, marker="o", label="CHADO", color="green")

for i in range(n):
    plt.plot([seeds[i], seeds[i]], [baseline[i], chado[i]],
             color="gray", alpha=0.4)

plt.xlabel("Seed")
plt.ylabel(metric_name)
plt.title("Seed-wise Paired Comparison")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/plots/seedwise_paired.png", dpi=300)
plt.savefig("outputs/plots/seedwise_paired.pdf", dpi=300)
plt.show()

# ---------- Plot C: t-distribution with p-value ----------
x = np.linspace(-6, 6, 2000)
pdf = t.pdf(x, df=n-1)

plt.figure(figsize=(7.5,4))
plt.plot(x, pdf, label=f"t-distribution (df={n-1})")

t_abs = abs(t_stat)
plt.fill_between(x[x <= -t_abs], t.pdf(x[x <= -t_abs], n-1),
                 alpha=0.3)
plt.fill_between(x[x >= t_abs], t.pdf(x[x >= t_abs], n-1),
                 alpha=0.3)

plt.axvline(t_stat, linestyle="--", label=f"t = {t_stat:.2f}")
plt.title(f"Paired t-test p-value visualization (p={p_ttest:.2e})")
plt.xlabel("t")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.25)

plt.tight_layout()
plt.savefig("outputs/plots/pvalue_tdist.png", dpi=300)
plt.savefig("outputs/plots/pvalue_tdist.pdf", dpi=300)
plt.show()
