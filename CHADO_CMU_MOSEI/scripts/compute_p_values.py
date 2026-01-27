import numpy as np
from scipy.stats import ttest_rel, wilcoxon

# =========================
# EDIT THESE VALUES
# =========================

# Example: Baseline vs CHADO over 5 seeds
baseline_acc = np.array([73.2, 72.9, 73.5, 73.1, 72.8])
chado_acc    = np.array([78.1, 77.8, 78.4, 78.0, 77.9])

baseline_wf1 = np.array([52.0, 51.7, 52.3, 51.9, 51.6])
chado_wf1    = np.array([58.1, 57.8, 58.4, 58.0, 57.9])

# =========================
# PAIRED T-TEST
# =========================
t_acc, p_acc = ttest_rel(chado_acc, baseline_acc)
t_wf1, p_wf1 = ttest_rel(chado_wf1, baseline_wf1)

# =========================
# WILCOXON (NON-PARAMETRIC)
# =========================
w_acc, p_acc_w = wilcoxon(chado_acc, baseline_acc)
w_wf1, p_wf1_w = wilcoxon(chado_wf1, baseline_wf1)

print("=== Accuracy ===")
print(f"Paired t-test:   t={t_acc:.3f}, p={p_acc:.6f}")
print(f"Wilcoxon test:   W={w_acc:.3f}, p={p_acc_w:.6f}")

print("\n=== WF1 ===")
print(f"Paired t-test:   t={t_wf1:.3f}, p={p_wf1:.6f}")
print(f"Wilcoxon test:   W={w_wf1:.3f}, p={p_wf1_w:.6f}")
