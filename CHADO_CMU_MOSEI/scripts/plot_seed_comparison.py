import matplotlib.pyplot as plt
import numpy as np
import os

# =========================
# EDIT THESE WITH REAL RESULTS
# =========================

# Seeds you actually ran
seeds = np.array([42, 43, 44, 45, 46])

# Baseline results (replace with your baseline TAV results)
baseline_acc = np.array([
    72.9,
    73.1,
    72.8,
    73.3,
    73.0
])

# CHADO results (replace with your CHADO TAV results)
chado_acc = np.array([
    78.1,
    77.8,
    78.4,
    78.0,
    77.9
])

# =========================
# PLOTTING
# =========================

os.makedirs("outputs/plots", exist_ok=True)

plt.figure(figsize=(7, 4))

plt.plot(seeds, baseline_acc, marker="o", linewidth=2,
         label="Baseline (TAV)", color="red")

plt.plot(seeds, chado_acc, marker="o", linewidth=2,
         label="CHADO (TAV)", color="green")

# Draw connecting lines (paired nature)
for i in range(len(seeds)):
    plt.plot([seeds[i], seeds[i]],
             [baseline_acc[i], chado_acc[i]],
             color="gray", alpha=0.4)

plt.xlabel("Random Seed", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.title("Seed-wise Accuracy Comparison on CMU-MOSEI", fontsize=13)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("outputs/plots/seedwise_accuracy_comparison.pdf", dpi=300)
plt.savefig("outputs/plots/seedwise_accuracy_comparison.png", dpi=300)
plt.show()
