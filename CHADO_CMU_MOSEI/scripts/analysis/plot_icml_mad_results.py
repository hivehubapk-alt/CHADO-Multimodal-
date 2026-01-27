import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUT_DIR = "outputs/analysis"
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    corr = pd.read_csv(f"{OUT_DIR}/mad_correlations.csv")
    cost = pd.read_csv(f"{OUT_DIR}/cost_reduction.csv")

    # ---- extract values from long-format table ----
    corr_map = dict(zip(corr["metric"], corr["value"]))

    pearson_r = corr_map["pearson_r"]
    spearman_r = corr_map["spearman_r"]
    pearson_p = corr_map["pearson_p"]
    spearman_p = corr_map["spearman_p"]

    sns.set(style="whitegrid", font_scale=1.25)

    # ===============================
    # 1) MAD Correlation Bar Plot
    # ===============================
    plt.figure(figsize=(5.5,4))
    bars = sns.barplot(
        x=["Pearson", "Spearman"],
        y=[pearson_r, spearman_r],
        palette="Blues"
    )

    # annotate p-values
    for i, p in enumerate([pearson_p, spearman_p]):
        bars.text(
            i,
            [pearson_r, spearman_r][i] + 0.01,
            f"p={p:.3f}",
            ha="center",
            fontsize=11
        )

    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel("Correlation with Error")
    plt.title("MAD vs Prediction Error Correlation")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/mad_correlation_bar.png", dpi=300)
    plt.close()

    # ===============================
    # 2) Cost Reduction Curve
    # ===============================
    plt.figure(figsize=(6,4))
    plt.plot(
        cost["retained_fraction"],
        cost["accuracy"],
        marker="o",
        linewidth=2,
        label="Accuracy"
    )
    plt.plot(
        cost["retained_fraction"],
        cost["cost_reduction"],
        linestyle="--",
        linewidth=2,
        label="Cost Reduction"
    )
    plt.xlabel("Fraction of Samples Retained")
    plt.ylabel("Metric")
    plt.title("MAD-based Cost Reduction Analysis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/mad_cost_curve.png", dpi=300)
    plt.close()

    print("[SAVED]")
    print(" - outputs/analysis/mad_correlation_bar.png")
    print(" - outputs/analysis/mad_cost_curve.png")
    print("\n[VALUES]")
    print(f"Pearson r={pearson_r:.4f}, p={pearson_p:.4g}")
    print(f"Spearman r={spearman_r:.4f}, p={spearman_p:.4g}")

if __name__ == "__main__":
    main()
