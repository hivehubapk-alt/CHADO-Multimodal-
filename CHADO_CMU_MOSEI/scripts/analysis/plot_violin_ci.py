import pandas as pd, matplotlib.pyplot as plt, seaborn as sns

def main():
    df = pd.read_csv("outputs/analysis/seed_results.csv")

    plt.figure(figsize=(6,4))
    sns.violinplot(
        data=df, x="model", y="wf1",
        inner="quartile", cut=0
    )
    plt.title("WF1 distribution across seeds")
    plt.grid(axis="y", alpha=0.3)
    plt.savefig("outputs/analysis/plots/violin_wf1.png", dpi=300)
    print("[SAVED] violin_wf1.png")

if __name__ == "__main__":
    main()
