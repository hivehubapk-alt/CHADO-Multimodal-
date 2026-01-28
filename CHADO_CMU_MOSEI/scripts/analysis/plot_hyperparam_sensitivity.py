import pandas as pd, matplotlib.pyplot as plt

def main():
    df = pd.read_csv("outputs/analysis/hyperparam_sweep.csv")

    plt.figure(figsize=(6,4))
    plt.plot(df["mad_gamma"], df["wf1"], marker="o")
    plt.xlabel("MAD γ"); plt.ylabel("WF1")
    plt.grid(alpha=0.3)
    plt.savefig("outputs/analysis/plots/hyperparam_mad.png", dpi=300)
    print("[SAVED] hyperparam_mad.png")

if __name__ == "__main__":
    main()
