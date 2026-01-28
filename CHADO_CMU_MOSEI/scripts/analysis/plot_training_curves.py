import pandas as pd, matplotlib.pyplot as plt

def main():
    df = pd.read_csv("outputs/logs/TAV_curves.csv")

    plt.figure(figsize=(6,4))
    plt.plot(df["epoch"], df["val_acc"], label="Val Acc")
    plt.plot(df["epoch"], df["test_acc"], label="Test Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig("outputs/analysis/plots/acc_curve.png", dpi=300)
    print("[SAVED] acc_curve.png")

if __name__ == "__main__":
    main()
