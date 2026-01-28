import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/analysis/tables/hparam_results.csv")

for param in df["param"].unique():
    sub = df[df["param"] == param]
    plt.figure()
    plt.plot(sub["value"], sub["Accuracy"], marker="o")
    plt.xlabel(param)
    plt.ylabel("Accuracy (%)")
    plt.title(f"Hyperparameter Sensitivity: {param}")
    plt.tight_layout()
    plt.savefig(f"outputs/analysis/plots/sensitivity_{param}.png")
    plt.close()
