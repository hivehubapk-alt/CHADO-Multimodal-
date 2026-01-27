import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def mean_std(df, metric):
    g = df.groupby("ablation")[metric].agg(["mean","std"]).reset_index()
    return g

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", default="outputs/cross_env/figures")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["test_acc"] = df["test_acc"].astype(float)
    df["test_wf1"] = df["test_wf1"].astype(float)

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    for metric, ylabel, fname in [
        ("test_acc", "Accuracy (%)", "loeo_acc_bar.png"),
        ("test_wf1", "Weighted F1 (%)", "loeo_wf1_bar.png")
    ]:
        g = mean_std(df, metric)
        # convert to percent
        g["mean"] = g["mean"] * (100.0 if metric=="test_wf1" else 1.0)
        g["std"]  = g["std"]  * (100.0 if metric=="test_wf1" else 1.0)

        # If your CSV already stores acc as percent (it does: 68.04), keep it.
        # Your WF1 in CSV is in percent-like numbers (57.22), also keep it.
        # So do NOT rescale:
        g = mean_std(df, metric)

        x = np.arange(len(g))
        plt.figure(figsize=(6.5, 3.8))
        plt.bar(x, g["mean"].values, yerr=g["std"].values, capsize=4)
        plt.xticks(x, g["ablation"].values)
        plt.ylabel(ylabel)
        plt.xlabel("Modality")
        plt.grid(True, axis="y", alpha=0.25)

        for i, (m, s) in enumerate(zip(g["mean"].values, g["std"].values)):
            plt.text(i, m + s + 0.3, f"{m:.2f}±{s:.2f}", ha="center", va="bottom", fontsize=9)

        out_path = f"{args.out_dir}/{fname}"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        print(f"[SAVED] {out_path}")

if __name__ == "__main__":
    main()
