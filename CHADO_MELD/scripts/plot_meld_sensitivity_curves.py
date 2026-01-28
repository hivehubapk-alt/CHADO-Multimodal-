#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with lambda_type, lambda_value, test_acc, test_wf1")
    ap.add_argument("--out", required=True, help="Output image path")
    ap.add_argument("--title", default="MELD Sensitivity (λ / curvature)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # drop empty rows
    df = df.dropna(subset=["test_acc", "test_wf1"])
    df["lambda_value"] = df["lambda_value"].astype(float)

    types = ["mad", "ot", "hyp", "curv"]
    present = [t for t in types if t in set(df["lambda_type"].astype(str))]

    plt.figure(figsize=(10, 4.2), dpi=200)
    gs = plt.GridSpec(1, 2, wspace=0.25)

    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])

    for t in present:
        sub = df[df["lambda_type"] == t].sort_values("lambda_value")
        ax1.plot(sub["lambda_value"], sub["test_acc"], marker="o", linewidth=2, label=t)
        ax2.plot(sub["lambda_value"], sub["test_wf1"], marker="o", linewidth=2, label=t)

    ax1.set_title("Test Accuracy vs λ / curvature")
    ax2.set_title("Test WF1 vs λ / curvature")

    ax1.set_xlabel("λ value (or curvature c)")
    ax2.set_xlabel("λ value (or curvature c)")

    ax1.set_ylabel("Test Acc (%)")
    ax2.set_ylabel("Test WF1 (%)")

    ax1.grid(True, alpha=0.25)
    ax2.grid(True, alpha=0.25)

    ax2.legend(loc="best", frameon=True)

    plt.suptitle(args.title, y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight")
    print(f"[OK] Saved → {args.out}")

if __name__ == "__main__":
    main()
