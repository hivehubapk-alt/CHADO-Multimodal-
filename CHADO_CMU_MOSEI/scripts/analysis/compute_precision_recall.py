import argparse
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

EMO_NAMES = ["happy", "sad", "angry", "fearful", "disgust", "surprise"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True, help="Path to test_probs.pt")
    ap.add_argument("--labels", required=True, help="Path to test_labels.pt")
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()

    probs = torch.load(args.probs, map_location="cpu")    # [N,6]
    labels = torch.load(args.labels, map_location="cpu")  # [N,6]

    y_true = (labels > 0.5).int().numpy()
    y_pred = (probs > args.thr).int().numpy()

    # Per-class
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    print("\nPer-class Precision / Recall / F1")
    for i, name in enumerate(EMO_NAMES):
        print(f"{name:8s}  P={p[i]:.4f}  R={r[i]:.4f}  F1={f1[i]:.4f}  supp={support[i]}")

    # Micro / Macro / Weighted
    for avg in ["micro", "macro", "weighted"]:
        p_avg, r_avg, f1_avg, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0
        )
        print(f"\n[{avg.upper()}]")
        print(f"Precision = {p_avg:.4f}")
        print(f"Recall    = {r_avg:.4f}")
        print(f"F1        = {f1_avg:.4f}")

if __name__ == "__main__":
    main()
