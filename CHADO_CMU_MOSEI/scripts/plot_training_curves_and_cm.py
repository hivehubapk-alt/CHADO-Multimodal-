import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

# Your 6 emotions
EMO_NAMES = ["happy", "sad", "angry", "fearful", "disgust", "surprise"]


def load_curves(curve_path: str):
    """
    Loads curve dict saved by train.py at outputs/logs/{ABLATION}_curves.pt
    Expected keys: epoch, train_loss, val_acc, val_wf1, test_acc, test_wf1, T
    """
    d = torch.load(curve_path, map_location="cpu")
    return d


def safe_arr(x):
    # curve may store None entries
    return np.array([np.nan if v is None else float(v) for v in x], dtype=float)


def plot_accuracy(curves, out_path):
    ep = np.array(curves["epoch"], dtype=int)
    val_acc = safe_arr(curves.get("val_acc", [])) * 100.0
    test_acc = safe_arr(curves.get("test_acc", [])) * 100.0

    plt.figure(figsize=(6.5, 4))
    plt.plot(ep, val_acc, marker="o", linewidth=2, label="Val Accuracy")
    plt.plot(ep, test_acc, marker="o", linewidth=2, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation vs Test Accuracy")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path + ".png", dpi=300)
    plt.savefig(out_path + ".pdf", dpi=300)
    plt.show()


def plot_loss(curves, out_path):
    ep = np.array(curves["epoch"], dtype=int)

    # In your current train.py we definitely have train_loss.
    train_loss = safe_arr(curves.get("train_loss", []))

    # Val/Test loss are only available if you logged them. If not present, we plot train loss only.
    val_loss = curves.get("val_loss", None)
    test_loss = curves.get("test_loss", None)

    plt.figure(figsize=(6.5, 4))
    plt.plot(ep, train_loss, marker="o", linewidth=2, label="Train Loss")

    if val_loss is not None:
        plt.plot(ep, safe_arr(val_loss), marker="o", linewidth=2, label="Val Loss")
    if test_loss is not None:
        plt.plot(ep, safe_arr(test_loss), marker="o", linewidth=2, label="Test Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path + ".png", dpi=300)
    plt.savefig(out_path + ".pdf", dpi=300)
    plt.show()


def plot_confusion_from_counts(cm_counts_path: str, out_path: str):
    """
    For multilabel tasks, the correct confusion is per-class TP/FP/FN/TN.
    This script expects a .pt saved dict:
      {"conf_counts": tensor shape [6,4]} where each row is [TP,FP,FN,TN]
    """
    obj = torch.load(cm_counts_path, map_location="cpu")
    conf = obj["conf_counts"]
    conf = conf.detach().cpu().numpy()

    # heatmap where columns are TP/FP/FN/TN
    labels = ["TP", "FP", "FN", "TN"]

    plt.figure(figsize=(7.5, 4.5))
    im = plt.imshow(conf, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.xticks(range(4), labels)
    plt.yticks(range(len(EMO_NAMES)), EMO_NAMES)
    plt.title("Per-class Confusion Counts (Multilabel)")
    plt.xlabel("Count Type")
    plt.ylabel("Emotion")

    # annotate values
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            plt.text(j, i, str(int(conf[i, j])), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path + ".png", dpi=300)
    plt.savefig(out_path + ".pdf", dpi=300)
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", type=str, default="TAV", choices=["T", "TA", "TV", "TAV"])
    ap.add_argument("--curves", type=str, default=None,
                    help="Path to curves .pt (default outputs/logs/{ablation}_curves.pt)")
    ap.add_argument("--conf_counts", type=str, default=None,
                    help="Path to confusion counts .pt (default outputs/logs/{ablation}_conf_counts.pt)")
    ap.add_argument("--out_dir", type=str, default="outputs/plots_paper")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    curves_path = args.curves or f"outputs/logs/{args.ablation}_curves.pt"
    if not os.path.exists(curves_path):
        raise FileNotFoundError(f"Curves file not found: {curves_path}")

    curves = load_curves(curves_path)

    # Accuracy plot
    plot_accuracy(curves, os.path.join(args.out_dir, f"{args.ablation}_val_test_accuracy"))

    # Loss plot
    plot_loss(curves, os.path.join(args.out_dir, f"{args.ablation}_loss_curve"))

    # Confusion counts plot (optional)
    conf_path = args.conf_counts or f"outputs/logs/{args.ablation}_conf_counts.pt"
    if os.path.exists(conf_path):
        plot_confusion_from_counts(conf_path, os.path.join(args.out_dir, f"{args.ablation}_confusion_counts"))
    else:
        print(f"[WARN] Confusion counts file not found: {conf_path}")
        print("       To generate it, run the provided save_conf_counts script (next).")


if __name__ == "__main__":
    main()
