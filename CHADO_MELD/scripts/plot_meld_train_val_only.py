import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Expected log line format:
# [epoch 3] loss=1.4589 train_acc=0.6128 val_acc=0.6440 val_f1w=0.6319
PATTERN = re.compile(
    r"\[epoch\s*(\d+)\].*?train_acc=([0-9.]+).*?val_acc=([0-9.]+)"
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Training log file")
    ap.add_argument("--out", default="figures/meld_train_vs_val_accuracy.png")
    ap.add_argument("--title", default="Training vs Validation Accuracy (MELD)")
    args = ap.parse_args()

    epochs, train_acc, val_acc = [], [], []

    with open(args.log, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = PATTERN.search(line)
            if m:
                epochs.append(int(m.group(1)))
                train_acc.append(float(m.group(2)) * 100)
                val_acc.append(float(m.group(3)) * 100)

    if not epochs:
        raise RuntimeError("No train/val accuracy lines found in log.")

    # Sort by epoch
    order = np.argsort(epochs)
    epochs = np.array(epochs)[order]
    train_acc = np.array(train_acc)[order]
    val_acc = np.array(val_acc)[order]

    # -------- PLOT --------
    plt.figure(figsize=(8, 4.8))
    plt.plot(epochs, train_acc, label="Train Accuracy", linewidth=2)
    plt.plot(epochs, val_acc, label="Validation Accuracy", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(args.title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=300)
    plt.close()

    print(f"[OK] Saved → {args.out}")
    print(f"Final Train Acc: {train_acc[-1]:.2f}%")
    print(f"Final Val Acc:   {val_acc[-1]:.2f}%")

if __name__ == "__main__":
    main()
