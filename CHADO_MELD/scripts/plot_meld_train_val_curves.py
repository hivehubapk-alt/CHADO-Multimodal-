import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt


EPOCH_RE = re.compile(
    r"\[epoch\s*(\d+)\]\s+loss=([0-9.]+)\s+train_acc=([0-9.]+)\s+val_acc=([0-9.]+)\s+val_f1w=([0-9.]+)"
)


def moving_average(x, w=3):
    if w <= 1 or len(x) < w:
        return np.array(x, dtype=float)
    x = np.array(x, dtype=float)
    out = np.convolve(x, np.ones(w) / w, mode="same")
    return out


def parse_log(log_path: str):
    epochs = []
    loss = []
    train_acc = []
    val_acc = []
    val_f1w = []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if m:
                e = int(m.group(1))
                epochs.append(e)
                loss.append(float(m.group(2)))
                train_acc.append(float(m.group(3)))
                val_acc.append(float(m.group(4)))
                val_f1w.append(float(m.group(5)))

    if not epochs:
        raise RuntimeError(
            f"No epoch lines matched.\n"
            f"Expected pattern like:\n"
            f"[epoch 3] loss=1.4589 train_acc=0.6128 val_acc=0.6440 val_f1w=0.6319\n"
            f"Log: {log_path}"
        )

    # Sort by epoch just in case
    order = np.argsort(epochs)
    epochs = np.array(epochs)[order]
    loss = np.array(loss)[order]
    train_acc = np.array(train_acc)[order]
    val_acc = np.array(val_acc)[order]
    val_f1w = np.array(val_f1w)[order]

    return epochs, loss, train_acc, val_acc, val_f1w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to training stdout log file")
    ap.add_argument("--out_dir", default="figures", help="Where to save plots")
    ap.add_argument("--title", default="MELD Training vs Validation", help="Plot title prefix")
    ap.add_argument("--smooth", type=int, default=1, help="Moving average window (1 = off)")
    ap.add_argument("--test_acc", type=float, default=None, help="Optional: final test accuracy (0-1 or 0-100)")
    ap.add_argument("--test_f1", type=float, default=None, help="Optional: final test weighted F1 (0-1 or 0-100)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    epochs, loss, tr_acc, va_acc, va_f1w = parse_log(args.log)

    # Convert to percent for nicer presentation
    tr_acc_pct = tr_acc * 100.0
    va_acc_pct = va_acc * 100.0
    va_f1_pct = va_f1w * 100.0

    # Optional test metrics: accept either [0,1] or [0,100]
    test_acc = None
    test_f1 = None
    if args.test_acc is not None:
        test_acc = args.test_acc * 100.0 if args.test_acc <= 1.0 else args.test_acc
    if args.test_f1 is not None:
        test_f1 = args.test_f1 * 100.0 if args.test_f1 <= 1.0 else args.test_f1

    # Smoothing
    tr_acc_s = moving_average(tr_acc_pct, args.smooth)
    va_acc_s = moving_average(va_acc_pct, args.smooth)
    va_f1_s = moving_average(va_f1_pct, args.smooth)

    best_idx = int(np.argmax(va_f1_pct))
    best_epoch = int(epochs[best_idx])
    best_val_acc = float(va_acc_pct[best_idx])
    best_val_f1 = float(va_f1_pct[best_idx])

    # -------- Plot 1: Accuracy curves (clean + professional) --------
    plt.figure(figsize=(8.5, 4.8))
    plt.plot(epochs, tr_acc_pct, label="Train Acc", linewidth=2)
    plt.plot(epochs, va_acc_pct, label="Val Acc", linewidth=2)

    if args.smooth and args.smooth > 1:
        plt.plot(epochs, tr_acc_s, linestyle="--", linewidth=1.8, label=f"Train Acc (MA{args.smooth})")
        plt.plot(epochs, va_acc_s, linestyle="--", linewidth=1.8, label=f"Val Acc (MA{args.smooth})")

    # Mark best epoch (by Val F1)
    plt.axvline(best_epoch, linestyle=":", linewidth=2)
    plt.text(
        best_epoch,
        max(va_acc_pct.max(), tr_acc_pct.max()) * 0.92,
        f"Best (Val F1)\nE{best_epoch}: ValAcc={best_val_acc:.2f} ValF1={best_val_f1:.2f}",
        ha="left",
        va="top",
        fontsize=9,
    )

    if test_acc is not None:
        plt.axhline(test_acc, linestyle="--", linewidth=1.8)
        plt.text(epochs.min(), test_acc + 0.5, f"Test Acc = {test_acc:.2f}%", va="bottom", fontsize=9)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{args.title}: Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out1 = os.path.join(args.out_dir, "meld_train_vs_val_accuracy.png")
    plt.savefig(out1, dpi=300)
    plt.close()

    # -------- Plot 2: Validation Acc vs Validation WF1 (attractive, paper-friendly) --------
    plt.figure(figsize=(8.5, 4.8))
    plt.plot(epochs, va_acc_pct, label="Val Acc", linewidth=2)
    plt.plot(epochs, va_f1_pct, label="Val WF1", linewidth=2)

    if args.smooth and args.smooth > 1:
        plt.plot(epochs, va_acc_s, linestyle="--", linewidth=1.8, label=f"Val Acc (MA{args.smooth})")
        plt.plot(epochs, va_f1_s, linestyle="--", linewidth=1.8, label=f"Val WF1 (MA{args.smooth})")

    plt.scatter([best_epoch], [best_val_f1], s=80, zorder=5)
    plt.text(best_epoch, best_val_f1 + 0.6, f"Best Val WF1 = {best_val_f1:.2f}%", fontsize=9)

    if test_f1 is not None:
        plt.axhline(test_f1, linestyle="--", linewidth=1.8)
        plt.text(epochs.min(), test_f1 + 0.5, f"Test WF1 = {test_f1:.2f}%", va="bottom", fontsize=9)

    plt.xlabel("Epoch")
    plt.ylabel("Score (%)")
    plt.title(f"{args.title}: Validation Acc vs WF1")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out2 = os.path.join(args.out_dir, "meld_val_accuracy_vs_wf1.png")
    plt.savefig(out2, dpi=300)
    plt.close()

    print("\n=== Parsed Training Summary ===")
    print(f"Log: {args.log}")
    print(f"Epochs parsed: {len(epochs)} (min={epochs.min()} max={epochs.max()})")
    print(f"Best epoch by Val WF1: {best_epoch} | ValAcc={best_val_acc:.2f}% | ValWF1={best_val_f1:.2f}%")
    if test_acc is not None:
        print(f"Test Acc: {test_acc:.2f}%")
    if test_f1 is not None:
        print(f"Test WF1: {test_f1:.2f}%")
    print(f"[OK] Saved:\n  {out1}\n  {out2}\n")


if __name__ == "__main__":
    main()
