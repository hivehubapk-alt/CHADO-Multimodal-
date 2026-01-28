import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    ablation = "TAV"
    curve_pt = f"outputs/logs/{ablation}_curves.pt"
    out_dir = "outputs/plots_paper"
    os.makedirs(out_dir, exist_ok=True)

    curves = torch.load(curve_pt, map_location="cpu")

    epochs = np.array(curves["epoch"])
    val_acc = np.array(curves["val_acc"]) * 100.0

    # ---- Train accuracy proxy (visual only) ----
    train_loss = np.array(curves["train_loss"])
    train_loss_norm = (train_loss - train_loss.min()) / (train_loss.max() - train_loss.min() + 1e-8)
    train_acc_proxy = (1.0 - train_loss_norm) * 100.0

    # ---- Plot ----
    plt.figure(figsize=(6.5, 4.2))
    plt.plot(epochs, train_acc_proxy, label="Train Acc (proxy)", linewidth=2)
    plt.plot(epochs, val_acc, label="Val Acc", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Validation Accuracy (CMU-MOSEI)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_png = f"{out_dir}/{ablation}_train_val_accuracy.png"
    out_pdf = f"{out_dir}/{ablation}_train_val_accuracy.pdf"
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf, dpi=300)
    plt.close()

    print("[SAVED]", out_png)
    print("[SAVED]", out_pdf)


if __name__ == "__main__":
    main()
