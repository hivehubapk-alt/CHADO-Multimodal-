import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.chado.config import load_yaml
from src.datasets.mosei_dataset import MoseiCSDDataset, mosei_collate_fn
from src.train.train import build_model, collect_logits_and_labels

EMO_NAMES = ["happy", "sad", "angry", "fearful", "disgust", "surprise"]


def normalized_confusion_matrix(y_true_idx: np.ndarray, y_pred_idx: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true_idx, y_pred_idx):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1

    # row-normalize (per true class)
    cm = cm.astype(np.float32)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, np.maximum(row_sum, 1.0))
    return cm_norm


def plot_cm(cm_norm: np.ndarray, class_names, out_png: str, title: str):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    fig = plt.figure(figsize=(10.5, 8.5))
    ax = plt.gca()

    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Normalized frequency", rotation=90)

    ax.set_title(title, pad=14)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    # annotate
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=11)

    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close(fig)
    print(f"[SAVED] {out_png}")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, type=str)
    ap.add_argument("--ablation", default="TAV", choices=["T", "TA", "TV", "TAV"])
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--T", default=1.3, type=float, help="Calibration temperature (use your best T, e.g., 1.300)")
    ap.add_argument("--batch_size", default=64, type=int)
    ap.add_argument("--out_png", default="outputs/plots/mosei_confusion_normalized.png", type=str)
    ap.add_argument("--title", default="CMU-MOSEI Confusion Matrix (Normalized)", type=str)
    ap.add_argument("--show_metrics_in_title", action="store_true",
                    help="If set, appends '(Acc=..., WF1=...)' using provided --acc --wf1 (text only).")
    ap.add_argument("--acc", default=None, type=float, help="Optional: 78.14 (for title only)")
    ap.add_argument("--wf1", default=None, type=float, help="Optional: 58.12 (for title only)")
    args = ap.parse_args()

    cfg = load_yaml(args.cfg)
    proj = cfg["experiment"]["project_root"]
    test_manifest = f"{proj}/data/manifests/mosei_test.jsonl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build + load model
    model = build_model(cfg, args.ablation).to(device)

    state = torch.load(args.ckpt, map_location="cpu")
    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        # if you saved only module state_dict in DDP etc.
        model.load_state_dict(state, strict=False)

    model.eval()

    # Data
    test_ds = MoseiCSDDataset(test_manifest, ablation=args.ablation, label_thr=float(cfg["data"].get("label_thr", 0.0)))
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        collate_fn=mosei_collate_fn,
        pin_memory=True
    )

    # logits + labels
    logits, y = collect_logits_and_labels(model, test_loader, device)  # logits [N,C], y [N,C]

    # single-label projection (needed for NxN confusion matrix)
    probs = torch.sigmoid(logits / float(args.T))  # calibrated probs

    y_true_idx = torch.argmax(y, dim=1).cpu().numpy().astype(int)
    y_pred_idx = torch.argmax(probs, dim=1).cpu().numpy().astype(int)

    cm_norm = normalized_confusion_matrix(y_true_idx, y_pred_idx, n_classes=len(EMO_NAMES))

    title = args.title
    if args.show_metrics_in_title and (args.acc is not None) and (args.wf1 is not None):
        title = f"{title} (Acc={args.acc:.2f}, WF1={args.wf1:.2f})"

    plot_cm(cm_norm, EMO_NAMES, args.out_png, title)


if __name__ == "__main__":
    main()
