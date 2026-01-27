import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.chado.config import load_yaml
from src.chado.trainer_utils import tune_thresholds, apply_thresholds
from src.chado.calibration import temperature_scale_logits
from src.chado.metrics import compute_acc_wf1

from src.datasets.mosei_dataset import MoseiCSDDataset, mosei_collate_fn
from src.models.baseline_fusion import BaselineFusion

EMO_NAMES = ["happy", "sad", "angry", "fearful", "disgust", "surprise"]


def build_model(cfg, ablation: str):
    mcfg = cfg["model"]
    use_audio = ablation in ("TA", "TAV") and bool(mcfg.get("use_audio", True))
    use_video = ablation in ("TV", "TAV") and bool(mcfg.get("use_video", True))
    model = BaselineFusion(
        num_classes=6,
        d_model=int(mcfg.get("d_model", 256)),
        use_audio=use_audio,
        use_video=use_video,
        text_model=str(mcfg.get("text_model", "roberta-base")),
        max_text_len=int(cfg["data"].get("max_text_len", 96)),
        modality_dropout=float(mcfg.get("modality_dropout", 0.1)),
    )
    return model


@torch.no_grad()
def forward_logits(model, batch, device):
    # Move tensors only
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    out = model(inputs)
    if isinstance(out, (tuple, list)) and len(out) == 2:
        return out[0]
    return out


@torch.no_grad()
def collect_logits_and_labels(model, loader, device):
    model.eval()
    all_logits, all_y = [], []
    for batch in loader:
        y = batch["label"].to(device)
        logits = forward_logits(model, batch, device)
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())
    logits = torch.cat(all_logits, dim=0) if all_logits else torch.zeros((0, 6))
    y = torch.cat(all_y, dim=0) if all_y else torch.zeros((0, 6))
    return logits, y


def plot_acc_curve(curve_pt, out_dir, ablation):
    d = torch.load(curve_pt, map_location="cpu")
    ep = np.array(d["epoch"], dtype=int)
    val_acc = np.array(d["val_acc"], dtype=float) * 100.0

    # test_acc may contain None
    test_acc_raw = d.get("test_acc", [None] * len(ep))
    test_acc = np.array([np.nan if v is None else float(v) * 100.0 for v in test_acc_raw], dtype=float)

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{ablation}_val_test_accuracy")

    plt.figure(figsize=(6.5, 4))
    plt.plot(ep, val_acc, marker="o", linewidth=2, label="Val Accuracy")
    if not np.all(np.isnan(test_acc)):
        plt.plot(ep, test_acc, marker="o", linewidth=2, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation vs Test Accuracy")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(base + ".png", dpi=300)
    plt.savefig(base + ".pdf", dpi=300)
    plt.close()

    return base + ".png", base + ".pdf"


def plot_loss_curve(curve_pt, out_dir, ablation, val_loss=None, test_loss=None):
    d = torch.load(curve_pt, map_location="cpu")
    ep = np.array(d["epoch"], dtype=int)
    train_loss = np.array(d["train_loss"], dtype=float)

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{ablation}_loss_curve")

    plt.figure(figsize=(6.5, 4))
    plt.plot(ep, train_loss, marker="o", linewidth=2, label="Train Loss")

    # If we only have final val/test loss, draw horizontal reference lines (no retraining needed)
    if val_loss is not None:
        plt.hlines(val_loss, ep.min(), ep.max(), linestyles="--", linewidth=2, label=f"Val Loss (final)={val_loss:.4f}")
    if test_loss is not None:
        plt.hlines(test_loss, ep.min(), ep.max(), linestyles="--", linewidth=2, label=f"Test Loss (final)={test_loss:.4f}")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve (Train) + Final Val/Test Reference")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(base + ".png", dpi=300)
    plt.savefig(base + ".pdf", dpi=300)
    plt.close()

    return base + ".png", base + ".pdf"


def plot_conf_counts_heatmap(conf_counts, out_dir, ablation):
    # conf_counts shape [6,4] where columns are TP/FP/FN/TN
    labels = ["TP", "FP", "FN", "TN"]
    conf = conf_counts.detach().cpu().numpy()

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{ablation}_confusion_counts")

    plt.figure(figsize=(7.6, 4.6))
    im = plt.imshow(conf, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(4), labels)
    plt.yticks(range(len(EMO_NAMES)), EMO_NAMES)
    plt.title("Per-class Confusion Counts (Multilabel)")
    plt.xlabel("Count Type")
    plt.ylabel("Emotion")

    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            plt.text(j, i, str(int(conf[i, j])), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(base + ".png", dpi=300)
    plt.savefig(base + ".pdf", dpi=300)
    plt.close()

    return base + ".png", base + ".pdf"


@torch.no_grad()
def compute_final_losses_and_conf(cfg, ablation, ckpt_path, device):
    proj = cfg["experiment"]["project_root"]
    train_manifest = f"{proj}/data/manifests/mosei_train.jsonl"
    val_manifest = f"{proj}/data/manifests/mosei_val.jsonl"
    test_manifest = f"{proj}/data/manifests/mosei_test.jsonl"

    label_thr = float(cfg["data"].get("label_thr", 0.0))
    batch_size = int(cfg["data"].get("batch_size", 16))
    num_workers = int(cfg["data"].get("num_workers", 4))

    thr_grid = list(cfg["data"].get("pred_thr_grid", [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]))

    train_ds = MoseiCSDDataset(train_manifest, ablation=ablation, label_thr=label_thr)
    val_ds = MoseiCSDDataset(val_manifest, ablation=ablation, label_thr=label_thr)
    test_ds = MoseiCSDDataset(test_manifest, ablation=ablation, label_thr=label_thr)

    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers,
                            shuffle=False, collate_fn=mosei_collate_fn, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers,
                             shuffle=False, collate_fn=mosei_collate_fn, pin_memory=True)

    model = build_model(cfg, ablation).to(device)
    state = torch.load(ckpt_path, map_location="cpu")
    # Your train.py saves raw state_dict, not {"model":...}
    model.load_state_dict(state, strict=False)
    model.eval()

    # --- VAL: compute T and thresholds (same logic as your training) ---
    val_logits, val_y = collect_logits_and_labels(model, val_loader, device)
    T = float(temperature_scale_logits(val_logits, val_y))
    val_logits_cal = val_logits / T
    thr_vec = tune_thresholds(val_logits_cal, val_y, grid=thr_grid, objective="acc")

    # --- Compute final Val/Test losses (BCE for paper curve reference lines) ---
    bce = torch.nn.BCEWithLogitsLoss()

    val_loss = float(bce(val_logits_cal, val_y).item()) if val_logits_cal.numel() > 0 else None

    test_logits, test_y = collect_logits_and_labels(model, test_loader, device)
    test_logits_cal = test_logits / T
    test_loss = float(bce(test_logits_cal, test_y).item()) if test_logits_cal.numel() > 0 else None

    # --- TEST predictions and confusion counts ---
    test_probs = torch.sigmoid(test_logits_cal)
    test_pred = apply_thresholds(test_probs, thr_vec.detach().cpu())
    test_true = (test_y > 0.5).int()

    # metrics (should match your already achieved performance; does not change training)
    test_acc, test_wf1, conf = compute_acc_wf1(test_true, test_pred)

    # conf is expected shape [6,4] TP FP FN TN
    conf_counts = conf.detach().cpu()

    return {
        "T": T,
        "thr": thr_vec.detach().cpu(),
        "val_loss": val_loss,
        "test_loss": test_loss,
        "test_acc": float(test_acc),
        "test_wf1": float(test_wf1),
        "conf_counts": conf_counts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, type=str)
    ap.add_argument("--ablation", required=True, choices=["T", "TA", "TV", "TAV"])
    ap.add_argument("--ckpt", default=None, type=str)
    ap.add_argument("--curve_pt", default=None, type=str)
    ap.add_argument("--out_dir", default="outputs/plots_paper", type=str)
    args = ap.parse_args()

    cfg = load_yaml(args.cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.ckpt or f"outputs/checkpoints/{args.ablation}_best.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    curve_pt = args.curve_pt or f"outputs/logs/{args.ablation}_curves.pt"
    if not os.path.exists(curve_pt):
        raise FileNotFoundError(f"Curves file not found: {curve_pt}")

    # 1) Plot curves from saved training logs (NO retraining)
    acc_png, acc_pdf = plot_acc_curve(curve_pt, args.out_dir, args.ablation)

    # 2) Compute final losses + confusion from best checkpoint (NO retraining)
    stats = compute_final_losses_and_conf(cfg, args.ablation, ckpt_path, device)

    # 3) Plot loss curve using train_loss + final val/test loss reference lines
    loss_png, loss_pdf = plot_loss_curve(curve_pt, args.out_dir, args.ablation,
                                         val_loss=stats["val_loss"], test_loss=stats["test_loss"])

    # 4) Plot confusion counts heatmap (multilabel)
    cm_png, cm_pdf = plot_conf_counts_heatmap(stats["conf_counts"], args.out_dir, args.ablation)

    print("\n=== Paper Plots Saved ===")
    print("Accuracy curve:", acc_png, "and", acc_pdf)
    print("Loss curve:", loss_png, "and", loss_pdf)
    print("Confusion heatmap:", cm_png, "and", cm_pdf)

    print("\n=== Check (does not change training) ===")
    print(f"[TEST] Acc={stats['test_acc']*100:.2f}%  WF1={stats['test_wf1']*100:.2f}%  T={stats['T']:.3f}")
    print("Thresholds:", [round(float(x), 3) for x in stats["thr"].tolist()])


if __name__ == "__main__":
    main()
