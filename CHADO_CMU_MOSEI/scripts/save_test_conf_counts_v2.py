import os
import argparse
import torch
from torch.utils.data import DataLoader

from src.chado.config import load_yaml
from src.chado.trainer_utils import tune_thresholds, apply_thresholds
from src.chado.calibration import temperature_scale_logits
from src.chado.metrics import compute_acc_wf1

from src.datasets.mosei_dataset import MoseiCSDDataset, mosei_collate_fn
from src.train.train import build_model, collect_logits_and_labels  # reuse your tested functions

EMO_NAMES = ["happy", "sad", "angry", "fearful", "disgust", "surprise"]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True)
    ap.add_argument("--ablation", type=str, required=True, choices=["T", "TA", "TV", "TAV"])
    ap.add_argument("--ckpt", type=str, default=None, help="default outputs/checkpoints/{ablation}_best.pt")
    ap.add_argument("--out", type=str, default=None, help="default outputs/logs/{ablation}_conf_counts.pt")
    args = ap.parse_args()

    cfg = load_yaml(args.cfg)
    proj = cfg["experiment"]["project_root"]

    train_manifest = f"{proj}/data/manifests/mosei_train.jsonl"
    val_manifest   = f"{proj}/data/manifests/mosei_val.jsonl"
    test_manifest  = f"{proj}/data/manifests/mosei_test.jsonl"

    label_thr = float(cfg["data"].get("label_thr", 0.0))
    batch_size = int(cfg["data"].get("batch_size", 16))
    num_workers = int(cfg["data"].get("num_workers", 4))

    thr_grid = list(cfg["data"].get(
        "pred_thr_grid",
        [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    ))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.ckpt or f"outputs/checkpoints/{args.ablation}_best.pt"
    out_path  = args.out  or f"outputs/logs/{args.ablation}_conf_counts.pt"

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Build datasets/loaders exactly like train.py (no DDP here)
    val_ds = MoseiCSDDataset(val_manifest, ablation=args.ablation, label_thr=label_thr)
    test_ds = MoseiCSDDataset(test_manifest, ablation=args.ablation, label_thr=label_thr)

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, collate_fn=mosei_collate_fn, pin_memory=True, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, collate_fn=mosei_collate_fn, pin_memory=True, drop_last=False
    )

    # Build model and load weights
    model = build_model(cfg, args.ablation).to(device)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)  # your ckpt is a plain state_dict
    model.eval()

    # ---- Mirror your evaluation logic: compute T and thresholds on VAL ----
    val_logits, val_y = collect_logits_and_labels(model, val_loader, device)
    T = float(temperature_scale_logits(val_logits, val_y))
    val_logits_cal = val_logits / T
    val_probs = torch.sigmoid(val_logits_cal)

    thr_vec = tune_thresholds(val_logits_cal, val_y, grid=thr_grid, objective="acc")

    val_pred = apply_thresholds(val_probs, thr_vec)
    val_true = (val_y > 0.5).int()
    val_acc, val_wf1, _ = compute_acc_wf1(val_true, val_pred)

    # ---- Apply same T + thresholds on TEST ----
    test_logits, test_y = collect_logits_and_labels(model, test_loader, device)
    test_logits_cal = test_logits / T
    test_probs = torch.sigmoid(test_logits_cal)

    test_pred = apply_thresholds(test_probs, thr_vec.detach().cpu())
    test_true = (test_y > 0.5).int()
    test_acc, test_wf1, conf = compute_acc_wf1(test_true, test_pred)

    # Save confusion counts (conf already in [TP,FP,FN,TN] per class from your metrics)
    os.makedirs("outputs/logs", exist_ok=True)
    torch.save(
        {
            "conf_counts": conf.detach().cpu(),
            "T": T,
            "thr": thr_vec.detach().cpu(),
            "val_acc": float(val_acc),
            "val_wf1": float(val_wf1),
            "test_acc": float(test_acc),
            "test_wf1": float(test_wf1),
        },
        out_path
    )

    print(f"[VAL ] Acc={val_acc*100:.2f}% WF1={val_wf1*100:.2f}  T={T:.3f}")
    print(f"[TEST] Acc={test_acc*100:.2f}% WF1={test_wf1*100:.2f}  T={T:.3f}")
    print("thr:", [round(float(x), 3) for x in thr_vec.tolist()])
    print(f"[DONE] Saved confusion counts: {out_path}")


if __name__ == "__main__":
    main()
