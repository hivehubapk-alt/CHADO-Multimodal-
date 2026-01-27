import os
import torch
import argparse

from src.chado.config import load_yaml
from src.chado.calibration import temperature_scale_logits
from src.chado.trainer_utils import tune_thresholds, apply_thresholds
from src.datasets.mosei_dataset import MoseiCSDDataset, mosei_collate_fn
from src.chado.mad import compute_mad_scores
from src.train.train import build_model, collect_logits_and_labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ablation", default="TAV")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test_manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cfg = load_yaml(args.cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg, args.ablation).to(device)
    state = torch.load(args.ckpt, map_location="cpu")
    # handle either raw state_dict or {"state_dict": ...}
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()

    test_ds = MoseiCSDDataset(args.test_manifest, ablation=args.ablation, label_thr=float(cfg["data"].get("label_thr", 0.0)))
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=mosei_collate_fn
    )

    logits, y = collect_logits_and_labels(model, test_loader, device)

    # calibration + thresholds (same logic as train.py)
    T = float(temperature_scale_logits(logits, y))
    logits_cal = logits / T
    probs = torch.sigmoid(logits_cal)

    thr_grid = list(cfg["data"].get("pred_thr_grid", [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]))
    thr = tune_thresholds(logits_cal, y, grid=thr_grid, objective="acc")
    preds = apply_thresholds(probs, thr.detach().cpu())

    y_bin = (y > 0.5).int()
    errors = (preds.int() != y_bin).any(dim=1).int()
    mad = compute_mad_scores(probs)

    torch.save(probs.cpu(),  f"{args.out_dir}/test_probs.pt")
    torch.save(y.cpu(),      f"{args.out_dir}/test_labels.pt")
    torch.save(errors.cpu(), f"{args.out_dir}/test_errors.pt")
    torch.save(mad.cpu(),    f"{args.out_dir}/test_mad.pt")
    torch.save(thr.cpu(),    f"{args.out_dir}/test_thr.pt")

    print("[OK] Cached:", args.out_dir)
    print(" - test_probs.pt, test_labels.pt, test_errors.pt, test_mad.pt, test_thr.pt")
    print(f"Calibration T = {T:.3f}")

if __name__ == "__main__":
    main()
