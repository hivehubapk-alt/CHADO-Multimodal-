import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.chado.config import load_yaml
from src.train.train import build_model
from src.datasets.mosei_dataset import MoseiCSDDataset, mosei_collate_fn
from src.chado.mad import compute_mad_scores
from src.chado.metrics import compute_acc_wf1
from src.chado.calibration import temperature_scale_logits
from src.chado.trainer_utils import tune_thresholds, apply_thresholds

EMO_NAMES = ["happy", "sad", "angry", "fearful", "disgust", "surprise"]


@torch.no_grad()
def forward_logits(model, batch, device):
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    out = model(inputs)
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


@torch.no_grad()
def collect_logits_and_labels(model, loader, device):
    L, Y = [], []
    model.eval()
    for batch in loader:
        y = batch["label"].to(device)
        logits = forward_logits(model, batch, device)
        L.append(logits.detach().cpu())
        Y.append(y.detach().cpu())
    return torch.cat(L, 0), torch.cat(Y, 0)


def plot_bar(df, ycol, out_base, title):
    plt.figure(figsize=(6.4, 4.2))
    x = np.arange(len(df))
    plt.bar(x, df[ycol].values)
    plt.xticks(x, df["Stratum"].values)
    plt.ylabel(ycol + " (%)")
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_base + ".png", dpi=300)
    plt.savefig(out_base + ".pdf", dpi=300)
    plt.close()


def main():
    cfg_path = "src/configs/chado_mosei_emo6.yaml"
    ablation = "TAV"
    ckpt_path = f"outputs/checkpoints/{ablation}_best.pt"

    out_table = f"outputs/analysis/tables/{ablation}_ambiguity_strata.csv"
    out_plot_acc = f"outputs/analysis/plots/{ablation}_ambiguity_strata_acc"
    out_plot_wf1 = f"outputs/analysis/plots/{ablation}_ambiguity_strata_wf1"

    os.makedirs("outputs/analysis/tables", exist_ok=True)
    os.makedirs("outputs/analysis/plots", exist_ok=True)

    cfg = load_yaml(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loaders (test + val for calibration/thresholds)
    proj = cfg["experiment"]["project_root"]
    val_manifest = f"{proj}/data/manifests/mosei_val.jsonl"
    test_manifest = f"{proj}/data/manifests/mosei_test.jsonl"

    label_thr = float(cfg["data"].get("label_thr", 0.0))
    batch_size = int(cfg["data"].get("batch_size", 32))
    num_workers = int(cfg["data"].get("num_workers", 4))

    val_ds = MoseiCSDDataset(val_manifest, ablation=ablation, label_thr=label_thr)
    test_ds = MoseiCSDDataset(test_manifest, ablation=ablation, label_thr=label_thr)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=mosei_collate_fn, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=mosei_collate_fn, pin_memory=True)

    # Model + ckpt
    model = build_model(cfg, ablation).to(device)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()

    # Calibrate on VAL (matches your training behavior)
    val_logits, val_y = collect_logits_and_labels(model, val_loader, device)
    T = float(temperature_scale_logits(val_logits, val_y))
    val_logits_cal = val_logits / T

    thr_grid = list(cfg["data"].get("pred_thr_grid", [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]))
    thr_vec = tune_thresholds(val_logits_cal, val_y, grid=thr_grid, objective="acc")

    # Test inference
    test_logits, test_y = collect_logits_and_labels(model, test_loader, device)
    test_probs = torch.sigmoid(test_logits / T)
    test_pred = apply_thresholds(test_probs, thr_vec.detach().cpu())
    test_true = (test_y > 0.5).int()

    # MAD (per-sample)
    mad_gamma = float(cfg.get("chado", {}).get("mad", {}).get("gamma", 1.0))
    mad = compute_mad_scores(test_probs, gamma=mad_gamma).cpu().numpy()

    # Strata by tertiles
    q1, q2 = np.quantile(mad, [0.33, 0.66])
    masks = {
        "Low (≤33%)": mad <= q1,
        "Mid (33–66%)": (mad > q1) & (mad <= q2),
        "High (≥66%)": mad > q2,
    }

    rows = []
    for name, mask in masks.items():
        acc, wf1, _ = compute_acc_wf1(test_true[mask], test_pred[mask])
        rows.append({
            "Stratum": name,
            "NumSamples": int(mask.sum()),
            "Acc": float(acc) * 100.0,
            "WF1": float(wf1) * 100.0,
            "MAD_mean": float(np.mean(mad[mask])),
            "MAD_std": float(np.std(mad[mask])),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_table, index=False)

    plot_bar(df, "Acc", out_plot_acc, "Accuracy by Ambiguity Stratum (MAD tertiles)")
    plot_bar(df, "WF1", out_plot_wf1, "WF1 by Ambiguity Stratum (MAD tertiles)")

    print("[SAVED]", out_table)
    print("[SAVED]", out_plot_acc + ".png/.pdf")
    print("[SAVED]", out_plot_wf1 + ".png/.pdf")
    print(f"[INFO] Used T={T:.3f}  thr={ [round(float(x),3) for x in thr_vec.tolist()] }")


if __name__ == "__main__":
    main()
