import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.chado.config import load_yaml
from src.train.train import build_model
from src.datasets.mosei_dataset import MoseiCSDDataset, mosei_collate_fn
from src.chado.metrics import compute_acc_wf1
from src.chado.trainer_utils import tune_thresholds, apply_thresholds

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

def plot_sensitivity(df, metric, out_base, title):
    plt.figure(figsize=(6.6, 4.2))
    for obj in sorted(df["thr_objective"].unique()):
        sub = df[df["thr_objective"] == obj].sort_values("T")
        plt.plot(sub["T"].values, sub[metric].values, marker="o", linewidth=2, label=f"thr_tune={obj}")
    plt.xlabel("Temperature T")
    plt.ylabel(metric + " (%)")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_base + ".png", dpi=300)
    plt.savefig(out_base + ".pdf", dpi=300)
    plt.close()

def main():
    cfg_path = "src/configs/chado_mosei_emo6.yaml"
    ablation = "TAV"
    ckpt_path = f"outputs/checkpoints/{ablation}_best.pt"

    os.makedirs("outputs/analysis/tables", exist_ok=True)
    os.makedirs("outputs/analysis/plots", exist_ok=True)

    out_table = f"outputs/analysis/tables/{ablation}_hparam_sensitivity.csv"
    out_acc = f"outputs/analysis/plots/{ablation}_sensitivity_acc"
    out_wf1 = f"outputs/analysis/plots/{ablation}_sensitivity_wf1"

    cfg = load_yaml(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    model = build_model(cfg, ablation).to(device)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()

    val_logits, val_y = collect_logits_and_labels(model, val_loader, device)
    test_logits, test_y = collect_logits_and_labels(model, test_loader, device)

    thr_grid = list(cfg["data"].get("pred_thr_grid", [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]))
    T_values = [1.0, 1.1, 1.2, 1.3, 1.4]
    objectives = ["acc", "wf1"]

    rows = []
    for obj in objectives:
        for T in T_values:
            vlog = val_logits / T
            thr = tune_thresholds(vlog, val_y, grid=thr_grid, objective=obj)

            probs = torch.sigmoid(test_logits / T)
            pred = apply_thresholds(probs, thr.detach().cpu())
            true = (test_y > 0.5).int()

            acc, wf1, _ = compute_acc_wf1(true, pred)
            rows.append({
                "thr_objective": obj,
                "T": float(T),
                "Acc": float(acc) * 100.0,
                "WF1": float(wf1) * 100.0,
                "thr_mean": float(thr.mean().item()),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_table, index=False)

    plot_sensitivity(df, "Acc", out_acc, "Post-hoc sensitivity (Accuracy) vs Temperature")
    plot_sensitivity(df, "WF1", out_wf1, "Post-hoc sensitivity (WF1) vs Temperature")

    print("[SAVED]", out_table)
    print("[SAVED]", out_acc + ".png/.pdf")
    print("[SAVED]", out_wf1 + ".png/.pdf")

if __name__ == "__main__":
    main()
