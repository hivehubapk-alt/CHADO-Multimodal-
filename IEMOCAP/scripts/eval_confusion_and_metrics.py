#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoImageProcessor

from chado_lib.data.iemocap_dataset import IEMOCAPTriModal, LABEL2ID
from chado_lib.data.collate import collate_fn
from chado_lib.models.chado import CHADO
from chado_lib.metrics.classification import confusion_matrix_np, per_class_prf

LABELS = ["neu", "hap", "ang", "sad"]

def ablation_to_modalities(ablation: str):
    if ablation == "tri": return True, True, True
    if ablation == "text": return True, False, False
    if ablation == "audio": return False, True, False
    if ablation == "vision": return False, False, True
    if ablation == "ta": return True, True, False
    if ablation == "tv": return True, False, True
    if ablation == "av": return False, True, True
    raise ValueError(ablation)

@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    all_p, all_y = [], []
    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(device, non_blocking=True)
        logits, _ = model(batch["input_ids"], batch["attention_mask"], batch["wav"], batch["pixel_values"])
        pred = torch.argmax(logits, dim=-1)
        all_p.append(pred.cpu())
        all_y.append(batch["label"].cpu())
    p = torch.cat(all_p).numpy()
    y = torch.cat(all_y).numpy()
    return y, p

def plot_cm_style(cm, labels, out_png):
    # normalized + same “paper” feel
    cmn = cm.astype(np.float32) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig = plt.figure(figsize=(6.4, 5.6))
    ax = fig.add_subplot(111)
    im = ax.imshow(cmn, interpolation="nearest")
    fig.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix (normalized)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(cmn.shape[0]):
        for j in range(cmn.shape[1]):
            ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i,j] > 0.5 else "black")
    fig.tight_layout()
    fig.savefig(out_png, dpi=240)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_csv", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ablation", default="tri")
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("config", None)
    if cfg is None:
        raise RuntimeError("Checkpoint missing config; re-train with scripts/train_chado_ddp.py from this repo.")

    tok = AutoTokenizer.from_pretrained(cfg["models"]["text_model"], use_fast=False)
    imgp = AutoImageProcessor.from_pretrained(cfg["models"]["vision_model"], use_fast=False)

    ds = IEMOCAPTriModal(args.split_csv, tok, imgp,
                        max_text_len=cfg["data"]["max_text_len"],
                        audio_sec=cfg["data"]["audio_sec"],
                        n_frames=cfg["data"]["n_frames"])
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                         num_workers=4, pin_memory=True, collate_fn=collate_fn)

    use_text, use_audio, use_video = ablation_to_modalities(args.ablation)
    model = CHADO(cfg["models"]["text_model"], cfg["models"]["audio_model"], cfg["models"]["vision_model"],
                  num_classes=4, use_text=use_text, use_audio=use_audio, use_video=use_video)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    device = torch.device(args.device)
    model.to(device)

    y, p = run_eval(model, loader, device)

    cm = confusion_matrix_np(y, p, 4)
    plot_cm_style(cm, LABELS, os.path.join(args.out_dir, "confusion_matrix_norm.png"))

    rows = per_class_prf(cm, LABELS)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "per_class_precision_recall_f1.csv"), index=False)

    # micro/macro precision/recall
    # macro is average of per-class
    macro_prec = float(df["precision"].mean())
    macro_rec  = float(df["recall"].mean())
    macro_f1   = float(df["f1"].mean())
    acc = float((p == y).mean())

    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro Precision: {macro_prec:.4f}\n")
        f.write(f"Macro Recall: {macro_rec:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n")

    print("[OK] saved:", args.out_dir)
    print(f"Accuracy={acc:.4f} MacroP={macro_prec:.4f} MacroR={macro_rec:.4f} MacroF1={macro_f1:.4f}")

if __name__ == "__main__":
    main()
