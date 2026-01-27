#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoImageProcessor

from chado_lib.data.iemocap_dataset import IEMOCAPTriModal
from chado_lib.data.collate import collate_fn
from chado_lib.models.chado import CHADO
from chado_lib.metrics.ambiguity import predictive_entropy, bucket_by_entropy

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
def collect_probs_and_preds(model, loader, device, amp: bool):
    model.eval()
    probs, preds, gold = [], [], []
    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits, _ = model(batch["input_ids"], batch["attention_mask"], batch["wav"], batch["pixel_values"])
        pr = torch.softmax(logits.float(), dim=-1)
        pd_ = torch.argmax(pr, dim=-1)
        probs.append(pr.detach().cpu().numpy())
        preds.append(pd_.detach().cpu().numpy())
        gold.append(batch["label"].detach().cpu().numpy())
    return np.concatenate(probs), np.concatenate(preds), np.concatenate(gold)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ablation", default="tri")
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--low_thr", type=float, default=0.3)
    ap.add_argument("--high_thr", type=float, default=0.7)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("config", None)
    if cfg is None:
        raise RuntimeError("Checkpoint missing config; re-train with scripts/train_chado_ddp.py from this repo.")

    tok = AutoTokenizer.from_pretrained(cfg["models"]["text_model"], use_fast=False)
    imgp = AutoImageProcessor.from_pretrained(cfg["models"]["vision_model"], use_fast=False)

    ds = IEMOCAPTriModal(args.test_csv, tok, imgp,
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

    probs, pred, gold = collect_probs_and_preds(model, loader, device, args.amp)
    ent = predictive_entropy(probs)

    low, mid, high = bucket_by_entropy(ent, args.low_thr, args.high_thr)

    def stats(mask):
        if mask.sum() == 0:
            return dict(N=0, Acc=np.nan, WF1=np.nan, EntropyMean=np.nan)
        y = gold[mask]
        p = pred[mask]
        acc = (p == y).mean()
        # weighted F1 approximation via per-class support-weighted f1
        wf1 = 0.0
        for c in range(4):
            tp = ((p==c)&(y==c)).sum()
            fp = ((p==c)&(y!=c)).sum()
            fn = ((p!=c)&(y==c)).sum()
            prec = tp / (tp + fp + 1e-9)
            rec  = tp / (tp + fn + 1e-9)
            f1   = (2*prec*rec)/(prec+rec+1e-9)
            wf1 += f1 * (y==c).sum()
        wf1 = wf1 / max(1, len(y))
        return dict(N=int(mask.sum()), Acc=float(acc), WF1=float(wf1), EntropyMean=float(ent[mask].mean()))

    rows = []
    rows.append({"Bucket":"Low", **stats(low)})
    rows.append({"Bucket":"Medium", **stats(mid)})
    rows.append({"Bucket":"High", **stats(high)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "ambiguity_stratified.csv"), index=False)

    print("\n=== Ambiguity-Stratified Performance (Predictive Entropy) ===")
    print(f"Thresholds: Low < {args.low_thr}, Medium [{args.low_thr},{args.high_thr}], High > {args.high_thr}")
    print(df.to_string(index=False))

    # Plot bars (ICML-friendly)
    fig = plt.figure(figsize=(7.2, 4.0))
    ax = fig.add_subplot(111)
    ax.bar(df["Bucket"], df["Acc"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Ambiguity (Entropy Strata)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "ambiguity_strata_accuracy.png"), dpi=240)
    plt.close(fig)

    fig = plt.figure(figsize=(7.2, 4.0))
    ax = fig.add_subplot(111)
    ax.bar(df["Bucket"], df["WF1"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Weighted F1")
    ax.set_title("Weighted F1 vs Ambiguity (Entropy Strata)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "ambiguity_strata_wf1.png"), dpi=240)
    plt.close(fig)

    print(f"\n[OK] saved ambiguity outputs -> {args.out_dir}")

if __name__ == "__main__":
    main()
