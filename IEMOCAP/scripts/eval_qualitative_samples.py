#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import random
import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoImageProcessor

from chado_lib.data.iemocap_dataset import IEMOCAPTriModal
from chado_lib.data.collate import collate_fn
from chado_lib.models.chado import CHADO

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
def infer_all(model, loader, device, amp: bool):
    model.eval()
    preds, gold = [], []
    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits, _ = model(batch["input_ids"], batch["attention_mask"], batch["wav"], batch["pixel_values"])
        p = torch.argmax(logits, dim=-1)
        preds.append(p.cpu().numpy())
        gold.append(batch["label"].cpu().numpy())
    return np.concatenate(preds), np.concatenate(gold)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--ablation", default="tri")
    ap.add_argument("--k_true", type=int, default=5)
    ap.add_argument("--k_wrong", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Use a safe writable default: inside outputs, not /mnt/data
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["config"]

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

    preds, gold = infer_all(model, loader, device, args.amp)

    correct_idx = np.where(preds == gold)[0].tolist()
    wrong_idx   = np.where(preds != gold)[0].tolist()

    random.shuffle(correct_idx)
    random.shuffle(wrong_idx)

    sel_true = correct_idx[:args.k_true]
    sel_wrong = wrong_idx[:args.k_wrong]

    df = pd.read_csv(args.test_csv)
    out_rows = []

    for tag, idxs in [("correct", sel_true), ("wrong", sel_wrong)]:
        for i in idxs:
            r = df.iloc[i]
            out_rows.append({
                "group": tag,
                "row_index": int(i),
                "gold": int(gold[i]),
                "pred": int(preds[i]),
                "label_4": str(r.get("label_4","")),
                "transcript": str(r.get("transcript","")),
                "wav_path": str(r.get("wav_path","")),
                "avi_path": str(r.get("avi_path","")),
                "start": float(r.get("start",0.0)),
                "end": float(r.get("end",0.0)),
            })

    out_csv = os.path.join(args.out_dir, "qualitative_samples.csv")
    pd.DataFrame(out_rows).to_csv(out_csv, index=False)
    print("[OK] saved:", out_csv)

if __name__ == "__main__":
    main()
