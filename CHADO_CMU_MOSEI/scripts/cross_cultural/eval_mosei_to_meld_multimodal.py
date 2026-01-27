import os
import json
import argparse
from typing import Dict, Any, List

import torch
from torch.utils.data import Dataset, DataLoader

from src.chado.config import load_yaml
from src.chado.metrics import compute_acc_wf1
from src.chado.report import multilabel_prf_report, format_report

from src.train.train import build_model, forward_model


EMO_NAMES = ["happy", "sad", "angry", "fearful", "disgust", "surprise"]


class MeldCrossDataset(Dataset):
    def __init__(self, jsonl_path: str, ablation: str):
        self.path = jsonl_path
        self.ablation = ablation
        self.rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        y = torch.tensor(r["label"], dtype=torch.float32)

        # Text always provided
        text = r.get("text", "")

        # Load features if available; else zeros (but we will extract them so usually present)
        need_audio = self.ablation in ("TA", "TAV")
        need_video = self.ablation in ("TV", "TAV")

        audio = None
        video = None
        if need_audio:
            ap = r.get("audio_path", None)
            if ap and os.path.exists(ap):
                audio = torch.load(ap, map_location="cpu").float()  # [400,74]
            else:
                audio = torch.zeros(400, 74)

        if need_video:
            vp = r.get("video_path", None)
            if vp and os.path.exists(vp):
                video = torch.load(vp, map_location="cpu").float()  # [400,35]
            else:
                video = torch.zeros(400, 35)

        return {
            "utt_id": r.get("utt_id", str(idx)),
            "text": text,
            "label": y,
            "audio": audio,
            "video": video,
        }


def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {}
    out["utt_id"] = [b["utt_id"] for b in batch]
    out["text"] = [b["text"] for b in batch]
    out["label"] = torch.stack([b["label"] for b in batch], dim=0)

    # Audio/video may be None for T ablation
    if batch[0].get("audio", None) is not None:
        out["audio"] = torch.stack([b["audio"] for b in batch], dim=0)  # [B,400,74]
    else:
        out["audio"] = None

    if batch[0].get("video", None) is not None:
        out["video"] = torch.stack([b["video"] for b in batch], dim=0)  # [B,400,35]
    else:
        out["video"] = None

    return out


@torch.no_grad()
def collect_logits_and_labels(model, loader, device):
    model.eval()
    L, Y = [], []
    for batch in loader:
        y = batch["label"].to(device)
        logits, _ = forward_model(model, batch, device)
        L.append(logits.detach().cpu())
        Y.append(y.detach().cpu())
    return torch.cat(L, dim=0), torch.cat(Y, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ablation", required=True, choices=["T", "TA", "TV", "TAV"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--T", type=float, default=1.3, help="Use MOSEI calibration temperature (default 1.3)")
    ap.add_argument("--thr", type=float, default=0.5, help="Use fixed threshold (cross-domain; no tuning leakage)")
    args = ap.parse_args()

    cfg = load_yaml(args.cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg, args.ablation).to(device)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    ds = MeldCrossDataset(args.manifest, ablation=args.ablation)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate)

    logits, y = collect_logits_and_labels(model, loader, device)

    # Cross-domain: DO NOT tune thresholds on MELD (avoid leakage)
    T = float(args.T)
    probs = torch.sigmoid(logits / T)
    thr = torch.full((6,), float(args.thr))
    pred = (probs >= thr).int()
    true = (y > 0.5).int()

    acc, wf1, conf = compute_acc_wf1(true, pred)

    print(f"\n[MOSEI→MELD] Ablation={args.ablation}")
    print(f"[CROSS-TEST] Acc={acc*100:.2f}%  WF1={wf1*100:.2f}  T={T:.3f}")
    print("Fixed thresholds:", [float(args.thr)] * 6)

    print("\nPer-emotion confusion counts: [TP, FP, FN, TN]")
    for i, n in enumerate(EMO_NAMES):
        tp, fp, fn, tn = conf[i].tolist()
        print(f"  {n:8s}: TP={tp:5d} FP={fp:5d} FN={fn:5d} TN={tn:5d}")

    rows, summary, _ = multilabel_prf_report(true, pred, class_names=EMO_NAMES)
    print("\nPer-class metrics (multilabel):")
    print(format_report(rows, summary))
    print(f"\nWeighted Avg F1: {summary['weighted_avg']['f1']*100:.2f}")

    # Save a compact CSV row
    os.makedirs("outputs/analysis/cross", exist_ok=True)
    out_csv = f"outputs/analysis/cross/mosei_to_meld_{args.ablation}.csv"
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("ablation,acc,wf1,T,thr\n")
        f.write(f"{args.ablation},{acc:.6f},{wf1:.6f},{T:.3f},{args.thr:.3f}\n")
    print(f"[SAVED] {out_csv}")


if __name__ == "__main__":
    main()
