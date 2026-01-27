import os, argparse, json
import torch
from torch.utils.data import DataLoader

from src.chado.config import load_yaml
from src.chado.calibration import temperature_scale_logits
from src.chado.trainer_utils import tune_thresholds, apply_thresholds
from src.chado.metrics import compute_acc_wf1
from src.chado.report import multilabel_prf_report, format_report

from src.train.train import build_model, forward_model

EMO_NAMES = ["happy", "sad", "angry", "fearful", "disgust", "surprise"]

class CrossMeldJSONL(torch.utils.data.Dataset):
    def __init__(self, jsonl_path: str):
        self.items = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.items.append(json.loads(line))

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        r = self.items[idx]
        return {
            "utt_id": r.get("utt_id", str(idx)),
            "text": r.get("text_raw", ""),
            "label": torch.tensor(r["label"], dtype=torch.float32),
            "audio": None,
            "video": None,
        }

def collate(batch):
    utt = [b["utt_id"] for b in batch]
    txt = [b["text"] for b in batch]
    y = torch.stack([b["label"] for b in batch], dim=0)
    return {"utt_id": utt, "text": txt, "label": y, "audio": None, "video": None}

@torch.no_grad()
def collect_logits_and_labels(model, loader, device):
    model.eval()
    L, Y = [], []
    for batch in loader:
        y = batch["label"].to(device)
        out = forward_model(model, batch, device)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        L.append(logits.detach().cpu())
        Y.append(y.detach().cpu())
    return torch.cat(L, 0), torch.cat(Y, 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ablation", required=True, choices=["T","TA","TV","TAV"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    cfg = load_yaml(args.cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg, args.ablation).to(device)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()

    ds = CrossMeldJSONL(args.manifest)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    logits, y = collect_logits_and_labels(model, loader, device)

    # Calibration + thresholds (same logic as train.py)
    T = float(temperature_scale_logits(logits, y))
    logits_cal = logits / T
    probs = torch.sigmoid(logits_cal)

    thr_grid = cfg["data"].get("pred_thr_grid", [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5])
    thr = tune_thresholds(logits_cal, y, grid=thr_grid, objective="acc")
    pred = apply_thresholds(probs, thr)

    true = (y > 0.5).int()
    acc, wf1, conf = compute_acc_wf1(true, pred)

    print(f"\n[MOSEI→MELD] Ablation={args.ablation}")
    print(f"[CROSS-TEST] Acc={acc*100:.2f}%  WF1={wf1*100:.2f}  T={T:.3f}")
    print("Tuned thresholds:", [round(float(x),3) for x in thr.tolist()])

    print("\nPer-emotion confusion counts: [TP, FP, FN, TN]")
    for i, n in enumerate(EMO_NAMES):
        tp, fp, fn, tn = conf[i].tolist()
        print(f"  {n:8s}: TP={tp:5d} FP={fp:5d} FN={fn:5d} TN={tn:5d}")

    rows, summary, _ = multilabel_prf_report(true, pred, class_names=EMO_NAMES)
    print(format_report(rows, summary))
    print(f"\nWeighted Avg F1: {summary['weighted_avg']['f1']*100:.2f}")

if __name__ == "__main__":
    main()
