import os
import json
import argparse
import textwrap
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.chado.config import load_yaml
from src.chado.calibration import temperature_scale_logits
from src.chado.trainer_utils import tune_thresholds, apply_thresholds
from src.datasets.mosei_dataset import MoseiCSDDataset, mosei_collate_fn

EMO_NAMES = ["happy", "sad", "angry", "fearful", "disgust", "surprise"]

def forward_model(model, batch, device):
    # match your train.py behavior (model may return logits or (logits,z))
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    out = model(inputs)
    if isinstance(out, (tuple, list)):
        return out[0]
    return out

@torch.no_grad()
def collect_logits_and_labels(model, loader, device):
    model.eval()
    L, Y, U = [], [], []
    for batch in loader:
        logits = forward_model(model, batch, device)
        y = batch["label"].to(device)
        L.append(logits.detach().cpu())
        Y.append(y.detach().cpu())
        U.extend(batch["utt_id"])
    return torch.cat(L, 0), torch.cat(Y, 0), U

def decode_text(x):
    # your manifest text_raw is often list of bytes tokens
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        toks = []
        for t in x:
            try:
                if isinstance(t, (bytes, bytearray)):
                    toks.append(t.decode("utf-8", errors="ignore"))
                else:
                    toks.append(str(t))
            except Exception:
                continue
        s = " ".join([w for w in toks if w not in ("sp", "b'sp'")])
        s = s.replace("  ", " ").strip()
        return s
    return str(x)

def labels_to_list(y01):
    idx = (y01 > 0.5).nonzero(as_tuple=False).view(-1).tolist()
    return [EMO_NAMES[i] for i in idx]

def safe_tensor(x, name):
    if not torch.is_tensor(x):
        raise RuntimeError(f"Expected tensor for {name}, got {type(x)}")
    return x

def audio_energy_curve(audio_400x74):
    # [T,74] -> energy per frame (L2)
    a = audio_400x74.float()
    e = torch.sqrt(torch.clamp((a * a).sum(dim=1), min=1e-12))
    e = e.cpu().numpy()
    return e

def video_mean_abs_bar(video_400x35):
    v = video_400x35.float()
    m = v.abs().mean(dim=0).cpu().numpy()  # [35]
    return m

def pick_examples(test_true01, test_pred01, utt_ids, n_each=5):
    # Correct = exact match across all 6 labels
    correct_mask = (test_true01 == test_pred01).all(dim=1)
    wrong_mask = ~correct_mask

    correct_idx = correct_mask.nonzero(as_tuple=False).view(-1).tolist()
    wrong_idx = wrong_mask.nonzero(as_tuple=False).view(-1).tolist()

    # deterministic selection: first N
    correct_idx = correct_idx[:n_each]
    wrong_idx = wrong_idx[:n_each]
    return correct_idx, wrong_idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="e.g., src/configs/chado_mosei_emo6.yaml")
    ap.add_argument("--ablation", default="TAV", choices=["T", "TA", "TV", "TAV"])
    ap.add_argument("--ckpt", default="outputs/checkpoints/TAV_best.pt")
    ap.add_argument("--n_each", type=int, default=5)
    ap.add_argument("--out", default="outputs/analysis/qualitative/feature_trimodal_TAV.png")
    ap.add_argument("--max_text_chars", type=int, default=260)
    args = ap.parse_args()

    cfg = load_yaml(args.cfg)
    proj = cfg["experiment"]["project_root"]

    val_manifest = f"{proj}/data/manifests/mosei_val.jsonl"
    test_manifest = f"{proj}/data/manifests/mosei_test.jsonl"

    label_thr = float(cfg["data"].get("label_thr", 0.0))
    bs = int(cfg["data"].get("batch_size", 32))
    nw = int(cfg["data"].get("num_workers", 2))
    thr_grid = list(cfg["data"].get(
        "pred_thr_grid",
        [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
    ))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Import your build_model from train.py (keeps architecture identical)
    from src.train.train import build_model  # uses BaselineFusion + flags

    model = build_model(cfg, args.ablation).to(device)

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    state = torch.load(args.ckpt, map_location="cpu")
    # ckpt saved as state_dict of underlying module in your train.py
    model.load_state_dict(state, strict=True)

    # Datasets
    val_ds = MoseiCSDDataset(val_manifest, ablation=args.ablation, label_thr=label_thr)
    test_ds = MoseiCSDDataset(test_manifest, ablation=args.ablation, label_thr=label_thr)

    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=nw,
        collate_fn=mosei_collate_fn, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=bs, shuffle=False, num_workers=nw,
        collate_fn=mosei_collate_fn, pin_memory=True
    )

    # ---- Calibrate + tune thresholds on VAL (matches your pipeline) ----
    val_logits, val_y, _ = collect_logits_and_labels(model, val_loader, device)
    T = float(temperature_scale_logits(val_logits, val_y))
    val_logits_cal = val_logits / T
    thr_vec = tune_thresholds(val_logits_cal, val_y, grid=thr_grid, objective="acc")

    # ---- Predict on TEST ----
    test_logits, test_y, test_utts = collect_logits_and_labels(model, test_loader, device)
    test_probs = torch.sigmoid(test_logits / T)
    test_pred = apply_thresholds(test_probs, thr_vec.detach().cpu())

    test_true01 = (test_y > 0.5).int()
    test_pred01 = test_pred.int()

    correct_idx, wrong_idx = pick_examples(test_true01, test_pred01, test_utts, n_each=args.n_each)

    # For text/audio/video visualization, we need to re-index examples.
    # We will load raw records from manifest to get text_raw easily (and avoid extra dataset calls).
    recs = []
    with open(test_manifest, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))

    # Build a mapping utt_id -> record
    id2rec = {r.get("utt_id"): r for r in recs if r.get("utt_id") is not None}

    # To get audio/video tensors for each selected example, we index the dataset directly.
    # MoseiCSDDataset supports __getitem__ returning dict with audio/video already padded.
    def fetch_example(i):
        item = test_ds[i]
        utt = item["utt_id"]
        rec = id2rec.get(utt, {})
        text = decode_text(rec.get("text_raw")) if "text_raw" in rec else decode_text(item.get("text"))
        audio = item.get("audio", None)
        video = item.get("video", None)
        return utt, text, audio, video

    # Prepare figure (10 rows x 2 columns: left=Correct, right=Wrong)
    n = args.n_each
    rows = n
    fig_h = max(10, rows * 2.6)
    fig, axes = plt.subplots(rows, 2, figsize=(18, fig_h))
    if rows == 1:
        axes = np.array([axes])

    def draw_cell(ax, title, utt, text, true_list, pred_list, true_single, pred_single, audio, video):
        ax.axis("off")

        # Text block
        text = (text or "").strip()
        if len(text) > args.max_text_chars:
            text = text[:args.max_text_chars].rstrip() + "..."

        header = (
            f"{title}\n"
            f"utt_id: {utt}\n"
            f"True: {', '.join(true_list) if true_list else 'none'}\n"
            f"Pred: {', '.join(pred_list) if pred_list else 'none'}\n"
            f"Single True: {true_single}   |   Single Pred: {pred_single}\n"
        )

        # Put header and text at top
        ax.text(0.01, 0.98, header, va="top", ha="left", fontsize=10, family="monospace")
        ax.text(0.01, 0.62, "\n".join(textwrap.wrap(text, width=110)), va="top", ha="left", fontsize=10)

        # Small inset plots for audio/video features
        # audio energy curve
        if audio is not None and torch.is_tensor(audio):
            a = safe_tensor(audio, "audio")  # [400,74]
            e = audio_energy_curve(a)
            ax_a = ax.inset_axes([0.02, 0.07, 0.46, 0.18])
            ax_a.plot(e)
            ax_a.set_title("Audio (COVAREP) energy", fontsize=9)
            ax_a.set_xticks([])
            ax_a.set_yticks([])

        # video mean abs bar
        if video is not None and torch.is_tensor(video):
            v = safe_tensor(video, "video")  # [400,35]
            m = video_mean_abs_bar(v)
            ax_v = ax.inset_axes([0.52, 0.07, 0.46, 0.18])
            ax_v.bar(np.arange(len(m)), m)
            ax_v.set_title("Video (FACET) mean|x| per dim", fontsize=9)
            ax_v.set_xticks([])
            ax_v.set_yticks([])

    for r in range(rows):
        # Correct
        i = correct_idx[r] if r < len(correct_idx) else None
        if i is not None:
            utt, text, audio, video = fetch_example(i)
            tlist = labels_to_list(test_true01[i])
            plist = labels_to_list(test_pred01[i])
            true_single = EMO_NAMES[int(torch.argmax(test_true01[i]).item())] if int(test_true01[i].sum().item()) > 0 else "none"
            pred_single = EMO_NAMES[int(torch.argmax(test_probs[i]).item())]
            draw_cell(axes[r, 0], "CORRECT", utt, text, tlist, plist, true_single, pred_single, audio, video)
        else:
            axes[r, 0].axis("off")

        # Wrong
        j = wrong_idx[r] if r < len(wrong_idx) else None
        if j is not None:
            utt, text, audio, video = fetch_example(j)
            tlist = labels_to_list(test_true01[j])
            plist = labels_to_list(test_pred01[j])
            true_single = EMO_NAMES[int(torch.argmax(test_true01[j]).item())] if int(test_true01[j].sum().item()) > 0 else "none"
            pred_single = EMO_NAMES[int(torch.argmax(test_probs[j]).item())]
            draw_cell(axes[r, 1], "WRONG", utt, text, tlist, plist, true_single, pred_single, audio, video)
        else:
            axes[r, 1].axis("off")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=220, bbox_inches="tight")
    print(f"[SAVED] {args.out}")
    print(f"[INFO] Calibration T={T:.3f} | thresholds={ [round(float(x),3) for x in thr_vec.tolist()] }")

if __name__ == "__main__":
    main()
