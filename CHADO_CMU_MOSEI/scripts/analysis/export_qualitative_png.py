import os
import argparse
import textwrap
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.chado.config import load_yaml
from src.datasets.mosei_dataset import MoseiCSDDataset, mosei_collate_fn
from src.train.train import build_model, apply_thresholds

EMO_NAMES = ["happy", "sad", "angry", "fearful", "disgust", "surprise"]


@torch.no_grad()
def forward_logits(model, batch, device):
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    out = model(inputs)
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def wrap(s, width=62, max_lines=2):
    s = (s or "").strip()
    lines = textwrap.wrap(s, width=width)
    lines = lines[:max_lines]
    if len(lines) == 0:
        return ""
    if len(textwrap.wrap(s, width=width)) > max_lines:
        lines[-1] = lines[-1].rstrip(".") + " ..."
    return "\n".join(lines)


def audio_tag_from_energy(e):
    if e is None:
        return "NA"
    if e < 0.60:
        return "Neutral"
    if e < 0.80:
        return "Neutral"
    return "Expressive"


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def load_thumb(frames_dir, utt_id):
    if not frames_dir:
        return None
    for ext in (".jpg", ".png", ".jpeg", ".webp"):
        p = os.path.join(frames_dir, str(utt_id) + ext)
        if os.path.exists(p):
            try:
                import matplotlib.image as mpimg
                return mpimg.imread(p)
            except Exception:
                return None
    return None


def single_label_from_multihot(y01):
    # If multi-hot has at least one positive -> pick first positive.
    # If none -> return "none".
    idx = np.where(y01.astype(np.int32) == 1)[0]
    if len(idx) == 0:
        return "none", None
    return EMO_NAMES[int(idx[0])], int(idx[0])


def single_label_from_probs(probs):
    j = int(np.argmax(probs))
    return EMO_NAMES[j], j


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="src/configs/chado_mosei_emo6.yaml")
    ap.add_argument("--ablation", default="TAV", choices=["T", "TA", "TV", "TAV"])
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--ckpt", default=None, help="defaults to outputs/checkpoints/{ablation}_best.pt")
    ap.add_argument("--thr_path", default=None, help="optional best thresholds file; otherwise 0.5")
    ap.add_argument("--n_true", type=int, default=5)
    ap.add_argument("--n_wrong", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_text_len", type=int, default=5)
    ap.add_argument("--out", default=None, help="output png path")
    ap.add_argument("--frames_dir", default=None,
                    help="Optional: directory containing thumbnails named {utt_id}.jpg or {utt_id}.png")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_yaml(args.cfg)
    proj = cfg["experiment"]["project_root"]
    manifest = f"{proj}/data/manifests/mosei_{args.split}.jsonl"

    ckpt = args.ckpt or f"outputs/checkpoints/{args.ablation}_best.pt"
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    # thresholds (optional)
    if args.thr_path and os.path.exists(args.thr_path):
        thr = torch.load(args.thr_path, map_location="cpu")
        if isinstance(thr, dict) and "thr" in thr:
            thr = thr["thr"]
        best_thr = torch.tensor(thr, dtype=torch.float32).view(6)
    else:
        best_thr = torch.full((6,), 0.5, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_thr = float(cfg["data"].get("label_thr", 0.0))
    bs = int(cfg["data"].get("batch_size", 32))
    nw = int(cfg["data"].get("num_workers", 4))

    ds = MoseiCSDDataset(manifest, ablation=args.ablation, label_thr=label_thr)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw,
                        collate_fn=mosei_collate_fn, pin_memory=True)

    model = build_model(cfg, args.ablation).to(device)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()

    examples = []
    for batch in loader:
        utt_ids = batch.get("utt_id", None)
        texts = batch.get("text", None)

        logits = forward_logits(model, batch, device)
        probs = torch.sigmoid(logits).detach().cpu().numpy()

        y = batch["label"].detach().cpu()
        y_true = (y > 0.5).int().numpy()
        y_pred = apply_thresholds(torch.tensor(probs), best_thr).numpy()

        # exact match (multilabel)
        exact_ok = (y_pred == y_true).all(axis=1)

        # confidence = max prob
        conf = probs.max(axis=1)

        # audio energy summary
        a_energy = None
        if "audio" in batch and torch.is_tensor(batch["audio"]):
            a = batch["audio"].detach().cpu().float()
            a_energy = a.abs().mean(dim=(1, 2)).numpy()

        B = probs.shape[0]
        for i in range(B):
            txt = texts[i] if (texts is not None) else ""
            txt = (txt or "").strip()
            if len(txt.split()) < args.min_text_len:
                continue

            true_single, _ = single_label_from_multihot(y_true[i])
            pred_single, _ = single_label_from_probs(probs[i])

            # define "single-label correct" for display (your request)
            single_ok = (true_single == pred_single) and (true_single != "none")

            examples.append({
                "utt_id": utt_ids[i] if utt_ids is not None else f"idx_{len(examples)}",
                "text": txt,
                "true_single": true_single,
                "pred_single": pred_single,
                "single_ok": bool(single_ok),
                "conf": float(conf[i]),
                "a_energy": safe_float(a_energy[i]) if a_energy is not None else None,
            })

    if len(examples) == 0:
        raise RuntimeError("No examples found. Try lowering --min_text_len or check manifests.")

    # pick top confident single-label correct + wrong
    correct = sorted([e for e in examples if e["single_ok"]], key=lambda x: -x["conf"])[: args.n_true]
    wrong = sorted([e for e in examples if not e["single_ok"]], key=lambda x: -x["conf"])[: args.n_wrong]

    if len(correct) < args.n_true or len(wrong) < args.n_wrong:
        print(f"[WARN] Available: correct={len(correct)} wrong={len(wrong)} (requested {args.n_true}/{args.n_wrong})")

    picks = correct + wrong
    n = len(picks)

    # --- Figure: compact and “paper-like”
    fig_h = 1.45 * n + 1.2
    fig = plt.figure(figsize=(11.5, fig_h))
    gs = fig.add_gridspec(nrows=n, ncols=2, width_ratios=[1.55, 5.45], hspace=0.55, wspace=0.20)

    fig.suptitle(
        f"CMU-MOSEI Qualitative Cases ({args.ablation}, {args.split}) — Top {len(correct)} Correct + {len(wrong)} Wrong",
        fontsize=15,
        y=0.995
    )

    for idx, ex in enumerate(picks):
        axL = fig.add_subplot(gs[idx, 0])
        axR = fig.add_subplot(gs[idx, 1])
        axL.set_axis_off()
        axR.set_axis_off()

        status = "CORRECT" if ex["single_ok"] else "WRONG"

        # --- LEFT: “image” box + “audio” box + labels
        axL.text(0.02, 0.98, f"Sample {idx+1}", fontsize=12, weight="bold", va="top")
        axL.text(0.02, 0.83, status, fontsize=11, weight="bold", va="top")

        # Frame thumbnail (if available) else placeholder
        thumb = load_thumb(args.frames_dir, ex["utt_id"])
        x0, y0, w, h = 0.06, 0.36, 0.88, 0.40
        if thumb is not None:
            axL.imshow(thumb, extent=(x0, x0+w, y0, y0+h))
            axL.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, linewidth=1.8))
        else:
            axL.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, linewidth=1.8))
            axL.text(x0 + w/2, y0 + h/2, "Frame\n(unavailable)", ha="center", va="center", fontsize=10)

        # Audio icon panel (not waveform; we only have features)
        axL.add_patch(plt.Rectangle((x0, 0.08), w, 0.18, fill=False, linewidth=1.8))
        a_tag = audio_tag_from_energy(ex["a_energy"])
        a_e = ex["a_energy"]
        if a_e is None:
            axL.text(x0 + w/2, 0.17, "Audio: NA", ha="center", va="center", fontsize=10)
        else:
            axL.text(x0 + w/2, 0.19, f"Audio: {a_tag}", ha="center", va="center", fontsize=10)
            axL.text(x0 + w/2, 0.11, f"Energy={a_e:.3f}", ha="center", va="center", fontsize=9)

        # --- RIGHT: Text + single truth/pred
        t = wrap(ex["text"], width=70, max_lines=2)

        axR.text(0.00, 0.94, f"Text:  {t}", fontsize=12, va="top")
        axR.text(0.00, 0.54, f"Truth:  {ex['true_single']}", fontsize=12, va="top")
        axR.text(0.28, 0.54, f"Predict:  {ex['pred_single']}", fontsize=12, va="top")

        # dotted border like sample
        for ax in (axL, axR):
            ax.add_patch(
                plt.Rectangle(
                    (-0.02, -0.08), 1.04, 1.16,
                    fill=False, linestyle="--", linewidth=1.4,
                    transform=ax.transAxes
                )
            )

    os.makedirs("outputs/analysis/qualitative", exist_ok=True)
    out = args.out or f"outputs/analysis/qualitative/{args.ablation}_{args.split}_qualitative_5true_5wrong_singlelabel.png"
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out, dpi=300)
    print("[SAVED]", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
