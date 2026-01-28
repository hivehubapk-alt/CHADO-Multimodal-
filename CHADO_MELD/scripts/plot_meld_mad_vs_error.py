import os
import yaml
import math
import inspect
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr

from src.data.meld_dataset import (
    MeldDataset,
    collate_meld,
    EMO_ORDER_7,
    build_label_map_from_order,
)
from src.models.chado_trimodal import CHADOTrimodal


# ------------------------- utilities -------------------------
def load_ckpt_state_dict(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # common wrappers
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt = ckpt[k]
                break

    # strip "module."
    if isinstance(ckpt, dict):
        cleaned = {}
        for k, v in ckpt.items():
            nk = k[len("module."):] if k.startswith("module.") else k
            cleaned[nk] = v
        return cleaned

    raise ValueError(f"Unsupported checkpoint format at: {ckpt_path}")


def build_dataset_from_cfg(cfg: dict, split_csv: str, label_map: dict):
    """
    Build MeldDataset without causing:
      - multiple values for argument 'label_map'
      - missing required args like 'text_model_name'
    Uses signature filtering to pass only supported kwargs.
    """
    sig = inspect.signature(MeldDataset.__init__)
    accepted = set(sig.parameters.keys())

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    # Defaults consistent with your repo usage
    candidates = {
        "csv_path": split_csv,
        "text_col": data_cfg.get("text_col", "text"),
        "label_col": data_cfg.get("label_col", "emotion"),
        "audio_path_col": data_cfg.get("audio_path_col", "audio_path"),
        "video_path_col": data_cfg.get("video_path_col", "video_path"),
        "utt_id_col": data_cfg.get("utt_id_col", "utt_id"),
        "num_frames": data_cfg.get("num_frames", 8),
        "frame_size": data_cfg.get("frame_size", 224),
        "sample_rate": data_cfg.get("sample_rate", 16000),
        "max_audio_seconds": data_cfg.get("max_audio_seconds", 6.0),
        "label_map": label_map,
        "use_text": model_cfg.get("use_text", True),
        "use_audio": model_cfg.get("use_audio", True),
        "use_video": model_cfg.get("use_video", True),
        # tokenizer model id (some repos require this in dataset init)
        "text_model_name": model_cfg.get("text_model_name", "roberta-base"),
        # alternate names some repos use:
        "hf_model_name": model_cfg.get("text_model_name", "roberta-base"),
        "bert_name": model_cfg.get("text_model_name", "roberta-base"),
    }

    kwargs = {k: v for k, v in candidates.items() if (k in accepted and v is not None)}

    # Check required parameters
    required = [
        p.name for p in sig.parameters.values()
        if p.name != "self"
        and p.default is inspect._empty
        and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    missing = [r for r in required if r not in kwargs]
    if missing:
        raise TypeError(f"MeldDataset missing required args: {missing}. Full signature: {sig}")

    ds = MeldDataset(**kwargs)
    return ds


def compute_mad_from_probs(probs: np.ndarray):
    """
    Fallback MAD ambiguity: mean absolute deviation from the per-sample mean prob.
    probs: [N, C], rows sum to 1
    Returns: [N]
    """
    mu = probs.mean(axis=1, keepdims=True)         # [N,1]
    mad = np.mean(np.abs(probs - mu), axis=1)      # [N]
    return mad


# ------------------------- main -------------------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_points", type=int, default=3000, help="cap points for speed/plot readability")
    ap.add_argument("--out", default="figures/meld_mad_vs_error.png")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))

    # label map
    label_map = build_label_map_from_order(EMO_ORDER_7)

    # pick csv
    csv_key = {"train": "train_csv", "val": "val_csv", "test": "test_csv"}[args.split]
    split_csv = cfg["data"][csv_key]

    # dataset + loader
    ds = build_dataset_from_cfg(cfg, split_csv, label_map)

    def _collate(batch):
        return collate_meld(
            batch,
            getattr(ds, "tokenizer", None),
            cfg["model"]["use_text"],
            cfg["model"]["use_audio"],
            cfg["model"]["use_video"],
        )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=_collate,
        drop_last=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model: CHADOTrimodal with CHADO components OFF (baseline-compatible)
    model = CHADOTrimodal(
        text_model_name=cfg["model"]["text_model_name"],
        audio_model_name=cfg["model"]["audio_model_name"],
        video_model_name=cfg["model"]["video_model_name"],
        num_classes=cfg["data"]["num_classes"],
        proj_dim=cfg["model"]["proj_dim"],
        dropout=cfg["model"]["dropout"],
        use_text=cfg["model"]["use_text"],
        use_audio=cfg["model"]["use_audio"],
        use_video=cfg["model"]["use_video"],
        use_gated_fusion=cfg["model"]["use_gated_fusion"],
        use_causal=False,
        use_hyperbolic=False,
        use_transport=False,
        use_refinement=False,
    ).to(device)

    sd = load_ckpt_state_dict(args.ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[LOAD] ckpt={args.ckpt}")
    print(f"[LOAD] missing={len(missing)} unexpected={len(unexpected)} (strict=False)")

    model.eval()

    all_probs = []
    all_labels = []
    all_preds = []

    n_seen = 0
    for batch in loader:
        labels = batch.labels.to(device)

        text = {k: v.to(device) for k, v in batch.text_input.items()} if batch.text_input else None
        audio = batch.audio_wave.to(device) if batch.audio_wave is not None else None
        video = batch.video_frames.to(device) if batch.video_frames is not None else None

        logits, _, _ = model(
            text_input=text,
            audio_wave=audio,
            video_frames=video,
            modality_mask=None
        )

        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        all_probs.append(probs.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        all_preds.append(preds.detach().cpu().numpy())

        n_seen += labels.numel()
        if args.max_points and n_seen >= args.max_points:
            break

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    preds = np.concatenate(all_preds, axis=0)

    if args.max_points and len(labels) > args.max_points:
        probs = probs[:args.max_points]
        labels = labels[:args.max_points]
        preds = preds[:args.max_points]

    # error rate per sample (MELD single-label => 0/1)
    err = (preds != labels).astype(np.float32)

    # MAD ambiguity per sample
    mad = compute_mad_from_probs(probs)

    # correlations
    pr, pp = pearsonr(mad, err)
    sr, sp = spearmanr(mad, err)

    # regression line
    a, b = np.polyfit(mad, err, deg=1)  # err ~ a*mad + b
    xs = np.linspace(float(mad.min()), float(mad.max()), 200)
    ys = a * xs + b

    # plot
    plt.figure(figsize=(10, 7))
    hb = plt.hexbin(mad, err, gridsize=35, mincnt=1)  # default cmap is fine
    cb = plt.colorbar(hb)
    cb.set_label("Count", fontsize=14)

    plt.plot(xs, ys, linewidth=4)

    plt.title(f"MAD correlates with error on MELD ({args.split.upper()})", fontsize=20)
    plt.xlabel("MAD ambiguity score (per sample)", fontsize=16)
    plt.ylabel("Prediction error rate (fraction of labels wrong)", fontsize=16)

    # stats box (match your example style)
    box_txt = (
        f"Pearson r={pr:.3f}, p<{pp:.1e}\n"
        f"Spearman ρ={sr:.3f}, p<{sp:.1e}"
    )
    plt.text(
        0.03, 0.95, box_txt,
        transform=plt.gca().transAxes,
        va="top", ha="left",
        fontsize=16,
        bbox=dict(boxstyle="round", alpha=0.85)
    )

    plt.grid(True, alpha=0.25)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    plt.close()

    print(f"[OK] Saved → {args.out}")
    print(f"[STATS] Pearson r={pr:.4f} p={pp:.3e} | Spearman ρ={sr:.4f} p={sp:.3e}")
    print(f"[INFO] N={len(labels)} points (capped by --max_points)")


if __name__ == "__main__":
    main()
