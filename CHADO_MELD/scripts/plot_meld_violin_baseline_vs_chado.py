import os
import yaml
import math
import inspect
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from src.data.meld_dataset import (
    MeldDataset,
    collate_meld,
    EMO_ORDER_7,
    build_label_map_from_order,
)
from src.models.chado_trimodal import CHADOTrimodal


def _resolve_col(df_cols, preferred, fallbacks):
    """Return first existing column name from [preferred] + fallbacks."""
    if preferred in df_cols:
        return preferred
    for c in fallbacks:
        if c in df_cols:
            return c
    raise KeyError(f"None of these columns exist in CSV: {[preferred] + list(fallbacks)}")


def _safe_make_meld_dataset(cfg, split_csv, label_map):
    """
    Build MeldDataset by introspecting its __init__ signature and only passing
    args it actually accepts. This prevents 'got multiple values' errors.
    """
    df0 = pd.read_csv(split_csv, nrows=1)
    cols = list(df0.columns)

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    # Your CSV header shows: text, emotion, audio_path, video_path, utt_id, ...
    text_col = _resolve_col(cols, data_cfg.get("text_col", "text"), ["text", "utterance", "sentence"])
    label_col = _resolve_col(cols, data_cfg.get("label_col", "label"), ["emotion", "label", "emo", "target"])
    audio_col = _resolve_col(cols, data_cfg.get("audio_path_col", "audio_path"), ["audio_path", "audio", "wav_path"])
    video_col = _resolve_col(cols, data_cfg.get("video_path_col", "video_path"), ["video_path", "video", "mp4_path"])
    utt_id_col = _resolve_col(cols, data_cfg.get("utt_id_col", "utt_id"), ["utt_id", "utterance_id", "uid"])

    # Optional columns (if present)
    dialogue_col = None
    if "dialogue_id" in cols:
        dialogue_col = "dialogue_id"
    elif "dialogue" in cols:
        dialogue_col = "dialogue"

    # Defaults
    num_frames = int(data_cfg.get("num_frames", 8))
    frame_size = int(data_cfg.get("frame_size", 224))
    sample_rate = int(data_cfg.get("sample_rate", 16000))
    max_audio_seconds = float(data_cfg.get("max_audio_seconds", 6.0))

    use_text = bool(model_cfg.get("use_text", True))
    use_audio = bool(model_cfg.get("use_audio", True))
    use_video = bool(model_cfg.get("use_video", True))
    text_model_name = str(model_cfg.get("text_model_name", "roberta-base"))

    # Introspect Dataset signature
    sig = inspect.signature(MeldDataset.__init__)
    accepted = set(sig.parameters.keys())

    # Build kwargs safely
    kwargs = {}
    # Common names we have seen across your codebase
    candidates = {
        "csv_path": split_csv,
        "csv": split_csv,
        "csv_file": split_csv,
        "path": split_csv,

        "text_col": text_col,
        "label_col": label_col,
        "audio_path_col": audio_col,
        "audio_col": audio_col,
        "video_path_col": video_col,
        "video_col": video_col,
        "utt_id_col": utt_id_col,
        "dialogue_col": dialogue_col,

        "num_frames": num_frames,
        "frame_size": frame_size,
        "sample_rate": sample_rate,
        "max_audio_seconds": max_audio_seconds,

        "label_map": label_map,
        "use_text": use_text,
        "use_audio": use_audio,
        "use_video": use_video,
        "text_model_name": text_model_name,
    }

    for k, v in candidates.items():
        if k in accepted and v is not None:
            kwargs[k] = v

    # Some implementations take split name, pass if accepted
    if "split" in accepted:
        kwargs["split"] = "test"

    ds = MeldDataset(**kwargs)
    return ds, label_col


def _load_ckpt_state_dict(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model", "model_state_dict"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        # sometimes ckpt is already the state_dict
        # heuristic: keys look like 'base.' / 'module.' etc
        return ckpt
    raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")


def _strip_prefix_if_present(sd, prefix):
    if not any(key.startswith(prefix) for key in sd.keys()):
        return sd
    return {k[len(prefix):]: v for k, v in sd.items()}


@torch.no_grad()
def _predict_probs(cfg, ckpt_path, loader, device):
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    model = CHADOTrimodal(
        text_model_name=str(model_cfg.get("text_model_name", "roberta-base")),
        audio_model_name=str(model_cfg.get("audio_model_name", "facebook/wav2vec2-base")),
        video_model_name=str(model_cfg.get("video_model_name", "google/vit-base-patch16-224-in21k")),
        num_classes=int(data_cfg.get("num_classes", 7)),
        proj_dim=int(model_cfg.get("proj_dim", 256)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        use_text=bool(model_cfg.get("use_text", True)),
        use_audio=bool(model_cfg.get("use_audio", True)),
        use_video=bool(model_cfg.get("use_video", True)),
        use_gated_fusion=bool(model_cfg.get("use_gated_fusion", True)),
        # We only need logits; CHADO switches are irrelevant for plotting
        use_causal=False,
        use_hyperbolic=False,
        use_transport=False,
        use_refinement=False,
    ).to(device)
    model.eval()

    sd = _load_ckpt_state_dict(ckpt_path)

    # common wrappers
    sd = _strip_prefix_if_present(sd, "module.")
    # if checkpoint was saved as CHADO wrapper with "base." prefix
    # try load both ways
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if len(missing) > 0 and all(k.startswith("base.") for k in missing):
        # try stripping "base."
        sd2 = _strip_prefix_if_present(sd, "base.")
        model.load_state_dict(sd2, strict=False)

    all_probs = []
    all_labels = []

    for batch in loader:
        labels = batch.labels.to(device)

        text = {k: v.to(device) for k, v in batch.text_input.items()} if batch.text_input is not None else None
        audio = batch.audio_wave.to(device) if batch.audio_wave is not None else None
        video = batch.video_frames.to(device) if batch.video_frames is not None else None

        logits, _, _ = model(text_input=text, audio_wave=audio, video_frames=video, modality_mask=None)
        probs = torch.softmax(logits, dim=1)

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return probs, labels


def _bootstrap_metrics(labels, probs_a, probs_b, B=500, seed=123, max_lines=120):
    """
    Paired bootstrap (same resample indices for A and B) to create distributions
    for Acc and WF1.
    """
    rng = np.random.default_rng(seed)
    n = labels.shape[0]

    preds_a = probs_a.argmax(axis=1)
    preds_b = probs_b.argmax(axis=1)

    acc_a, acc_b = [], []
    f1_a, f1_b = [], []

    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        y = labels[idx]

        pa = preds_a[idx]
        pb = preds_b[idx]

        acc_a.append((pa == y).mean() * 100.0)
        acc_b.append((pb == y).mean() * 100.0)

        f1_a.append(f1_score(y, pa, average="weighted") * 100.0)
        f1_b.append(f1_score(y, pb, average="weighted") * 100.0)

    acc_a = np.asarray(acc_a); acc_b = np.asarray(acc_b)
    f1_a = np.asarray(f1_a); f1_b = np.asarray(f1_b)

    # For paired lines, keep a subset for readability
    line_idx = np.linspace(0, B - 1, num=min(max_lines, B), dtype=int)

    return {
        "acc": (acc_a, acc_b, line_idx),
        "wf1": (f1_a, f1_b, line_idx),
    }


def _mean_ci(x):
    m = float(np.mean(x))
    lo, hi = np.quantile(x, [0.025, 0.975])
    return m, float(lo), float(hi)


def _plot_violin_pair(out_path, dist, title_prefix="MELD"):
    # dist keys: "acc", "wf1" -> (A, B, line_idx)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=200)

    # --- Accuracy subplot ---
    ax = axes[0]
    a, b, li = dist["acc"]
    parts = ax.violinplot([a, b], positions=[1, 2], widths=0.8, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_alpha(0.25)

    # paired lines
    for i in li:
        ax.plot([1, 2], [a[i], b[i]], alpha=0.25, linewidth=1)

    ma, la, ha = _mean_ci(a)
    mb, lb, hb = _mean_ci(b)
    ax.errorbar([1, 2], [ma, mb], yerr=[[ma - la, mb - lb], [ha - ma, hb - mb]],
                fmt="o", capsize=6, linewidth=2)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Baseline", "CHADO"])
    ax.set_ylabel("Acc (%)")
    ax.set_title(f"{title_prefix} Accuracy: Violin + Mean 95% CI")
    ax.grid(True, alpha=0.25)

    # --- WF1 subplot ---
    ax = axes[1]
    a, b, li = dist["wf1"]
    parts = ax.violinplot([a, b], positions=[1, 2], widths=0.8, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_alpha(0.25)

    for i in li:
        ax.plot([1, 2], [a[i], b[i]], alpha=0.25, linewidth=1)

    ma, la, ha = _mean_ci(a)
    mb, lb, hb = _mean_ci(b)
    ax.errorbar([1, 2], [ma, mb], yerr=[[ma - la, mb - lb], [ha - ma, hb - mb]],
                fmt="o", capsize=6, linewidth=2)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Baseline", "CHADO"])
    ax.set_ylabel("WF1 (%)")
    ax.set_title(f"{title_prefix} WF1: Violin + Mean 95% CI")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/chado_meld.yaml")
    ap.add_argument("--baseline_ckpt", required=True)
    ap.add_argument("--chado_ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--bootstrap", type=int, default=600)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max_lines", type=int, default=120)
    ap.add_argument("--out", default="figures/meld_violin_pair.png")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    label_map = build_label_map_from_order(EMO_ORDER_7)

    data_cfg = cfg.get("data", {})
    if args.split == "test":
        split_csv = data_cfg.get("test_csv", None)
    else:
        split_csv = data_cfg.get("val_csv", None)
    if not split_csv:
        raise ValueError(f"Missing {args.split}_csv in config under data:")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataset + loader
    ds, label_col_used = _safe_make_meld_dataset(cfg, split_csv, label_map)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_meld(
            b,
            ds.tokenizer,
            cfg["model"].get("use_text", True),
            cfg["model"].get("use_audio", True),
            cfg["model"].get("use_video", True),
        ),
        pin_memory=True,
    )

    print(f"[DATA] split_csv={split_csv}")
    print(f"[DATA] label_col_used={label_col_used}")
    print(f"[EVAL] baseline_ckpt={args.baseline_ckpt}")
    print(f"[EVAL] chado_ckpt={args.chado_ckpt}")

    # predict
    probs_base, labels = _predict_probs(cfg, args.baseline_ckpt, loader, device)
    probs_chado, labels2 = _predict_probs(cfg, args.chado_ckpt, loader, device)
    if labels.shape[0] != labels2.shape[0] or not np.all(labels == labels2):
        raise RuntimeError("Label mismatch between baseline and CHADO passes (should not happen).")

    # bootstrap paired
    dist = _bootstrap_metrics(labels, probs_base, probs_chado, B=args.bootstrap, seed=args.seed, max_lines=args.max_lines)

    # plot
    _plot_violin_pair(args.out, dist, title_prefix="MELD")

    # report central estimates
    base_acc = (probs_base.argmax(1) == labels).mean() * 100.0
    chado_acc = (probs_chado.argmax(1) == labels).mean() * 100.0
    base_wf1 = f1_score(labels, probs_base.argmax(1), average="weighted") * 100.0
    chado_wf1 = f1_score(labels, probs_chado.argmax(1), average="weighted") * 100.0

    print("\n=== MELD (single-pass on split) ===")
    print(f"Baseline: Acc={base_acc:.2f} | WF1={base_wf1:.2f}")
    print(f"CHADO   : Acc={chado_acc:.2f} | WF1={chado_wf1:.2f}")
    print(f"[OK] Saved: {args.out}")


if __name__ == "__main__":
    main()
