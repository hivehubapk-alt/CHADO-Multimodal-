import os
import yaml
import inspect
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from scipy.stats import entropy

from src.data.meld_dataset import (
    MeldDataset,
    collate_meld,
    EMO_ORDER_7,
    build_label_map_from_order,
)
from src.models.chado_trimodal import CHADOTrimodal


def _extract_state_dict(obj):
    """
    Robustly extract a usable state_dict from different checkpoint formats.
    """
    if isinstance(obj, dict):
        for k in ["state_dict", "model", "model_state_dict", "net"]:
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break

    if not isinstance(obj, dict):
        raise RuntimeError("Checkpoint is not a dict and has no state_dict-like content.")

    # strip "module."
    cleaned = {}
    for kk, vv in obj.items():
        nk = kk[7:] if kk.startswith("module.") else kk
        cleaned[nk] = vv
    return cleaned


def build_meld_dataset(cfg, split_csv, label_map):
    """
    Build MeldDataset safely by matching __init__ signature
    (prevents double-passing args like label_map/text_model_name).
    """
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    # MELD-safe defaults
    text_col = data_cfg.get("text_col", "text")
    label_col = data_cfg.get("label_col", "emotion")
    audio_path_col = data_cfg.get("audio_path_col", "audio_path")
    video_path_col = data_cfg.get("video_path_col", "video_path")
    utt_id_col = data_cfg.get("utt_id_col", "utt_id")

    sig = inspect.signature(MeldDataset.__init__)
    accepted = set(sig.parameters.keys())  # includes "self"

    candidates = {
        "csv_path": split_csv,
        "text_col": text_col,
        "label_col": label_col,
        "audio_path_col": audio_path_col,
        "video_path_col": video_path_col,
        "utt_id_col": utt_id_col,
        "num_frames": data_cfg.get("num_frames", 8),
        "frame_size": data_cfg.get("frame_size", 224),
        "sample_rate": data_cfg.get("sample_rate", 16000),
        "max_audio_seconds": data_cfg.get("max_audio_seconds", 6.0),
        "label_map": label_map,
        "use_text": bool(model_cfg.get("use_text", True)),
        "use_audio": bool(model_cfg.get("use_audio", True)),
        "use_video": bool(model_cfg.get("use_video", True)),
        "text_model_name": model_cfg.get("text_model_name", "roberta-base"),
        # tolerate alternate arg names across repos
        "hf_model_name": model_cfg.get("text_model_name", "roberta-base"),
        "bert_name": model_cfg.get("text_model_name", "roberta-base"),
    }

    kwargs = {k: v for k, v in candidates.items() if k in accepted and v is not None}

    # Validate required args
    required = [
        p.name for p in sig.parameters.values()
        if p.name != "self"
        and p.default is inspect._empty
        and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    missing = [r for r in required if r not in kwargs]
    if missing:
        raise TypeError(
            f"MeldDataset missing required args: {missing}\n"
            f"Signature: {sig}\n"
            f"Provided keys: {sorted(list(kwargs.keys()))}\n"
            f"Config data keys: {list(data_cfg.keys())}"
        )

    ds = MeldDataset(**kwargs)

    if bool(model_cfg.get("use_text", True)) and not hasattr(ds, "tokenizer"):
        raise AttributeError("MeldDataset has no tokenizer but use_text=true; collate will fail.")

    return ds


@torch.no_grad()
def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/chado_meld.yaml")
    ap.add_argument("--ckpt", default="runs/baseline_trimodal_meld_best.pt")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_batches", type=int, default=0, help="0 = no limit")
    ap.add_argument("--out", default="figures/meld_performance_vs_ambiguity.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = yaml.safe_load(open(args.config))
    label_map = build_label_map_from_order(EMO_ORDER_7)

    # pick csv by split
    data_cfg = cfg["data"]
    if args.split == "train":
        split_csv = data_cfg["train_csv"]
    elif args.split == "val":
        split_csv = data_cfg["val_csv"]
    else:
        split_csv = data_cfg["test_csv"]

    # dataset + loader
    ds = build_meld_dataset(cfg, split_csv, label_map)

    use_text = bool(cfg["model"].get("use_text", True))
    use_audio = bool(cfg["model"].get("use_audio", True))
    use_video = bool(cfg["model"].get("use_video", True))

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda b: collate_meld(
            b,
            ds.tokenizer if use_text else None,
            use_text,
            use_audio,
            use_video,
        ),
    )

    # model
    m = CHADOTrimodal(
        text_model_name=cfg["model"]["text_model_name"],
        audio_model_name=cfg["model"]["audio_model_name"],
        video_model_name=cfg["model"]["video_model_name"],
        num_classes=cfg["data"]["num_classes"],
        proj_dim=cfg["model"]["proj_dim"],
        dropout=cfg["model"]["dropout"],
        use_text=use_text,
        use_audio=use_audio,
        use_video=use_video,
        use_gated_fusion=cfg["model"]["use_gated_fusion"],
        # For this plot we only need prediction quality; keep CHADO toggles off unless you want otherwise
        use_causal=False,
        use_hyperbolic=False,
        use_transport=False,
        use_refinement=False,
    ).to(device)

    # load ckpt
    print(f"[LOAD] ckpt={args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = _extract_state_dict(ckpt)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print(f"[LOAD] missing={len(missing)} unexpected={len(unexpected)} (strict=False)")
    m.eval()

    # collect probabilities
    all_probs = []
    all_labels = []

    for bi, batch in enumerate(loader):
        if args.max_batches and bi >= args.max_batches:
            break

        labels = batch.labels.to(device)

        text = {k: v.to(device) for k, v in batch.text_input.items()} if (use_text and batch.text_input) else None
        audio = batch.audio_wave.to(device) if (use_audio and batch.audio_wave is not None) else None
        video = batch.video_frames.to(device) if (use_video and batch.video_frames is not None) else None

        logits, _, _ = m(text_input=text, audio_wave=audio, video_frames=video, modality_mask=None)
        probs = torch.softmax(logits, dim=1)

        all_probs.append(probs.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    preds = probs.argmax(axis=1)

    # ambiguity score = predictive entropy
    amb = entropy(probs.T)  # shape [N]

    # bucket by quantiles
    q1, q2 = np.quantile(amb, [0.33, 0.66])
    buckets = {
        "Low": amb <= q1,
        "Medium": (amb > q1) & (amb <= q2),
        "High": amb > q2,
    }

    accs, f1s = [], []
    for name in ["Low", "Medium", "High"]:
        idx = buckets[name]
        if idx.sum() == 0:
            accs.append(0.0)
            f1s.append(0.0)
            continue
        acc = (preds[idx] == labels[idx]).mean() * 100.0
        f1w = f1_score(labels[idx], preds[idx], average="weighted") * 100.0
        accs.append(acc)
        f1s.append(f1w)

    # plot
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    x = np.arange(3)
    w = 0.35

    plt.figure(figsize=(7.2, 4.2))
    plt.bar(x - w / 2, accs, w, label="Accuracy")
    plt.bar(x + w / 2, f1s, w, label="Weighted F1")
    plt.xticks(x, ["Low", "Medium", "High"])
    plt.xlabel("Ambiguity Level (Entropy Quantiles)")
    plt.ylabel("Performance (%)")
    plt.title(f"Performance vs Ambiguity (MELD, split={args.split})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    plt.close()

    print("\n=== Performance vs Ambiguity (MELD) ===")
    for i, name in enumerate(["Low", "Medium", "High"]):
        print(f"{name}: Acc={accs[i]:.2f} | WF1={f1s[i]:.2f} | n={int(buckets[name].sum())}")
    print(f"\n[OK] Saved → {args.out}")


if __name__ == "__main__":
    main()
