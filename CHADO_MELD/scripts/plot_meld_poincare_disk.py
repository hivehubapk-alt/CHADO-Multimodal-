import os
import yaml
import math
import inspect
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from scipy.stats import entropy
from sklearn.decomposition import PCA

from src.data.meld_dataset import (
    MeldDataset,
    collate_meld,
    EMO_ORDER_7,
    build_label_map_from_order,
)
from src.models.chado_trimodal import CHADOTrimodal


def _load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_ckpt_state_dict(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # common patterns
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model", "model_state_dict"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt = ckpt[k]
                break
    # strip module.
    cleaned = {}
    for k, v in ckpt.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        cleaned[nk] = v
    return cleaned


def build_meld_dataset(cfg, split_csv: str, label_map):
    """
    Builds MeldDataset safely using the dataset __init__ signature.
    Prevents double-passing label_map / text_model_name.
    """
    sig = inspect.signature(MeldDataset.__init__)
    accepted = set(sig.parameters.keys())

    # config keys (use your baseline/chado yaml structure)
    data = cfg.get("data", {})
    model = cfg.get("model", {})

    candidates = {
        "csv_path": split_csv,
        "text_col": data.get("text_col", "text"),
        "label_col": data.get("label_col", "emotion"),
        "audio_path_col": data.get("audio_path_col", "audio_path"),
        "video_path_col": data.get("video_path_col", "video_path"),
        "utt_id_col": data.get("utt_id_col", "utt_id"),
        "num_frames": data.get("num_frames", 16),
        "frame_size": data.get("frame_size", 224),
        "sample_rate": data.get("sample_rate", 16000),
        "max_audio_seconds": data.get("max_audio_seconds", 6.0),
        "label_map": label_map,
        "use_text": bool(model.get("use_text", True)),
        "use_audio": bool(model.get("use_audio", True)),
        "use_video": bool(model.get("use_video", True)),
        # IMPORTANT: only pass text_model_name if dataset expects it
        "text_model_name": model.get("text_model_name", "roberta-base"),
    }

    kwargs = {}
    for k, v in candidates.items():
        if k in accepted and v is not None:
            kwargs[k] = v

    # check required params
    required = [
        p.name for p in sig.parameters.values()
        if p.name != "self"
        and p.default is inspect._empty
        and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    missing = [r for r in required if r not in kwargs]
    if missing:
        raise TypeError(f"MeldDataset missing required args: {missing}. Signature: {sig}")

    ds = MeldDataset(**kwargs)
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
    ap.add_argument("--max_points", type=int, default=1200)
    ap.add_argument("--out", default="figures/meld_poincare_disk.png")
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    label_map = build_label_map_from_order(EMO_ORDER_7)

    split_csv = cfg["data"][f"{args.split}_csv"]
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
            getattr(ds, "tokenizer", None),
            use_text, use_audio, use_video
        ),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build model (match your earlier “baseline-compat” settings)
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
        use_gated_fusion=cfg["model"].get("use_gated_fusion", True),

        # For visualization we can keep CHADO switches off; checkpoint is baseline
        use_causal=False,
        use_hyperbolic=False,
        use_transport=False,
        use_refinement=False,
    ).to(device).eval()

    sd = _load_ckpt_state_dict(args.ckpt)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print(f"[LOAD] ckpt={args.ckpt}")
    print(f"[LOAD] missing={len(missing)} unexpected={len(unexpected)} (strict=False)")

    # Collect logits + a stable embedding
    # Your model forward returns: logits, _, _
    # We will use the SECOND output if it is a tensor; otherwise fallback to logits as embedding.
    all_emb = []
    all_probs = []
    all_y = []

    seen = 0
    for batch in loader:
        y = batch.labels.to(device)

        text = {k: v.to(device) for k, v in batch.text_input.items()} if batch.text_input else None
        audio = batch.audio_wave.to(device) if batch.audio_wave is not None else None
        video = batch.video_frames.to(device) if batch.video_frames is not None else None

        out = m(text_input=text, audio_wave=audio, video_frames=video, modality_mask=None)
        logits = out[0]
        aux = out[1] if len(out) > 1 else None

        probs = torch.softmax(logits, dim=1)

        if isinstance(aux, torch.Tensor) and aux.ndim == 2:
            emb = aux
        else:
            # fallback embedding: logits (still works for PCA + disk)
            emb = logits

        all_emb.append(emb.detach().cpu())
        all_probs.append(probs.detach().cpu())
        all_y.append(y.detach().cpu())

        seen += y.size(0)
        if seen >= args.max_points:
            break

    emb = torch.cat(all_emb, dim=0).numpy()
    probs = torch.cat(all_probs, dim=0).numpy()
    y = torch.cat(all_y, dim=0).numpy()

    # Ambiguity = predictive entropy
    amb = entropy(probs.T)  # per-sample
    # normalize ambiguity for marker sizing
    amb_norm = (amb - amb.min()) / (amb.max() - amb.min() + 1e-12)

    # 2D projection + normalize into unit disk
    z2 = PCA(n_components=2).fit_transform(emb)
    r = np.linalg.norm(z2, axis=1, keepdims=True) + 1e-12
    z2_unit = z2 / np.maximum(r, 1.0)  # squash anything outside radius 1 back to boundary

    # Marker sizes (match the visual style)
    sizes = 40 + 260 * amb_norm  # 40..300

    # Plot
    plt.figure(figsize=(10, 10))

    # Unit circle boundary
    t = np.linspace(0, 2 * math.pi, 600)
    plt.plot(np.cos(t), np.sin(t), linewidth=3)

    # Scatter by class
    id2label = {i: lab for i, lab in enumerate(EMO_ORDER_7)}
    for c in range(len(EMO_ORDER_7)):
        idx = (y == c)
        if idx.sum() == 0:
            continue
        plt.scatter(
            z2_unit[idx, 0],
            z2_unit[idx, 1],
            s=sizes[idx],
            alpha=0.75,
            edgecolors="none",
            label=id2label[c],
        )

    plt.title("Poincaré Disk Embedding (size = ambiguity)", fontsize=16)
    plt.xlabel("Poincaré x (2D projection)", fontsize=12)
    plt.ylabel("Poincaré y (2D projection)", fontsize=12)
    plt.xlim([-1.05, 1.05])
    plt.ylim([-1.05, 1.05])
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper right", frameon=True)
    plt.text(
        -1.02, -1.12,
        "Marker size ∝ ambiguity (higher = more ambiguous)",
        fontsize=12
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    plt.close()

    print(f"[OK] Saved → {args.out}   (points={len(y)})")


if __name__ == "__main__":
    main()
