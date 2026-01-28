import os
import yaml
import math
import inspect
import argparse
import importlib
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.amp import autocast

from sklearn.decomposition import PCA
from scipy.stats import entropy

from src.data.meld_dataset import (
    MeldDataset,
    collate_meld,
    EMO_ORDER_7,
    build_label_map_from_order,
)

# CHADO model (exists in your repo)
from src.models.chado_trimodal import CHADOTrimodal


def extract_state_dict(ckpt_obj):
    """
    Robustly extract model weights from checkpoints like:
      - raw state_dict
      - {"state_dict": ...}
      - {"model": ...}
      - {"model_state_dict": ...}
    """
    if isinstance(ckpt_obj, dict):
        for k in ["state_dict", "model_state_dict", "model"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
    return ckpt_obj


def strip_module_prefix(sd: dict):
    out = {}
    for k, v in sd.items():
        out[k[7:]] = v if k.startswith("module.") else v
    # NOTE: above line wrong if not module.; fix:
    out = {}
    for k, v in sd.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        out[nk] = v
    return out


def is_chado_checkpoint(sd: dict) -> bool:
    # CHADO weights are typically under "base."
    return any(k.startswith("base.") for k in sd.keys())


def try_import_baseline_model():
    """
    Your repo baseline training file is train_baseline.py.
    Baseline model class name can differ. We try common locations.
    """
    candidates = [
        ("src.models.trimodal_baseline", "TriModalBaseline"),
        ("src.models.baseline_trimodal", "TriModalBaseline"),
        ("src.models.meld_baseline", "TriModalBaseline"),
        ("src.models.baseline", "TriModalBaseline"),
        ("src.models.trimodal", "TriModalBaseline"),
        ("src.models.baseline_trimodal", "MeldTriModalBaseline"),
        ("src.models.trimodal_baseline", "MeldTriModalBaseline"),
    ]
    for mod, cls in candidates:
        try:
            m = importlib.import_module(mod)
            if hasattr(m, cls):
                return getattr(m, cls)
        except Exception:
            continue
    return None


def build_meld_dataset(cfg, split_csv, label_map):
    """
    Signature-safe MeldDataset builder to avoid:
      - multiple values for label_map
      - missing required args (text_model_name)
    """
    sig = inspect.signature(MeldDataset.__init__)
    accepted = set(sig.parameters.keys())

    dcfg = cfg["data"]
    mcfg = cfg["model"]

    candidates = {
        "csv_path": split_csv,
        "text_col": dcfg.get("text_col", "text"),
        "label_col": dcfg.get("label_col", "emotion"),
        "audio_path_col": dcfg.get("audio_path_col", "audio_path"),
        "video_path_col": dcfg.get("video_path_col", "video_path"),
        "utt_id_col": dcfg.get("utt_id_col", "utt_id"),
        "num_frames": dcfg.get("num_frames", 8),
        "frame_size": dcfg.get("frame_size", 224),
        "sample_rate": dcfg.get("sample_rate", 16000),
        "max_audio_seconds": dcfg.get("max_audio_seconds", 6.0),
        "label_map": label_map,
        "use_text": bool(mcfg.get("use_text", True)),
        "use_audio": bool(mcfg.get("use_audio", True)),
        "use_video": bool(mcfg.get("use_video", True)),
        # IMPORTANT: if dataset requires it, provide valid model id
        "text_model_name": mcfg.get("text_model_name", "roberta-base"),
    }

    kwargs = {k: v for k, v in candidates.items() if k in accepted}

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
    if not hasattr(ds, "tokenizer"):
        raise RuntimeError("MeldDataset must have ds.tokenizer for collate_meld().")
    return ds


def ensure_unit_ball(x: np.ndarray, radius=0.98):
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    max_norm = norms.max()
    if max_norm <= 0:
        return x
    return (x / max_norm) * radius


def find_last_linear(model: torch.nn.Module, num_classes: int):
    """
    Find last Linear layer with out_features == num_classes.
    Works even if model doesn't expose .classifier.
    """
    last = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and getattr(mod, "out_features", None) == num_classes:
            last = (name, mod)
    return last  # (name, module) or None


def build_model(cfg, sd: dict):
    """
    Build either CHADO model or baseline model based on checkpoint keys.
    """
    num_classes = int(cfg["data"]["num_classes"])

    if is_chado_checkpoint(sd):
        model = CHADOTrimodal(
            text_model_name=cfg["model"]["text_model_name"],
            audio_model_name=cfg["model"]["audio_model_name"],
            video_model_name=cfg["model"]["video_model_name"],
            num_classes=num_classes,
            proj_dim=cfg["model"]["proj_dim"],
            dropout=cfg["model"]["dropout"],
            use_text=cfg["model"]["use_text"],
            use_audio=cfg["model"]["use_audio"],
            use_video=cfg["model"]["use_video"],
            use_gated_fusion=cfg["model"]["use_gated_fusion"],
            use_causal=bool(cfg.get("chado", {}).get("use_causal", False)),
            use_hyperbolic=bool(cfg.get("chado", {}).get("use_hyperbolic", False)),
            use_transport=bool(cfg.get("chado", {}).get("use_transport", False)),
            use_refinement=bool(cfg.get("chado", {}).get("use_refinement", False)),
        )
        kind = "chado"
        return model, kind

    # baseline
    BaselineCls = try_import_baseline_model()
    if BaselineCls is None:
        raise RuntimeError(
            "Could not import baseline model class. Please search your repo for 'class TriModalBaseline' "
            "and tell me its module path (e.g., src/models/xxx.py)."
        )

    # Try common init signatures for baseline model
    # We will attempt a few patterns safely.
    attempts = []

    attempts.append(dict(
        text_model_name=cfg["model"]["text_model_name"],
        audio_model_name=cfg["model"]["audio_model_name"],
        video_model_name=cfg["model"]["video_model_name"],
        num_classes=num_classes,
        proj_dim=cfg["model"]["proj_dim"],
        dropout=cfg["model"]["dropout"],
        use_text=cfg["model"]["use_text"],
        use_audio=cfg["model"]["use_audio"],
        use_video=cfg["model"]["use_video"],
        use_gated_fusion=cfg["model"]["use_gated_fusion"],
    ))

    attempts.append(dict(
        text_model_name=cfg["model"]["text_model_name"],
        audio_model_name=cfg["model"]["audio_model_name"],
        video_model_name=cfg["model"]["video_model_name"],
        num_classes=num_classes,
        proj_dim=cfg["model"]["proj_dim"],
        dropout=cfg["model"]["dropout"],
    ))

    attempts.append(dict(
        num_classes=num_classes,
    ))

    last_err = None
    for kw in attempts:
        try:
            model = BaselineCls(**kw)
            return model, "baseline"
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Failed to instantiate baseline model with known patterns. Last error: {last_err}")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/chado_meld.yaml")
    ap.add_argument("--ckpt", default="runs/baseline_trimodal_meld_best.pt")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_points", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = yaml.safe_load(open(args.config))
    os.makedirs("figures", exist_ok=True)

    # ---- data
    split_csv = cfg["data"][f"{args.split}_csv"]
    label_map = build_label_map_from_order(EMO_ORDER_7)
    ds = build_meld_dataset(cfg, split_csv, label_map)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda b: collate_meld(
            b, ds.tokenizer,
            bool(cfg["model"]["use_text"]),
            bool(cfg["model"]["use_audio"]),
            bool(cfg["model"]["use_video"]),
        ),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- load ckpt weights
    ckpt_obj = torch.load(args.ckpt, map_location="cpu")
    sd = strip_module_prefix(extract_state_dict(ckpt_obj))

    # ---- build correct model
    model, kind = build_model(cfg, sd)
    model = model.to(device).eval()

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[MODEL] kind={kind}")
    print(f"[LOAD] ckpt={args.ckpt}")
    print(f"[LOAD] missing={len(missing)} unexpected={len(unexpected)} (strict=False)")

    # ---- hook last classifier-like layer (Linear out=num_classes)
    last = find_last_linear(model, int(cfg["data"]["num_classes"]))
    if last is None:
        raise RuntimeError("Could not find final Linear(out=num_classes) to hook for embeddings.")
    last_name, last_mod = last
    print(f"[HOOK] using last linear: {last_name}")

    feats = []

    def hook_fn(module, inp, out):
        x = inp[0]
        feats.append(x.detach().float().cpu())

    h = last_mod.register_forward_hook(hook_fn)

    all_probs = []
    all_labels = []

    for batch in loader:
        labels = batch.labels.to(device)

        text = {k: v.to(device) for k, v in batch.text_input.items()} if batch.text_input else None
        audio = batch.audio_wave.to(device) if batch.audio_wave is not None else None
        video = batch.video_frames.to(device) if batch.video_frames is not None else None

        with autocast("cuda", enabled=bool(args.amp)):
            out = model(
                text_input=text,
                audio_wave=audio,
                video_frames=video,
                modality_mask=None,
            )

        # support both (logits, a, b) and logits only
        logits = out[0] if isinstance(out, (tuple, list)) else out
        probs = torch.softmax(logits, dim=-1).detach().cpu()
        all_probs.append(probs)
        all_labels.append(labels.detach().cpu())

    h.remove()

    probs = torch.cat(all_probs, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    feats = torch.cat(feats, dim=0).numpy()

    # ---- ambiguity = predictive entropy
    amb = entropy(probs.T)

    # ---- subsample
    n = feats.shape[0]
    if args.max_points and n > args.max_points:
        idx = np.random.choice(n, size=args.max_points, replace=False)
        feats, labels, amb, probs = feats[idx], labels[idx], amb[idx], probs[idx]

    # ---- PCA projections
    z2 = ensure_unit_ball(PCA(n_components=2, random_state=args.seed).fit_transform(feats), radius=0.98)
    z3 = ensure_unit_ball(PCA(n_components=3, random_state=args.seed).fit_transform(feats), radius=0.98)

    # ---- marker size
    amb_n = (amb - amb.min()) / (amb.max() - amb.min() + 1e-12)
    sizes = 30 + 170 * amb_n

    # ---- Poincaré disk
    plt.figure(figsize=(9, 9))
    ax = plt.gca()
    theta = np.linspace(0, 2 * np.pi, 512)
    ax.plot(np.cos(theta), np.sin(theta), linewidth=2)

    emo_names = EMO_ORDER_7
    for c in range(int(cfg["data"]["num_classes"])):
        m = (labels == c)
        if not np.any(m):
            continue
        ax.scatter(z2[m, 0], z2[m, 1], s=sizes[m], alpha=0.75, edgecolors="none",
                   label=emo_names[c] if c < len(emo_names) else f"class{c}")

    ax.set_title("Poincaré Disk Embedding (size = ambiguity)")
    ax.set_xlabel("Poincaré x (2D projection)")
    ax.set_ylabel("Poincaré y (2D projection)")
    ax.set_xlim([-1.05, 1.05])
    ax.set_ylim([-1.05, 1.05])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", frameon=True)
    ax.text(
        0.5, -0.08,
        "Marker size ∝ ambiguity (predictive entropy; higher = more ambiguous)",
        transform=ax.transAxes, ha="center", va="top", fontsize=10,
    )

    out_disk = "figures/meld_poincare_disk.png"
    plt.tight_layout()
    plt.savefig(out_disk, dpi=300)
    plt.close()
    print(f"[OK] Saved: {out_disk}")

    # ---- Poincaré ball (3D)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(10, 9))
    ax3 = fig.add_subplot(111, projection="3d")

    u = np.linspace(0, 2 * np.pi, 48)
    v = np.linspace(0, np.pi, 24)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax3.plot_wireframe(xs, ys, zs, rstride=2, cstride=2, linewidth=0.5, alpha=0.25)

    for c in range(int(cfg["data"]["num_classes"])):
        m = (labels == c)
        if not np.any(m):
            continue
        ax3.scatter(z3[m, 0], z3[m, 1], z3[m, 2], s=sizes[m], alpha=0.65, depthshade=True,
                    label=emo_names[c] if c < len(emo_names) else f"class{c}")

    ax3.set_title("Poincaré Ball Embedding (3D PCA; size = ambiguity)")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")
    ax3.set_xlim([-1.05, 1.05])
    ax3.set_ylim([-1.05, 1.05])
    ax3.set_zlim([-1.05, 1.05])
    ax3.legend(loc="upper right")

    out_ball = "figures/meld_poincare_ball.png"
    plt.tight_layout()
    plt.savefig(out_ball, dpi=300)
    plt.close()
    print(f"[OK] Saved: {out_ball}")

    pred = probs.argmax(axis=1)
    acc = (pred == labels).mean()
    print(f"[INFO] split={args.split} points={len(labels)} acc={acc:.4f}")
    print("[DONE]")


if __name__ == "__main__":
    main()
