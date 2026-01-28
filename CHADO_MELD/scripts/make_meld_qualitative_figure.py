import os
import yaml
import textwrap
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from matplotlib import patches
from matplotlib.gridspec import GridSpec
from PIL import Image

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

from torch.utils.data import DataLoader

from src.data.meld_dataset import (
    MeldDataset,
    collate_meld,
    EMO_ORDER_7,
    build_label_map_from_order,
)
from src.models.chado_trimodal import CHADOTrimodal


def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def unwrap_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for k in ["state_dict", "model", "model_state_dict", "net", "weights"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                ckpt_obj = ckpt_obj[k]
                break
    if not isinstance(ckpt_obj, dict):
        raise RuntimeError("Checkpoint is not a dict-like state_dict.")

    cleaned = {}
    for k, v in ckpt_obj.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        cleaned[nk] = v
    return cleaned


def load_video_thumbnail(video_path: str, out_size=(220, 140)):
    W, H = out_size
    blank = Image.new("RGB", (W, H), (240, 240, 240))

    if not video_path or not os.path.exists(video_path):
        return blank

    ext = os.path.splitext(video_path)[1].lower()

    if ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        try:
            img = Image.open(video_path).convert("RGB")
            return img.resize((W, H))
        except Exception:
            return blank

    if os.path.isdir(video_path):
        try:
            files = sorted([
                os.path.join(video_path, f) for f in os.listdir(video_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
            ])
            if not files:
                return blank
            img = Image.open(files[0]).convert("RGB")
            return img.resize((W, H))
        except Exception:
            return blank

    if ext == ".npy":
        try:
            arr = np.load(video_path)
            if arr.ndim == 4:
                arr = arr[len(arr)//2]
            if arr.ndim == 3:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
                img = Image.fromarray(arr).convert("RGB")
                return img.resize((W, H))
        except Exception:
            return blank

    if _HAS_CV2 and ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
        try:
            cap = cv2.VideoCapture(video_path)
            ok, frame = cap.read()
            cap.release()
            if not ok or frame is None:
                return blank
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame).convert("RGB")
            return img.resize((W, H))
        except Exception:
            return blank

    return blank


def build_dataset_from_cfg(cfg, split: str, label_map):
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    split_csv = {
        "train": data_cfg["train_csv"],
        "val": data_cfg["val_csv"],
        "test": data_cfg["test_csv"],
    }[split]

    kwargs = dict(
        csv_path=split_csv,
        text_col=data_cfg.get("text_col", "text"),
        label_col=data_cfg.get("label_col", "emotion"),
        audio_path_col=data_cfg.get("audio_path_col", "audio_path"),
        video_path_col=data_cfg.get("video_path_col", "video_path"),
        utt_id_col=data_cfg.get("utt_id_col", "utt_id"),
        num_frames=data_cfg.get("num_frames", 8),
        frame_size=data_cfg.get("frame_size", 224),
        sample_rate=data_cfg.get("sample_rate", 16000),
        max_audio_seconds=data_cfg.get("max_audio_seconds", 6.0),
        label_map=label_map,
        use_text=model_cfg.get("use_text", True),
        use_audio=model_cfg.get("use_audio", True),
        use_video=model_cfg.get("use_video", True),
        text_model_name=model_cfg.get("text_model_name", "roberta-base"),
    )

    import inspect
    sig = inspect.signature(MeldDataset.__init__)
    accepted = set(sig.parameters.keys())
    kwargs = {k: v for k, v in kwargs.items() if k in accepted}

    return MeldDataset(**kwargs)


def build_model_baseline(cfg, device):
    return CHADOTrimodal(
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


@torch.no_grad()
def run_probs_preds_labels(model, loader, device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    for batch in loader:
        labels = batch.labels.to(device)
        text = {k: v.to(device) for k, v in batch.text_input.items()} if batch.text_input else None
        audio = batch.audio_wave.to(device) if batch.audio_wave is not None else None
        video = batch.video_frames.to(device) if batch.video_frames is not None else None

        logits, _, _ = model(text_input=text, audio_wave=audio, video_frames=video, modality_mask=None)
        probs = torch.softmax(logits, dim=-1)

        all_probs.append(probs.cpu())
        all_preds.append(torch.argmax(probs, dim=-1).cpu())
        all_labels.append(labels.cpu())

    probs = torch.cat(all_probs, dim=0).numpy()
    preds = torch.cat(all_preds, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    return probs, preds, labels


@torch.no_grad()
def audio_only_emotion_label(model, batch, device, inv_label_map):
    """
    Force AUDIO-ONLY prediction using the trained CHADO model.
    """
    model.eval()

    if batch.audio_wave is None:
        return "unknown"

    audio = batch.audio_wave.to(device)

    # 🔑 FORCE modality mask: audio=1, text=0, video=0
    modality_mask = {
        "text": False,
        "audio": True,
        "video": False
    }

    logits, _, _ = model(
        text_input=None,
        audio_wave=audio,
        video_frames=None,
        modality_mask=modality_mask
    )

    pred_idx = int(torch.argmax(logits, dim=-1).item())
    return inv_label_map[pred_idx]



def make_figure(title_left, title_right, left_samples, right_samples, out_path):
    fig = plt.figure(figsize=(18, 6.2))
    gs = GridSpec(2, 2, figure=fig, wspace=0.12, hspace=0.22)

    def draw_panel(ax, panel_title, sample):
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        rect = patches.Rectangle(
            (0.01, 0.01), 0.98, 0.98,
            transform=ax.transAxes,
            linewidth=2.0,
            edgecolor="#1f77b4",
            facecolor="none",
            linestyle=(0, (6, 4)),
        )
        ax.add_patch(rect)

        ax.text(
            0.5, 0.92, panel_title,
            ha="center", va="center",
            fontsize=18, fontweight="bold",
            transform=ax.transAxes
        )

        # smaller thumbnail -> more text space
        inset = ax.inset_axes([0.04, 0.18, 0.30, 0.64])
        inset.axis("off")
        inset.imshow(sample["thumb"])

        # Longer wrap width + allow more lines naturally
        wrapped = textwrap.fill(sample["text"], width=62)

        content = (
            f"Text:  {wrapped}\n"
            f"Audio: {sample['audio_emotion']}\n\n"
            f"Truth:  {sample['truth']} ;  Predict:  {sample['pred']}"
        )
        ax.text(
            0.36, 0.80, content,
            ha="left", va="top",
            fontsize=12.5,
            transform=ax.transAxes
        )

    ax00 = fig.add_subplot(gs[0, 0])
    draw_panel(ax00, f"{title_left}\nSample 1", left_samples[0])

    ax01 = fig.add_subplot(gs[0, 1])
    draw_panel(ax01, f"{title_right}\nSample 1", right_samples[0])

    ax10 = fig.add_subplot(gs[1, 0])
    draw_panel(ax10, f"{title_left}\nSample 2", left_samples[1])

    ax11 = fig.add_subplot(gs[1, 1])
    draw_panel(ax11, f"{title_right}\nSample 2", right_samples[1])

    safe_mkdir(os.path.dirname(out_path) or ".")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved figure -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_scan", type=int, default=4000)
    ap.add_argument("--min_text_chars", type=int, default=80)  # enforce LONGER text
    ap.add_argument("--out", default="figures/meld_qualitative_correct_vs_failure_fixed.png")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    label_map = build_label_map_from_order(EMO_ORDER_7)
    inv_label_map = {v: k for k, v in label_map.items()}

    ds = build_dataset_from_cfg(cfg, args.split, label_map)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_meld(
            b, ds.tokenizer,
            cfg["model"]["use_text"],
            cfg["model"]["use_audio"],
            cfg["model"]["use_video"],
        ),
    )

    model = build_model_baseline(cfg, device)
    ckpt_obj = torch.load(args.ckpt, map_location="cpu")
    sd = unwrap_state_dict(ckpt_obj)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[LOAD] ckpt={args.ckpt}")
    print(f"[LOAD] missing={len(missing)} unexpected={len(unexpected)} (strict=False)")

    probs, preds, labels = run_probs_preds_labels(model, loader, device)

    if hasattr(ds, "df"):
        df = ds.df
    elif hasattr(ds, "data"):
        df = ds.data
    else:
        raise RuntimeError("Cannot access dataset table. Expected ds.df or ds.data in MeldDataset.")

    text_col = cfg["data"].get("text_col", "text")
    audio_col = cfg["data"].get("audio_path_col", "audio_path")
    video_col = cfg["data"].get("video_path_col", "video_path")

    N = probs.shape[0]
    scan_N = min(N, args.max_scan)
    idxs = np.arange(scan_N)

    # Enforce longer text
    text_lens = np.array([len(str(df.iloc[i][text_col])) if text_col in df.columns else 0 for i in idxs])
    long_mask = text_lens >= args.min_text_chars

    conf = probs[idxs].max(axis=1)
    ent = -(probs[idxs] * np.log(probs[idxs] + 1e-12)).sum(axis=1)

    correct_mask = (preds[idxs] == labels[idxs]) & long_mask
    wrong_mask = (preds[idxs] != labels[idxs]) & long_mask

    corr_candidates = idxs[correct_mask]
    fail_candidates = idxs[wrong_mask]

    # Score: prioritize confident correct / high-entropy failures
    corr_scores = conf[correct_mask] - 0.25 * ent[correct_mask]
    fail_scores = ent[wrong_mask] - conf[wrong_mask]

    # If long_mask filters too much, fallback to any length (avoid empty)
    if len(corr_candidates) < 2:
        corr_candidates = idxs[(preds[idxs] == labels[idxs])]
        corr_scores = conf[(preds[idxs] == labels[idxs])] - 0.25 * ent[(preds[idxs] == labels[idxs])]
    if len(fail_candidates) < 2:
        fail_candidates = idxs[(preds[idxs] != labels[idxs])]
        fail_scores = ent[(preds[idxs] != labels[idxs])] - conf[(preds[idxs] != labels[idxs])]

    corr_sorted = corr_candidates[np.argsort(-corr_scores)]
    fail_sorted = fail_candidates[np.argsort(-fail_scores)]

    corr_pick = corr_sorted[:2].tolist()
    fail_pick = fail_sorted[:2].tolist()

    # Build a tiny single-item loader to compute audio-only label per picked sample
    def one_item_loader(i):
        tmp_ds = torch.utils.data.Subset(ds, [i])
        return DataLoader(
            tmp_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda b: collate_meld(
                b, ds.tokenizer,
                cfg["model"]["use_text"],
                cfg["model"]["use_audio"],
                cfg["model"]["use_video"],
            ),
        )

    def sample_to_dict(i: int):
        row = df.iloc[i]
        text = str(row[text_col]) if text_col in row else ""
        audio_path = str(row[audio_col]) if audio_col in row else ""
        video_path = str(row[video_col]) if video_col in row else ""

        truth_idx = int(labels[i])
        pred_idx = int(preds[i])

        # audio-only label from the model
        aud_label = "missing"
        try:
            for b in one_item_loader(i):
                aud_label = audio_only_emotion_label(model, b, device, inv_label_map)
                break
        except Exception:
            aud_label = "unavailable"

        return {
            "thumb": load_video_thumbnail(video_path),
            "text": text,  # now we prefer longer ones; also wrap wider
            "audio_emotion": aud_label,
            "truth": inv_label_map.get(truth_idx, str(truth_idx)),
            "pred": inv_label_map.get(pred_idx, str(pred_idx)),
        }

    correct_samples = [sample_to_dict(i) for i in corr_pick]
    fail_samples = [sample_to_dict(i) for i in fail_pick]

    make_figure(
        title_left="MELD (Correct)",
        title_right="MELD (Failure)",
        left_samples=correct_samples,
        right_samples=fail_samples,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
