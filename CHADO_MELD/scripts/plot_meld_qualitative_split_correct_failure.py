#!/usr/bin/env python3
import os
import argparse
import textwrap
import yaml
import math
import numpy as np

import torch
import torch.nn.functional as F

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Optional deps (usually present). If missing, the script still runs but uses fallbacks.
try:
    import cv2
except Exception:
    cv2 = None

try:
    import torchaudio
except Exception:
    torchaudio = None

from transformers import AutoTokenizer

# Your model (same as you used before)
from src.models.chado_trimodal import CHADOTrimodal


# ---------------------------
# Utils
# ---------------------------
EMO_ORDER_7 = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]
EMO2ID = {e: i for i, e in enumerate(EMO_ORDER_7)}
ID2EMO = {i: e for e, i in EMO2ID.items()}

def safe_lower(x):
    return str(x).strip().lower()

def wrap_text(s, width):
    s = str(s).replace("\n", " ").strip()
    return "\n".join(textwrap.wrap(s, width=width, break_long_words=False, replace_whitespace=True))

def load_yaml(p):
    with open(p, "r") as f:
        return yaml.safe_load(f)

def extract_state_dict(ckpt_obj):
    # Supports common checkpoint formats
    if isinstance(ckpt_obj, dict):
        for k in ["state_dict", "model", "model_state_dict"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
    # If it's already a state_dict
    if isinstance(ckpt_obj, dict):
        return ckpt_obj
    raise ValueError("Unsupported checkpoint format (expected dict).")

def ensure_dir(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)

def tokenize_batch(tokenizer, texts, device, max_len=128):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    return {k: v.to(device) for k, v in enc.items()}

def load_audio_tensor(path, sample_rate=16000, max_seconds=6.0):
    """
    Returns: 1D torch tensor (L,) float32 in [-1,1], or None if fails.
    """
    if path is None or (not isinstance(path, str)) or len(path) == 0:
        return None
    if not os.path.exists(path):
        return None
    if torchaudio is None:
        return None

    try:
        wav, sr = torchaudio.load(path)  # (C, L)
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0)  # (L,)

        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)

        max_len = int(sample_rate * float(max_seconds))
        if wav.numel() > max_len:
            wav = wav[:max_len]
        elif wav.numel() < max_len:
            pad = max_len - wav.numel()
            wav = F.pad(wav, (0, pad))

        return wav.float().contiguous()
    except Exception:
        return None

def load_video_square_frame(path, frame_size=224):
    """
    Returns: (3,H,W) float tensor in [0,1] or None if fails.
    Uses center frame from video via cv2.
    """
    if path is None or (not isinstance(path, str)) or len(path) == 0:
        return None
    if not os.path.exists(path):
        return None
    if cv2 is None:
        return None

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:
            # try read first
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)

        ok, frame = cap.read()
        if not ok or frame is None:
            return None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (frame_size, frame_size), interpolation=cv2.INTER_AREA)
        frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0  # (3,H,W)
        return frame.contiguous()
    finally:
        cap.release()

def forward_logits(model, device, tokenizer, texts, audio_wavs, video_frames,
                   text_on=True, audio_on=True, video_on=True,
                   max_len=128):
    """
    Always supplies a valid text_input mapping to avoid:
      RobertaModel argument after ** must be a mapping, not NoneType
    When text_on=False, it tokenizes empty strings so text_input is still a mapping.
    """
    bs = len(texts)
    if text_on:
        text_input = tokenize_batch(tokenizer, texts, device, max_len=max_len)
    else:
        text_input = tokenize_batch(tokenizer, [""] * bs, device, max_len=max_len)

    if audio_on:
        audio = audio_wavs
    else:
        audio = None

    if video_on:
        video = video_frames
    else:
        video = None

    logits, _, _ = model(
        text_input=text_input,
        audio_wave=audio,
        video_frames=video,
        modality_mask=None
    )
    return logits


def render_rows(rows, out_path, title_right, max_text_width=70):
    """
    Rows: list of dict {sample_id, img_np, text, audio_label, gt, pred}
    Output: stacked figure, each row boxed, with square thumbnail.
    """
    n = len(rows)
    # Compact height; readable font
    fig_h = 2.2 * n
    fig = plt.figure(figsize=(16, fig_h))
    fig.suptitle("", y=0.995)

    for i, r in enumerate(rows, start=1):
        ax = fig.add_subplot(n, 1, i)
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # dashed border
        ax.add_patch(
            Rectangle(
                (0.01, 0.05), 0.98, 0.90,
                fill=False,
                linestyle=(0, (6, 4)),
                linewidth=1.8
            )
        )

        # Left label "Sample i"
        ax.text(0.02, 0.92, f"Sample {i}", fontsize=14, weight="bold", ha="left", va="top")

        # Right header e.g. "MELD (Correct)" / "MELD (Failure)"
        ax.text(0.98, 0.92, title_right, fontsize=14, weight="bold", ha="right", va="top")

        # Square thumbnail region
        img = r.get("img_np", None)
        img_x0, img_x1 = 0.16, 0.30
        img_y0, img_y1 = 0.30, 0.70  # square when aspect="equal"

        if img is not None:
            ax.imshow(img, extent=(img_x0, img_x1, img_y0, img_y1), aspect="equal")
        else:
            # placeholder square
            ax.add_patch(Rectangle((img_x0, img_y0), img_x1-img_x0, img_y1-img_y0, fill=False, linewidth=1.0))
            ax.text((img_x0+img_x1)/2, (img_y0+img_y1)/2, "No\nFrame", ha="center", va="center", fontsize=10)

        # Text block
        text_x = 0.34
        wrapped = wrap_text(r.get("text", ""), width=max_text_width)
        audio_label = r.get("audio_label", "N/A")

        ax.text(text_x, 0.72, f"Text:  {wrapped}", fontsize=14, ha="left", va="top")
        ax.text(text_x, 0.42, f"Audio:  {audio_label}", fontsize=14, ha="left", va="top")

        gt = r.get("gt", "N/A")
        pred = r.get("pred", "N/A")
        ax.text(text_x, 0.18, f"Truth:  {gt}; Predict:  {pred}", fontsize=15, ha="left", va="bottom")

    ensure_dir(out_path)
    plt.tight_layout()
    plt.savefig(out_path, dpi=350, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch_size", default=8, type=int)
    ap.add_argument("--max_scan_batches", default=1200, type=int)
    ap.add_argument("--pick_correct", default=5, type=int)
    ap.add_argument("--pick_failure", default=5, type=int)
    ap.add_argument("--max_text_width", default=70, type=int)
    ap.add_argument("--min_text_chars", default=35, type=int)
    ap.add_argument("--out_correct", required=True, type=str)
    ap.add_argument("--out_failure", required=True, type=str)
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    # CSV + columns (match your header)
    split_csv = cfg["data"][f"{args.split}_csv"]
    text_col = cfg["data"].get("text_col", "text")
    label_col = cfg["data"].get("label_col", "emotion")  # IMPORTANT: your csv has 'emotion'
    audio_col = cfg["data"].get("audio_path_col", "audio_path")
    video_col = cfg["data"].get("video_path_col", "video_path")

    sample_rate = cfg["data"].get("sample_rate", 16000)
    max_audio_seconds = cfg["data"].get("max_audio_seconds", 6.0)
    frame_size = cfg["data"].get("frame_size", 224)
    num_frames = cfg["data"].get("num_frames", 8)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tokenizer
    text_model_name = cfg["model"].get("text_model_name", "roberta-base")
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)

    # Model (disable CHADO extras; we only need base logits)
    model = CHADOTrimodal(
        text_model_name=text_model_name,
        audio_model_name=cfg["model"].get("audio_model_name", "facebook/wav2vec2-base"),
        video_model_name=cfg["model"].get("video_model_name", "google/vit-base-patch16-224-in21k"),
        num_classes=cfg["data"].get("num_classes", 7),
        proj_dim=cfg["model"].get("proj_dim", 256),
        dropout=cfg["model"].get("dropout", 0.2),
        use_text=True,   # keep encoders available
        use_audio=True,
        use_video=True,
        use_gated_fusion=cfg["model"].get("use_gated_fusion", True),
        use_causal=False,
        use_hyperbolic=False,
        use_transport=False,
        use_refinement=False,
    ).to(device)

    ckpt_obj = torch.load(args.ckpt, map_location="cpu")
    sd = extract_state_dict(ckpt_obj)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[LOAD] ckpt={args.ckpt}")
    print(f"[LOAD] missing={len(missing)} unexpected={len(unexpected)} (strict=False)")
    model.eval()

    # Read CSV
    df = pd.read_csv(split_csv)
    # Filter longer text (user requested long text)
    df[text_col] = df[text_col].astype(str)
    df = df[df[text_col].str.len() >= int(args.min_text_chars)].reset_index(drop=True)

    # Scan + collect
    correct_rows = []
    failure_rows = []

    bs = args.batch_size
    max_batches = args.max_scan_batches

    def to_img_np(frame_tensor):
        # (3,H,W) -> (H,W,3) uint8
        if frame_tensor is None:
            return None
        x = frame_tensor.detach().cpu().clamp(0, 1)
        x = (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return x

    n_total = len(df)
    n_batches = min(max_batches, math.ceil(n_total / bs))

    for bi in range(n_batches):
        start = bi * bs
        end = min(n_total, (bi + 1) * bs)
        if start >= end:
            break

        batch_df = df.iloc[start:end]

        texts = batch_df[text_col].tolist()
        gts_raw = [safe_lower(x) for x in batch_df[label_col].tolist()]

        # Map GT labels to ids; skip unknown
        gt_ids = []
        keep_idx = []
        for j, g in enumerate(gts_raw):
            if g in EMO2ID:
                gt_ids.append(EMO2ID[g])
                keep_idx.append(j)
        if len(keep_idx) == 0:
            continue

        # Load audio/video for kept rows
        audio_list = []
        video_list = []
        img_list = []
        for j in keep_idx:
            a_path = batch_df.iloc[j][audio_col] if audio_col in batch_df.columns else None
            v_path = batch_df.iloc[j][video_col] if video_col in batch_df.columns else None

            wav = load_audio_tensor(a_path, sample_rate=sample_rate, max_seconds=max_audio_seconds)
            frame = load_video_square_frame(v_path, frame_size=frame_size)

            # If no frame, keep None
            img_list.append(to_img_np(frame))

            # Build audio batch tensor later
            audio_list.append(wav)

            # Build video batch tensor: repeat one frame num_frames times if available
            if frame is None:
                video_list.append(None)
            else:
                vf = frame.unsqueeze(0).repeat(num_frames, 1, 1, 1)  # (T,3,H,W)
                video_list.append(vf)

        # Pack tensors
        # audio: (B,L)
        audio_ok = all(x is not None for x in audio_list)
        if audio_ok:
            audio_wavs = torch.stack(audio_list, dim=0).to(device)  # (B,L)
        else:
            # replace missing with zeros (still allow forward)
            max_len = int(sample_rate * float(max_audio_seconds))
            fixed = []
            for x in audio_list:
                if x is None:
                    fixed.append(torch.zeros(max_len))
                else:
                    fixed.append(x)
            audio_wavs = torch.stack(fixed, dim=0).to(device)

        # video: (B,T,3,H,W)
        video_ok = all(x is not None for x in video_list)
        if video_ok:
            video_frames = torch.stack(video_list, dim=0).to(device)
        else:
            # replace missing with zeros
            fixed = []
            for x in video_list:
                if x is None:
                    fixed.append(torch.zeros(num_frames, 3, frame_size, frame_size))
                else:
                    fixed.append(x)
            video_frames = torch.stack(fixed, dim=0).to(device)

        # Forward: tri-modal prediction
        logits_tri = forward_logits(
            model, device, tokenizer,
            [texts[j] for j in keep_idx],
            audio_wavs, video_frames,
            text_on=True, audio_on=True, video_on=True
        )
        pred_tri = logits_tri.argmax(dim=1).detach().cpu().numpy().tolist()

        # Forward: audio-only label (what you want as "Audio: joy/anger/...")
        logits_a = forward_logits(
            model, device, tokenizer,
            [texts[j] for j in keep_idx],  # not used semantically; text_on=False uses empty text mapping
            audio_wavs, video_frames,
            text_on=False, audio_on=True, video_on=False
        )
        pred_a = logits_a.argmax(dim=1).detach().cpu().numpy().tolist()

        # Add to pools
        for local_i, j in enumerate(keep_idx):
            gt_name = ID2EMO[int(gt_ids[local_i])]
            pred_name = ID2EMO[int(pred_tri[local_i])]
            audio_name = ID2EMO[int(pred_a[local_i])]  # Audio type label

            row = {
                "text": texts[j],
                "audio_label": audio_name,
                "gt": gt_name,
                "pred": pred_name,
                "img_np": img_list[local_i],
            }

            if pred_name == gt_name:
                if len(correct_rows) < args.pick_correct:
                    correct_rows.append(row)
            else:
                if len(failure_rows) < args.pick_failure:
                    failure_rows.append(row)

        if len(correct_rows) >= args.pick_correct and len(failure_rows) >= args.pick_failure:
            break

    if len(correct_rows) < args.pick_correct or len(failure_rows) < args.pick_failure:
        print(f"[WARN] Collected correct={len(correct_rows)}/{args.pick_correct}, failure={len(failure_rows)}/{args.pick_failure}.")

    # Render TWO separate figures (as you requested for visibility)
    render_rows(correct_rows, args.out_correct, title_right="MELD (Correct)", max_text_width=args.max_text_width)
    render_rows(failure_rows, args.out_failure, title_right="MELD (Failure)", max_text_width=args.max_text_width)


if __name__ == "__main__":
    main()
