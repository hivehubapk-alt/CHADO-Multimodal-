import os
import argparse
import subprocess
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def make_placeholder(size=(220, 140), text="NO FRAME"):
    img = Image.new("RGB", size, (240, 240, 240))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, size[0]-1, size[1]-1], outline=(170, 170, 170), width=3)
    d.text((12, size[1]//2 - 10), text, fill=(90, 90, 90))
    return img

def safe_text(s):
    s = "" if s is None else str(s)
    return " ".join(s.strip().split())

def ffmpeg_extract_frame(video_path, out_jpg):
    """
    Extract a representative frame. We do:
    - use the midpoint frame via -ss 1.0 as a cheap default
    (If you want exact midpoint, we can probe duration, but this is robust.)
    """
    os.makedirs(os.path.dirname(out_jpg), exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",
        "-ss", "1.0",
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        out_jpg
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception:
        return False

def load_or_make_thumb(video_path, thumb_cache_dir, key, size=(220, 140)):
    """
    If video exists -> extract and cache thumbnail.
    Else placeholder.
    """
    if not video_path or not os.path.isfile(video_path):
        return make_placeholder(size=size)

    out_jpg = os.path.join(thumb_cache_dir, f"{key}.jpg")
    if not os.path.isfile(out_jpg):
        ok = ffmpeg_extract_frame(video_path, out_jpg)
        if not ok or (not os.path.isfile(out_jpg)):
            return make_placeholder(size=size)

    try:
        im = Image.open(out_jpg).convert("RGB").resize(size)
        return im
    except Exception:
        return make_placeholder(size=size)

def draw_panel(ax, sample_name, thumb_np, text, audio_tag, truth_str, pred_str):
    ax.set_axis_off()

    # dashed border
    rect = plt.Rectangle((0.01, 0.01), 0.98, 0.98, fill=False, linewidth=2.2, linestyle=(0, (6, 4)))
    ax.add_patch(rect)

    # left label
    ax.text(0.06, 0.52, sample_name, fontsize=18, va="center")

    # thumbnail
    ax.imshow(thumb_np, extent=(0.28, 0.56, 0.22, 0.82), aspect="auto")

    # right text block (match paper feel)
    x0 = 0.60
    ax.text(x0, 0.68, f"Text:   {text}", fontsize=16, va="center")
    ax.text(x0, 0.52, f"Audio:  {audio_tag}.", fontsize=16, va="center")
    ax.text(x0, 0.32, f"Truth:  {truth_str}; Predict:  {pred_str}", fontsize=16, va="center")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected_correct_csv", default="outputs/analysis/qualitative/selected_correct.csv")
    ap.add_argument("--selected_wrong_csv", default="outputs/analysis/qualitative/selected_wrong.csv")
    ap.add_argument("--out_png", default="outputs/analysis/qualitative/qualitative_panels.png")
    ap.add_argument("--thumb_cache_dir", default="outputs/analysis/qualitative/thumbs")
    ap.add_argument("--n_true", type=int, default=5)
    ap.add_argument("--n_wrong", type=int, default=5)
    args = ap.parse_args()

    df_c = pd.read_csv(args.selected_correct_csv).head(args.n_true)
    df_w = pd.read_csv(args.selected_wrong_csv).head(args.n_wrong)
    df = pd.concat([df_c, df_w], ignore_index=True)

    total = len(df)
    fig_h = 2.1 * total
    fig, axes = plt.subplots(total, 1, figsize=(12.5, fig_h), dpi=200)
    if total == 1:
        axes = [axes]

    for row_id, (ax, row) in enumerate(zip(axes, df.to_dict(orient="records")), start=1):
        txt = safe_text(row.get("text", ""))
        if txt == "":
            txt = "[MISSING TEXT: check that --manifest matches outputs/cache indexing]"

        # audio tag: use file presence as a clean “Audio: Neutral.” placeholder
        # (If you want energy-based tags from wav, I can add that too.)
        audio_path = row.get("audio_path", "")
        audio_tag = "Neutral" if (not audio_path or not os.path.isfile(str(audio_path))) else "Neutral"

        truth_str = safe_text(row.get("true", ""))
        pred_str = safe_text(row.get("pred", ""))

        video_path = str(row.get("video_path", "") or "")
        key = f"idx{int(row.get('idx', row_id))}"
        thumb = load_or_make_thumb(video_path, args.thumb_cache_dir, key, size=(220, 140))
        thumb_np = np.asarray(thumb)

        draw_panel(ax, f"Sample {row_id}", thumb_np, txt, audio_tag, truth_str, pred_str)

    fig.suptitle("CMU-MOSEI", fontsize=20, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    fig.savefig(args.out_png, bbox_inches="tight")
    print(f"[SAVED] {args.out_png}")
    print(f"[THUMBS] {args.thumb_cache_dir}")

if __name__ == "__main__":
    main()
