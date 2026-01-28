#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd


def ffmpeg_extract(video_path: str, wav_path: str) -> bool:
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        "-acodec", "pcm_s16le",
        wav_path
    ]
    try:
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return r.returncode == 0 and os.path.exists(wav_path) and os.path.getsize(wav_path) > 0
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", type=str, required=True)
    ap.add_argument("--audio_dir", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    in_csv = Path(args.in_csv).resolve()
    audio_dir = Path(args.audio_dir).resolve()
    out_csv = Path(args.out_csv).resolve()
    audio_dir.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)

    # checks
    for c in ["utt_id", "video_path"]:
        if c not in df.columns:
            raise KeyError(f"Missing column in manifest: {c}")

    # ffmpeg check
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg before continuing.")

    tasks = []
    # pre-fill audio_path if already extracted
    for i, r in df.iterrows():
        utt_id = str(r["utt_id"])
        vp = str(r["video_path"]) if pd.notna(r["video_path"]) else ""
        if not vp or not os.path.exists(vp):
            df.at[i, "audio_path"] = ""
            df.at[i, "has_audio"] = 0
            continue

        wav_path = str((audio_dir / f"{utt_id}.wav").resolve())
        if (not args.overwrite) and os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
            df.at[i, "audio_path"] = wav_path
            df.at[i, "has_audio"] = 1
        else:
            tasks.append((i, vp, wav_path))

    print(f"[INFO] total_rows={len(df)} to_extract={len(tasks)} workers={args.num_workers}")

    ok = 0
    fail = 0

    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futs = {ex.submit(ffmpeg_extract, vp, wp): (i, vp, wp) for (i, vp, wp) in tasks}
        for n_done, fut in enumerate(as_completed(futs), start=1):
            i, vp, wp = futs[fut]
            success = False
            try:
                success = fut.result()
            except Exception:
                success = False

            if success:
                ok += 1
                df.at[i, "audio_path"] = wp
                df.at[i, "has_audio"] = 1
            else:
                fail += 1
                df.at[i, "audio_path"] = ""
                df.at[i, "has_audio"] = 0

            if n_done % 500 == 0 or n_done == len(tasks):
                print(f"[PROG] done={n_done}/{len(tasks)} ok={ok} fail={fail}")

    missing_audio = int((df["has_audio"] != 1).sum())
    print(f"[STATS] extracted_ok={ok} extracted_fail={fail} missing_audio_total={missing_audio}")

    df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote: {out_csv}")


if __name__ == "__main__":
    main()
