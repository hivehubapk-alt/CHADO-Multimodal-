#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path
import pandas as pd


def run_cmd(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return r.returncode, r.stdout, r.stderr


def extract(vp: str, wp: str) -> bool:
    # more robust: force audio stream selection if present; ignore minor errors
    cmd = [
        "ffmpeg", "-y",
        "-i", vp,
        "-map", "0:a:0?",
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        "-acodec", "pcm_s16le",
        wp
    ]
    code, _, _ = run_cmd(cmd)
    return code == 0 and os.path.exists(wp) and os.path.getsize(wp) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    audio_dir = Path(args.audio_dir).resolve()
    audio_dir.mkdir(parents=True, exist_ok=True)

    # ensure string dtype to avoid pandas FutureWarning
    if "audio_path" not in df.columns:
        df["audio_path"] = ""
    df["audio_path"] = df["audio_path"].fillna("").astype(str)

    bad_idx = df.index[df["has_audio"] != 1].tolist()
    print(f"[INFO] retry_count={len(bad_idx)}")

    fixed = 0
    for i in bad_idx:
        utt_id = str(df.at[i, "utt_id"])
        vp = str(df.at[i, "video_path"])
        if not vp or not os.path.exists(vp):
            print(f"[FAIL] {utt_id} video missing: {vp}")
            continue
        wp = str((audio_dir / f"{utt_id}.wav").resolve())
        ok = extract(vp, wp)
        if ok:
            df.at[i, "audio_path"] = wp
            df.at[i, "has_audio"] = 1
            fixed += 1
            print(f"[OK] {utt_id} -> {wp}")
        else:
            # get ffmpeg error message for diagnosis
            code, out, err = run_cmd(["ffmpeg","-i",vp])
            print(f"[FAIL] {utt_id} ffmpeg_probe_code={code}")
            # keep short last lines
            err_lines = (err or "").splitlines()[-8:]
            print("\n".join(err_lines))

    missing_after = int((df["has_audio"] != 1).sum())
    print(f"[STATS] fixed={fixed} missing_audio_after={missing_after}")

    out_csv = Path(args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv}")


if __name__ == "__main__":
    main()
