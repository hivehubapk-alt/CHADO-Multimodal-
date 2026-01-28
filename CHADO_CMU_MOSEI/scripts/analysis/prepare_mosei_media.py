import os
import re
import json
import argparse
import subprocess

SEG_RE = re.compile(r"^(?P<vid>.+?)\[(?P<st>-?\d+(\.\d+)?),(?P<ed>-?\d+(\.\d+)?)\]$")

def run(cmd):
    subprocess.run(cmd, check=True)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def parse_from_rec(rec):
    """
    Try multiple schema variants. Returns (utt_id, video_id, start, end) or (utt_id, None, None, None).
    """
    utt_id = rec.get("utt_id") or rec.get("id") or rec.get("segment_id") or rec.get("segment") or rec.get("uid")
    if utt_id is None:
        return None, None, None, None

    # Direct fields if present
    video_id = rec.get("video_id") or rec.get("vid") or rec.get("video") or rec.get("youtube_id")
    start = rec.get("start") or rec.get("start_time") or rec.get("t_start")
    end = rec.get("end") or rec.get("end_time") or rec.get("t_end")

    # If missing, parse from utt_id like: VIDEOID[12.34,18.90]
    if (video_id is None or start is None or end is None) and isinstance(utt_id, str):
        m = SEG_RE.match(utt_id.strip())
        if m:
            video_id = m.group("vid")
            start = float(m.group("st"))
            end = float(m.group("ed"))

    # Final sanitize
    try:
        if start is not None:
            start = float(start)
        if end is not None:
            end = float(end)
    except Exception:
        return utt_id, None, None, None

    if video_id is None or start is None or end is None:
        return utt_id, None, None, None

    return utt_id, str(video_id), float(start), float(end)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="mosei_{split}.jsonl")
    ap.add_argument("--videos_dir", default="data/media/videos")
    ap.add_argument("--out_audio_dir", default="data/media/audio_wav")
    ap.add_argument("--out_frames_dir", default="data/media/frames")
    ap.add_argument("--max_items", type=int, default=0, help="0 = all")
    ap.add_argument("--frame_mode", default="middle", choices=["middle", "start", "end"])
    ap.add_argument("--quiet_ffmpeg", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.out_audio_dir)
    ensure_dir(args.out_frames_dir)

    total = 0
    parsed = 0
    missing_mp4 = 0
    extracted = 0
    sample_missing = []

    with open(args.manifest, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            rec = json.loads(line)

            utt_id, video_id, start, end = parse_from_rec(rec)
            if utt_id is None:
                continue

            if video_id is None:
                continue
            parsed += 1

            mp4 = os.path.join(args.videos_dir, f"{video_id}.mp4")
            if not os.path.exists(mp4):
                missing_mp4 += 1
                if len(sample_missing) < 10:
                    sample_missing.append((utt_id, mp4))
                continue

            dur = max(0.05, float(end - start))
            if args.frame_mode == "start":
                t_frame = start + 0.01
            elif args.frame_mode == "end":
                t_frame = max(start + 0.01, end - 0.02)
            else:
                t_frame = start + 0.5 * dur

            out_wav = os.path.join(args.out_audio_dir, f"{utt_id}.wav")
            out_jpg = os.path.join(args.out_frames_dir, f"{utt_id}.jpg")

            # Extract audio segment (mono, 16k)
            if not os.path.exists(out_wav):
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", f"{start:.3f}",
                    "-t", f"{dur:.3f}",
                    "-i", mp4,
                    "-vn",
                    "-ac", "1",
                    "-ar", "16000",
                    out_wav
                ]
                if args.quiet_ffmpeg:
                    cmd += ["-loglevel", "error"]
                run(cmd)

            # Extract one frame thumbnail
            if not os.path.exists(out_jpg):
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", f"{t_frame:.3f}",
                    "-i", mp4,
                    "-frames:v", "1",
                    "-q:v", "2",
                    out_jpg
                ]
                if args.quiet_ffmpeg:
                    cmd += ["-loglevel", "error"]
                run(cmd)

            extracted += 1
            if args.max_items and extracted >= args.max_items:
                break

    print(f"[SUMMARY] manifest lines             : {total}")
    print(f"[SUMMARY] parsed (video_id,start,end): {parsed}")
    print(f"[SUMMARY] missing mp4 in videos_dir  : {missing_mp4}")
    print(f"[SUMMARY] extracted segments         : {extracted}")
    print(f"[PATHS] frames: {args.out_frames_dir}")
    print(f"[PATHS] audio : {args.out_audio_dir}")

    if missing_mp4 > 0:
        print("\n[EXAMPLE] missing mp4 (first 10):")
        for utt, p in sample_missing:
            print(" ", utt, "->", p)

    if extracted == 0 and parsed > 0 and missing_mp4 == parsed:
        print("\n[CAUSE] You have segment timing, but NO matching MP4 files in data/media/videos/")
        print("       Put videos as: data/media/videos/<video_id>.mp4 (video_id comes from utt_id prefix).")

    if extracted == 0 and parsed == 0:
        print("\n[CAUSE] Could not parse (video_id,start,end) from manifest.")
        print("       Likely your utt_id is not in VIDEOID[start,end] format or fields differ.")

if __name__ == "__main__":
    main()
