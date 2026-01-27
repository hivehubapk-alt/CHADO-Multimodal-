import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

VIDEO_EXTS = [".mp4", ".mkv", ".webm", ".avi", ".mov"]
AUDIO_EXTS = [".wav", ".flac", ".mp3", ".m4a"]

def build_basename_index(search_roots: List[str], exts: List[str]) -> Dict[str, str]:
    """
    Build a mapping: stem -> full_path, where stem is filename without extension.
    We index only basenames (fast and robust if your files are named by utt_id).
    """
    idx = {}
    for root in search_roots:
        rootp = Path(root)
        if not rootp.exists():
            continue
        for ext in exts:
            for p in rootp.rglob(f"*{ext}"):
                stem = p.stem
                # keep first occurrence; if duplicates exist, you can change priority
                if stem not in idx:
                    idx[stem] = str(p)
    return idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--search_roots", nargs="+", default=[
        "data", "datasets", "media", "outputs", "CMU_MOSEI", "CMU-MultimodalSDK"
    ])
    ap.add_argument("--project_root", default=".", help="base path for relative search_roots")
    args = ap.parse_args()

    proj = Path(args.project_root).resolve()
    roots = [str((proj / r).resolve()) for r in args.search_roots]

    print("[INFO] Building indices. This can take time if roots are large.")
    vid_idx = build_basename_index(roots, VIDEO_EXTS)
    aud_idx = build_basename_index(roots, AUDIO_EXTS)
    print(f"[INFO] Indexed videos: {len(vid_idx)}  audios: {len(aud_idx)}")

    kept = 0
    miss_v = 0
    miss_a = 0

    os.makedirs(str(Path(args.out_jsonl).parent), exist_ok=True)

    with open(args.in_jsonl, "r", encoding="utf-8") as fin, open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            utt_id = str(rec.get("utt_id", "")).strip()
            # If your utt_id is like "abcd_0001", and files are "abcd_0001.mp4", stem matches directly.
            # If files use a different naming convention, you will need to modify here.
            vpath = vid_idx.get(utt_id, None)
            apath = aud_idx.get(utt_id, None)

            if vpath is None:
                miss_v += 1
            else:
                rec["video_path"] = vpath

            if apath is None:
                miss_a += 1
            else:
                rec["audio_path"] = apath

            # Always expose a unified text field for your renderer
            if "text" not in rec or rec.get("text") in (None, ""):
                if "text_raw" in rec and rec["text_raw"] is not None:
                    rec["text"] = rec["text_raw"]

            kept += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[OK] Wrote: {args.out_jsonl}")
    print(f"[STATS] lines={kept}  missing_video={miss_v}  missing_audio={miss_a}")
    print("[NOTE] If missing_video is high, your videos are not stored under the searched roots or not named by utt_id.")

if __name__ == "__main__":
    main()
