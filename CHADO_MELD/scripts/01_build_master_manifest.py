#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import pandas as pd


VIDEO_RE = re.compile(r"dia(?P<dia>\d+)_utt(?P<utt>\d+)\.mp4$", re.IGNORECASE)


def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def detect_csv_paths(meld_root: Path) -> Dict[str, Path]:
    """
    Robust CSV discovery for MELD across common bundle layouts.
    Your case:
      - train/train_sent_emo.csv
      - MELD.Raw/MELD.Raw/dev_sent_emo.csv
      - MELD.Raw/MELD.Raw/test_sent_emo.csv
    So we must search recursively under meld_root and a few parents.
    """
    search_roots = []
    for base in [meld_root, meld_root.parent, meld_root.parent.parent]:
        if base and base.exists():
            search_roots.append(base)

    # also include direct 'train' subfolder if present in parent (common in MELD)
    for base in list(search_roots):
        if (base / "train").exists():
            search_roots.append(base / "train")

    # gather all csv recursively
    all_csvs = []
    for r in search_roots:
        all_csvs.extend(list(r.rglob("*.csv")))

    # de-dup
    uniq = {}
    for p in all_csvs:
        uniq[str(p.resolve())] = p.resolve()
    all_csvs = list(uniq.values())

    def best_for(split: str) -> Path:
        split = split.lower()
        # prefer exact canonical name if present
        canonical = f"{split}_sent_emo.csv"
        for p in all_csvs:
            if p.name.lower() == canonical:
                return p

        # otherwise choose best scoring candidate
        cands = []
        for p in all_csvs:
            n = p.name.lower()
            if split not in n:
                continue
            score = 0
            if "sent_emo" in n:
                score += 10
            if "emo" in n or "emotion" in n:
                score += 3
            if n.endswith(".csv"):
                score += 1
            # prefer shallower path
            score -= min(len(p.parts), 40) // 10
            cands.append((score, p))

        if not cands:
            raise FileNotFoundError(f"No CSV found for split='{split}' under {search_roots}")
        cands.sort(key=lambda x: x[0], reverse=True)
        return cands[0][1]

    found = {s: best_for(s) for s in ["train", "dev", "test"]}
    return found



def index_videos(video_root: Path) -> Dict[Tuple[int, int], str]:
    """
    Build mapping: (dialogue_id, utterance_id) -> absolute video path.
    Ignores macOS resource fork files starting with ._ .
    """
    mp4s = list(video_root.rglob("*.mp4"))
    idx: Dict[Tuple[int, int], str] = {}

    for p in mp4s:
        name = p.name
        if name.startswith("._"):
            continue
        m = VIDEO_RE.search(name)
        if not m:
            continue
        dia = int(m.group("dia"))
        utt = int(m.group("utt"))
        # prefer first seen; if duplicates, keep the shortest path (more canonical)
        key = (dia, utt)
        sp = str(p.resolve())
        if key not in idx:
            idx[key] = sp
        else:
            # choose shorter path (heuristic)
            if len(sp) < len(idx[key]):
                idx[key] = sp
    return idx


def infer_required_cols(df: pd.DataFrame) -> Dict[str, str]:
    """
    Map common MELD column names to canonical fields.
    """
    cols = set(df.columns)

    def pick(options: List[str]) -> str:
        for o in options:
            if o in cols:
                return o
        raise KeyError(f"Missing required column; tried {options}, have {sorted(cols)}")

    return {
        "dialogue_id": pick(["dialogue_id", "dialogue id"]),
        "utterance_id": pick(["utterance_id", "utterance id"]),
        "speaker": pick(["speaker"]),
        "text": pick(["utterance", "text"]),
        "emotion": pick(["emotion"]),
    }


def build_manifest_for_split(
    csv_path: Path,
    split_name: str,
    video_idx: Dict[Tuple[int, int], str],
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = norm_cols(df)
    col = infer_required_cols(df)

    rows = []
    missing_video = 0
    missing_text = 0

    for _, r in df.iterrows():
        dia = int(r[col["dialogue_id"]])
        utt = int(r[col["utterance_id"]])
        speaker = str(r[col["speaker"]]) if pd.notna(r[col["speaker"]]) else ""
        text = str(r[col["text"]]) if pd.notna(r[col["text"]]) else ""
        emo = str(r[col["emotion"]]) if pd.notna(r[col["emotion"]]) else ""

        vpath = video_idx.get((dia, utt), "")
        has_video = 1 if vpath and os.path.exists(vpath) else 0
        has_text = 1 if text.strip() else 0

        if not has_video:
            missing_video += 1
        if not has_text:
            missing_text += 1

        utt_id = f"dia{dia}_utt{utt}"

        rows.append(
            {
                "utt_id": utt_id,
                "dialogue_id": dia,
                "utterance_id": utt,
                "speaker": speaker,
                "text": text,
                "emotion": emo,
                "split_source": split_name,
                "video_path": vpath,
                "audio_path": "",  # filled later
                "has_video": has_video,
                "has_text": has_text,
                "has_audio": 0,  # filled later
            }
        )

    out = pd.DataFrame(rows)
    print(f"[{split_name}] rows={len(out)} missing_video={missing_video} missing_text={missing_text}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meld_raw_root", type=str, required=True,
                    help="Path to extracted MELD.Raw root that contains train/dev/test dirs OR their parent.")
    ap.add_argument("--out_csv", type=str, required=True)
    args = ap.parse_args()

    meld_root = Path(args.meld_raw_root).resolve()
    if not meld_root.exists():
        raise FileNotFoundError(f"meld_raw_root not found: {meld_root}")

    # Your current extracted layout is .../MELD.Raw/MELD.Raw with train/dev/test dirs.
    # We'll treat meld_root as that directory; if user passes parent, we still find train/dev/test below.
    # Find a directory that contains train/dev/test subfolders.
    candidates = [meld_root, meld_root / "MELD.Raw"]
    chosen = None
    for c in candidates:
        if (c / "train").exists() and (c / "dev").exists() and (c / "test").exists():
            chosen = c
            break
    if chosen is None:
        # fall back: search one level
        for c in meld_root.rglob("*"):
            if c.is_dir() and (c / "train").exists() and (c / "dev").exists() and (c / "test").exists():
                chosen = c
                break
    if chosen is None:
        raise FileNotFoundError(f"Could not locate train/dev/test folders under {meld_root}")

    video_root = chosen
    print(f"[OK] Using video_root: {video_root}")

    # CSVs may live one directory above (as in your case)
    csv_paths = detect_csv_paths(video_root)
    print("[OK] Found CSVs:")
    for k, v in csv_paths.items():
        print(f"  {k}: {v}")

    # Build one global video index (covers train/dev/test)
    video_idx = index_videos(video_root)
    print(f"[OK] Indexed videos: {len(video_idx)} unique (dialogue_id, utterance_id) pairs")

    parts = []
    for split in ["train", "dev", "test"]:
        parts.append(build_manifest_for_split(csv_paths[split], split, video_idx))

    master = pd.concat(parts, ignore_index=True)

    # Hard sanity: keep only tri-modal candidates for now (video+text required at this stage)
    total = len(master)
    master["is_trimodal_ready"] = ((master["has_video"] == 1) & (master["has_text"] == 1)).astype(int)

    out_path = Path(args.out_csv).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(out_path, index=False)

    print(f"[OK] Wrote: {out_path}")
    print(f"[STATS] total_rows={total} trimodal_ready_now={int(master['is_trimodal_ready'].sum())}")

    # Emotion distribution (quick)
    print("[STATS] emotion distribution (top 20):")
    print(master["emotion"].value_counts().head(20))


if __name__ == "__main__":
    main()
