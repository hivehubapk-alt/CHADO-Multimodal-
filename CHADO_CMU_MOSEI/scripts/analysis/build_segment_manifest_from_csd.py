import os
import json
import argparse
import numpy as np

from mmsdk.mmdatasdk import computational_sequence as cs

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def read_csd_sequence(csd_path: str):
    """
    Version-robust: uses computational_sequence directly.
    Returns cs object with .keys() and item access via seq.data[vid]["features"].
    """
    seq = cs(csd_path)
    # triggers read + integrity checks
    _ = seq.data
    return seq

def extract_start_end_from_words_features(features):
    """
    features is typically an array-like with rows: [start, end, token]
    Token may be bytes; times should be numeric.
    """
    if features is None:
        return None, None
    starts, ends = [], []
    try:
        for row in features:
            st = safe_float(row[0])
            ed = safe_float(row[1])
            if st is None or ed is None:
                continue
            starts.append(st)
            ends.append(ed)
    except Exception:
        return None, None

    if len(starts) == 0:
        return None, None
    st0 = float(np.min(starts))
    ed0 = float(np.max(ends))
    if ed0 <= st0:
        return None, None
    return st0, ed0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_manifest", required=True)
    ap.add_argument("--words_csd", default=None)
    ap.add_argument("--out_manifest", required=True)
    ap.add_argument("--min_dur", type=float, default=0.8)
    ap.add_argument("--max_dur", type=float, default=20.0)
    ap.add_argument("--debug_keys", action="store_true")
    args = ap.parse_args()

    # infer words csd path from manifest
    first = None
    with open(args.in_manifest, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                first = json.loads(line)
                break
    if first is None:
        raise RuntimeError("Empty in_manifest.")

    if args.words_csd is None:
        args.words_csd = first["csd"]["words"]

    if not os.path.exists(args.words_csd):
        raise FileNotFoundError(f"words_csd not found: {args.words_csd}")

    print("[LOAD] words CSD:", args.words_csd)
    words_seq = read_csd_sequence(args.words_csd)

    # Build mapping: video_id -> (start,end)
    vid2se = {}
    keys = list(words_seq.data.keys())
    print("[INFO] words sequence keys:", len(keys))
    if args.debug_keys:
        print("[DEBUG] first 10 keys:", keys[:10])

    n_ok = 0
    for vid in keys:
        try:
            item = words_seq.data[vid]
            feats = item.get("features", None) if isinstance(item, dict) else None
        except Exception:
            continue
        st, ed = extract_start_end_from_words_features(feats)
        if st is None:
            continue
        vid2se[str(vid)] = (st, ed)
        n_ok += 1

    print("[INFO] keys with valid timestamps:", n_ok)

    n_in = 0
    n_out = 0
    n_missing = 0
    n_short = 0

    os.makedirs(os.path.dirname(args.out_manifest), exist_ok=True)

    with open(args.in_manifest, "r", encoding="utf-8") as fin, open(args.out_manifest, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            n_in += 1
            rec = json.loads(line)

            # your manifest uses utt_id as youtube/video id
            vid = rec.get("utt_id")
            if vid not in vid2se:
                n_missing += 1
                continue

            st, ed = vid2se[vid]
            dur = ed - st
            if dur < args.min_dur:
                n_short += 1
                continue

            if dur > args.max_dur:
                mid = 0.5 * (st + ed)
                st = max(st, mid - 0.5 * args.max_dur)
                ed = st + args.max_dur

            rec2 = dict(rec)
            rec2["video_id"] = vid
            rec2["start"] = float(st)
            rec2["end"] = float(ed)
            rec2["utt_id"] = f"{vid}__{st:.3f}_{ed:.3f}"

            fout.write(json.dumps(rec2, ensure_ascii=False) + "\n")
            n_out += 1

    print("[DONE] in:", n_in, "out:", n_out, "missing_ts:", n_missing, "too_short:", n_short)
    print("[SAVED]", args.out_manifest)

if __name__ == "__main__":
    main()
