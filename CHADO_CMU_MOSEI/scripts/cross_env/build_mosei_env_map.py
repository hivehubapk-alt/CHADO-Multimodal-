import os
import json
import argparse
import hashlib
from collections import defaultdict, Counter

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def speaker_id_from_rec(rec):
    # Prefer explicit speaker if exists; otherwise use video/utt id
    if "speaker" in rec and rec["speaker"]:
        return str(rec["speaker"])
    uid = str(rec.get("utt_id", ""))
    if "[" in uid:
        uid = uid.split("[", 1)[0]
    return uid

def stable_hash_to_bucket(s: str, k: int) -> int:
    # Stable across runs/machines
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h, 16) % k

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", required=True,
                    help="One or more jsonl manifests (train/val/test) to build env map over ALL speakers.")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # collect speakers from all manifests
    speakers = set()
    per_split_counts = Counter()
    for mp in args.manifests:
        n = 0
        for r in load_jsonl(mp):
            sid = speaker_id_from_rec(r)
            if sid:
                speakers.add(sid)
            n += 1
        per_split_counts[os.path.basename(mp)] = n

    speakers = sorted(list(speakers))
    print(f"[INFO] Manifests read:")
    for k, v in per_split_counts.items():
        print(f"  {k}: {v} lines")
    print(f"[INFO] Found {len(speakers)} unique speakers (union of provided manifests)")

    # assign env by stable hashing (speaker-disjoint environments)
    env_map = {sid: stable_hash_to_bucket(sid, args.k) for sid in speakers}

    # env sizes (speakers)
    env_sizes = Counter(env_map.values())
    print("[INFO] Environment sizes (speakers):")
    for e in range(args.k):
        print(f"  Env {e}: {env_sizes.get(e, 0)} speakers")

    # write (ensure directory exists)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(env_map, f, indent=2)

    print(f"[OK] Saved environment map → {args.out}")

if __name__ == "__main__":
    main()
