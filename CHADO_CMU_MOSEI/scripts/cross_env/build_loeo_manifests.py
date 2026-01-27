import os
import json
import argparse
from collections import Counter, defaultdict

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def save_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def speaker_id_from_rec(rec):
    """
    For your MOSEI manifests, utt_id is typically a YouTube video id
    (e.g., '-3nNcZdcdvU'). In your env-map builder you found 2659 unique
    'speakers' — this is consistent with treating the video_id as speaker_id.

    If later you add explicit speaker_id to manifest, this function will use it.
    """
    if "speaker" in rec and rec["speaker"]:
        return str(rec["speaker"])
    uid = str(rec.get("utt_id", ""))
    # If your utt_id ever becomes VIDEOID[start,end], split safely:
    if "[" in uid:
        uid = uid.split("[", 1)[0]
    return uid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_map", required=True, help="data/envs/mosei_env_map.json")
    ap.add_argument("--train_manifest", required=True)
    ap.add_argument("--val_manifest", required=True)
    ap.add_argument("--test_manifest", required=True)
    ap.add_argument("--out_root", default="data/cross_env", help="root dir to write fold projects")
    ap.add_argument("--k", type=int, default=None, help="number of envs (optional; inferred if not set)")
    args = ap.parse_args()

    with open(args.env_map, "r", encoding="utf-8") as f:
        env_map = json.load(f)  # speaker_id -> env_id

    # infer K if not given
    env_ids = list(env_map.values())
    K = args.k if args.k is not None else (max(env_ids) + 1 if env_ids else 0)
    if K <= 0:
        raise RuntimeError("Could not infer K from env_map. Is env_map empty?")

    # Load original split records
    split_records = {
        "train": list(load_jsonl(args.train_manifest)),
        "val": list(load_jsonl(args.val_manifest)),
        "test": list(load_jsonl(args.test_manifest)),
    }

    # Precompute speaker->records per split
    split_spk = {}
    for split, recs in split_records.items():
        spks = []
        for r in recs:
            sid = speaker_id_from_rec(r)
            spks.append(sid)
        split_spk[split] = spks

    # Stats: coverage of env_map
    missing = 0
    for split, recs in split_records.items():
        for r in recs:
            sid = speaker_id_from_rec(r)
            if sid not in env_map:
                missing += 1
    if missing > 0:
        print(f"[WARN] {missing} records have speakers not in env_map. They will be DROPPED in LOEO folds.")

    # Build folds
    summary = []
    for heldout in range(K):
        fold_root = os.path.join(args.out_root, f"fold_{heldout}")
        mani_dir = os.path.join(fold_root, "data", "manifests")
        os.makedirs(mani_dir, exist_ok=True)

        fold = {"train": [], "val": [], "test": []}
        fold_drop = Counter()

        # rule:
        #   heldout env speakers -> fold.test
        #   others -> keep same split name (train->train, val->val, test->test? NO)
        # CHADO LOEO: train/val use non-heldout; test is heldout from ORIGINAL test split only (to avoid leakage).
        #
        # We therefore:
        #   - fold.train = orig train filtered to env != heldout
        #   - fold.val   = orig val   filtered to env != heldout
        #   - fold.test  = orig test  filtered to env == heldout
        #
        for split in ["train", "val"]:
            for r in split_records[split]:
                sid = speaker_id_from_rec(r)
                eid = env_map.get(sid, None)
                if eid is None:
                    fold_drop[f"{split}_missing_env"] += 1
                    continue
                if int(eid) == heldout:
                    fold_drop[f"{split}_heldout_removed"] += 1
                    continue
                fold[split].append(r)

        for r in split_records["test"]:
            sid = speaker_id_from_rec(r)
            eid = env_map.get(sid, None)
            if eid is None:
                fold_drop["test_missing_env"] += 1
                continue
            if int(eid) != heldout:
                fold_drop["test_not_heldout_removed"] += 1
                continue
            fold["test"].append(r)

        out_train = os.path.join(mani_dir, "mosei_train.jsonl")
        out_val   = os.path.join(mani_dir, "mosei_val.jsonl")
        out_test  = os.path.join(mani_dir, "mosei_test.jsonl")

        save_jsonl(out_train, fold["train"])
        save_jsonl(out_val, fold["val"])
        save_jsonl(out_test, fold["test"])

        # per-fold speaker counts
        spk_counts = {}
        for split in ["train", "val", "test"]:
            spks = [speaker_id_from_rec(r) for r in fold[split]]
            spk_counts[split] = len(set(spks))

        # save fold metadata
        meta = {
            "heldout_env": heldout,
            "K": K,
            "counts": {s: len(fold[s]) for s in fold},
            "unique_speakers": spk_counts,
            "dropped": dict(fold_drop),
        }
        with open(os.path.join(fold_root, "fold_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        summary.append({
            "heldout_env": heldout,
            "train_n": len(fold["train"]),
            "val_n": len(fold["val"]),
            "test_n": len(fold["test"]),
            "train_spk": spk_counts["train"],
            "val_spk": spk_counts["val"],
            "test_spk": spk_counts["test"],
        })

        print(f"[OK] fold_{heldout} written → {fold_root}")
        print(f"     train={len(fold['train'])} (spk={spk_counts['train']})  "
              f"val={len(fold['val'])} (spk={spk_counts['val']})  "
              f"test={len(fold['test'])} (spk={spk_counts['test']})")

    # Write summary CSV
    out_csv = os.path.join(args.out_root, "loeo_summary.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("heldout_env,train_n,val_n,test_n,train_spk,val_spk,test_spk\n")
        for r in summary:
            f.write(f"{r['heldout_env']},{r['train_n']},{r['val_n']},{r['test_n']},"
                    f"{r['train_spk']},{r['val_spk']},{r['test_spk']}\n")
    print(f"[SAVED] {out_csv}")

if __name__ == "__main__":
    main()
