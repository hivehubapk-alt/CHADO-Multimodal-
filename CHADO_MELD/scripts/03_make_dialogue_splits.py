#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    assert abs((args.train_ratio + args.val_ratio + args.test_ratio) - 1.0) < 1e-6, "ratios must sum to 1"

    in_csv = Path(args.in_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)

    required = ["utt_id", "dialogue_id", "emotion", "text", "video_path", "audio_path", "has_video", "has_text", "has_audio"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    total_before = len(df)

    # Keep only tri-modal rows
    df = df[(df["has_video"] == 1) & (df["has_text"] == 1) & (df["has_audio"] == 1)].copy()
    df.reset_index(drop=True, inplace=True)

    total_after = len(df)
    dropped = total_before - total_after
    print(f"[STATS] total_before={total_before} total_after_trimodal={total_after} dropped_nontrimodal={dropped}")

    # Dialogue IDs
    dialogues = df["dialogue_id"].unique().tolist()
    rng = np.random.default_rng(args.seed)
    rng.shuffle(dialogues)

    n = len(dialogues)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    n_test = n - n_train - n_val

    train_d = set(dialogues[:n_train])
    val_d = set(dialogues[n_train:n_train + n_val])
    test_d = set(dialogues[n_train + n_val:])

    # Sanity: no overlap
    assert train_d.isdisjoint(val_d)
    assert train_d.isdisjoint(test_d)
    assert val_d.isdisjoint(test_d)

    def assign_split(did: int) -> str:
        if did in train_d:
            return "train"
        if did in val_d:
            return "val"
        return "test"

    df["split_rand"] = df["dialogue_id"].apply(assign_split)

    train_df = df[df["split_rand"] == "train"].copy()
    val_df = df[df["split_rand"] == "val"].copy()
    test_df = df[df["split_rand"] == "test"].copy()

    # Verify no dialogue leakage
    assert set(train_df["dialogue_id"]).isdisjoint(set(val_df["dialogue_id"]))
    assert set(train_df["dialogue_id"]).isdisjoint(set(test_df["dialogue_id"]))
    assert set(val_df["dialogue_id"]).isdisjoint(set(test_df["dialogue_id"]))

    train_path = out_dir / "meld_train.csv"
    val_path = out_dir / "meld_val.csv"
    test_path = out_dir / "meld_test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[OK] wrote: {train_path}")
    print(f"[OK] wrote: {val_path}")
    print(f"[OK] wrote: {test_path}")

    print("[STATS] dialogues_total:", n, "train/val/test:", (len(train_d), len(val_d), len(test_d)))
    print("[STATS] rows_total:", len(df), "train/val/test:", (len(train_df), len(val_df), len(test_df)))

    def dist(name, dfx):
        vc = dfx["emotion"].value_counts()
        print(f"[STATS] emotion distribution {name}:")
        print(vc)

    dist("train", train_df)
    dist("val", val_df)
    dist("test", test_df)


if __name__ == "__main__":
    main()
