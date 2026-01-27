import os
import json
import argparse
import torch
import pandas as pd

EMO_NAMES = ["happy", "sad", "angry", "fearful", "disgust", "surprise"]

def vec_to_str01(v):
    idx = [i for i, x in enumerate(v.tolist()) if int(x) == 1]
    return "none" if len(idx) == 0 else ";".join([EMO_NAMES[i] for i in idx])

def pick_text(rec):
    for k in ["text", "sentence", "raw_text", "utterance", "transcript", "text_raw"]:
        if k in rec and rec[k] is not None:
            s = str(rec[k]).strip()
            if s:
                return " ".join(s.split())
    return ""


def pick_path(rec, keys):
    for k in keys:
        v = rec.get(k, None)
        if v is None:
            continue
        v = str(v)
        if v.strip():
            return v
    return ""

def load_manifest(manifest_jsonl):
    recs = []
    with open(manifest_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    return recs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--manifest", required=True, help="MUST correspond to cached test_* indexing")
    ap.add_argument("--k_true", type=int, default=5)
    ap.add_argument("--k_wrong", type=int, default=5)
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()

    probs_p = os.path.join(args.cache_dir, "test_probs.pt")
    labels_p = os.path.join(args.cache_dir, "test_labels.pt")
    mad_p = os.path.join(args.cache_dir, "test_mad.pt")

    probs = torch.load(probs_p, map_location="cpu")      # [N,6]
    labels = torch.load(labels_p, map_location="cpu")    # [N,6]
    mad = torch.load(mad_p, map_location="cpu")          # [N]

    pred = (probs > args.thr).int()
    true = (labels > 0.5).int()

    # correctness: exact match on 6 labels
    correct_mask = (pred == true).all(dim=1)
    wrong_mask = ~correct_mask

    # sort by MAD descending (high ambiguity first)
    idx_all = torch.arange(mad.numel())
    idx_correct = idx_all[correct_mask]
    idx_wrong = idx_all[wrong_mask]

    idx_correct = idx_correct[torch.argsort(mad[idx_correct], descending=True)][:args.k_true]
    idx_wrong = idx_wrong[torch.argsort(mad[idx_wrong], descending=True)][:args.k_wrong]

    # attach metadata from manifest (THIS solves your empty text/video/audio)
    recs = load_manifest(args.manifest)
    N = probs.shape[0]
    if len(recs) != N:
        print(f"[WARN] manifest lines={len(recs)} but cache N={N}.")
        print("       If text/video are wrong, you are using the wrong manifest for this cache.")
        # still proceed

    def build_rows(indices, split_name):
        rows = []
        for i in indices.tolist():
            rec = recs[i] if i < len(recs) else {}
            row = {
                "idx": int(i),
                "mad": float(mad[i].item()),
                "true": vec_to_str01(true[i]),
                "pred": vec_to_str01(pred[i]),
                "text": pick_text(rec),
                "audio_path": pick_path(rec, ["audio_path", "wav", "audio", "path_audio"]),
                "video_path": pick_path(rec, ["video_path", "mp4", "video", "path_video"]),
                "utt_id": pick_path(rec, ["utt_id", "uid", "segment_id", "id"]),
                "speaker": pick_path(rec, ["speaker", "spk", "speaker_id"]),
                "split": split_name,
            }
            rows.append(row)
        return rows

    os.makedirs(args.out_dir, exist_ok=True)
    correct_rows = build_rows(idx_correct, "correct")
    wrong_rows = build_rows(idx_wrong, "wrong")

    df_c = pd.DataFrame(correct_rows)
    df_w = pd.DataFrame(wrong_rows)

    df_c.to_csv(os.path.join(args.out_dir, "selected_correct.csv"), index=False)
    df_w.to_csv(os.path.join(args.out_dir, "selected_wrong.csv"), index=False)

    torch.save({"correct": idx_correct, "wrong": idx_wrong}, os.path.join(args.out_dir, "selected_idx.pt"))

    print("[OK] Selected qualitative samples (highest MAD first)")
    print(f"  correct: {len(df_c)} -> {args.out_dir}/selected_correct.csv")
    print(f"  wrong  : {len(df_w)} -> {args.out_dir}/selected_wrong.csv")
    print(f"  bundle : {args.out_dir}/selected_idx.pt")

    print("\n[CORRECT]")
    print(df_c[["idx", "mad", "true", "pred"]].to_string(index=False))
    print("\n[WRONG]")
    print(df_w[["idx", "mad", "true", "pred"]].to_string(index=False))

if __name__ == "__main__":
    main()
