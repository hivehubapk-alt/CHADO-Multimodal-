import os, json, argparse
import pandas as pd

EMO6 = ["happy", "sad", "angry", "fearful", "disgust", "surprise"]

# ---- Label mappings into EMO6 ----
# MELD typical: anger, disgust, fear, joy, neutral, sadness, surprise
MELD_TO_EMO6 = {
    "joy": "happy",
    "sadness": "sad",
    "anger": "angry",
    "fear": "fearful",
    "disgust": "disgust",
    "surprise": "surprise",
    # neutral is dropped (no mapping)
}

# IEMOCAP 6-class varies. Common: angry, happy, sad, neutral, excited, frustrated
# Map excited->happy, frustrated->angry; others not present become empty.
IEMOCAP_TO_EMO6 = {
    "happy": "happy",
    "excited": "happy",
    "sad": "sad",
    "angry": "angry",
    "frustrated": "angry",
    # neutral dropped
    # fearful/disgust/surprise typically absent in IEMOCAP6
}

def onehot_emo6(mapped_label: str):
    y = [0.0] * 6
    if mapped_label in EMO6:
        y[EMO6.index(mapped_label)] = 1.0
    return y

def normalize_label(x):
    if pd.isna(x):
        return None
    return str(x).strip().lower()

def build_from_meld(csv_path: str, out_jsonl: str):
    df = pd.read_csv(csv_path)
    # Try common column names
    # You may have: Utterance, Emotion, Dialogue_ID, Utterance_ID
    text_col = None
    for c in ["Utterance", "utterance", "text", "Sentence", "Transcript"]:
        if c in df.columns:
            text_col = c
            break
    emo_col = None
    for c in ["Emotion", "emotion", "label", "Label"]:
        if c in df.columns:
            emo_col = c
            break

    if text_col is None or emo_col is None:
        raise RuntimeError(f"MELD csv missing expected columns. Found: {df.columns.tolist()}")

    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    n_out = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            emo = normalize_label(row[emo_col])
            mapped = MELD_TO_EMO6.get(emo, None)
            if mapped is None:
                continue  # drop neutral/unknown
            rec = {
                "utt_id": f"meld_{i}",
                "split": "test",
                "text_raw": str(row[text_col]),
                "label": onehot_emo6(mapped),
                # cross-eval meta
                "src_dataset": "MELD",
                "src_label": emo,
                "mapped_label": mapped,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"[OK] MELD -> {out_jsonl} | kept={n_out} (neutral/unknown dropped)")

def build_from_iemocap(csv_path: str, out_jsonl: str):
    df = pd.read_csv(csv_path)
    # Your IEMOCAP CSVs often include: text, label, wav_path, avi_path, etc.
    text_col = None
    for c in ["text", "transcript", "utterance", "sentence", "Text"]:
        if c in df.columns:
            text_col = c
            break
    emo_col = None
    for c in ["label", "emotion", "Emotion", "y", "class"]:
        if c in df.columns:
            emo_col = c
            break

    if text_col is None or emo_col is None:
        raise RuntimeError(f"IEMOCAP csv missing expected columns. Found: {df.columns.tolist()}")

    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    n_out = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            emo = normalize_label(row[emo_col])
            mapped = IEMOCAP_TO_EMO6.get(emo, None)
            if mapped is None:
                continue  # drop neutral/unknown
            rec = {
                "utt_id": f"iemocap_{i}",
                "split": "test",
                "text_raw": str(row[text_col]),
                "label": onehot_emo6(mapped),
                "src_dataset": "IEMOCAP",
                "src_label": emo,
                "mapped_label": mapped,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"[OK] IEMOCAP -> {out_jsonl} | kept={n_out} (neutral/unknown dropped)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meld_csv", type=str, default=None)
    ap.add_argument("--iemocap_csv", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="data/manifests/cross")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.meld_csv:
        build_from_meld(args.meld_csv, os.path.join(args.out_dir, "meld_test_emo6.jsonl"))
    if args.iemocap_csv:
        build_from_iemocap(args.iemocap_csv, os.path.join(args.out_dir, "iemocap_test_emo6.jsonl"))

if __name__ == "__main__":
    main()
