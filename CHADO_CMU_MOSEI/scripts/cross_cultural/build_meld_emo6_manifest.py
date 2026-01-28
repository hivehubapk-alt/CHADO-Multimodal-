import os, json, argparse
import pandas as pd

EMO6 = ["happy", "sad", "angry", "fearful", "disgust", "surprise"]

# MELD typical labels: anger, disgust, fear, joy, neutral, sadness, surprise
MELD_TO_EMO6 = {
    "joy": "happy",
    "sadness": "sad",
    "anger": "angry",
    "fear": "fearful",
    "disgust": "disgust",
    "surprise": "surprise",
    # neutral -> dropped
}

def normalize(x):
    if pd.isna(x):
        return None
    return str(x).strip().lower()

def onehot(mapped_label: str):
    y = [0.0] * 6
    if mapped_label in EMO6:
        y[EMO6.index(mapped_label)] = 1.0
    return y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meld_csv", required=True)
    ap.add_argument("--out_jsonl", default="data/manifests/cross/meld_test_emo6.jsonl")
    args = ap.parse_args()

    df = pd.read_csv(args.meld_csv)

    # Find text + label columns robustly
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
        raise RuntimeError(f"[MELD] Missing columns. Found: {df.columns.tolist()}")

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    kept = 0
    dropped = 0
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            emo = normalize(row[emo_col])
            mapped = MELD_TO_EMO6.get(emo, None)
            if mapped is None:
                dropped += 1
                continue
            rec = {
                "utt_id": f"meld_test_{i}",
                "split": "test",
                "text_raw": str(row[text_col]),
                "label": onehot(mapped),
                "src_dataset": "MELD",
                "src_label": emo,
                "mapped_label": mapped,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[OK] Wrote: {args.out_jsonl}")
    print(f"     kept={kept}  dropped(neutral/unknown)={dropped}")
    print("     emo6 order:", EMO6)

if __name__ == "__main__":
    main()
