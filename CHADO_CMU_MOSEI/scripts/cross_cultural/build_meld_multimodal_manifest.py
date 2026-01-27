import os
import json
import argparse
import pandas as pd

from scripts.cross_cultural.meld_io import load_meld_split_df


EMO6 = ["happy", "sad", "angry", "fearful", "disgust", "surprise"]

# MELD emotion -> emo6
MAP = {
    "joy": "happy",
    "happy": "happy",
    "sadness": "sad",
    "sad": "sad",
    "anger": "angry",
    "angry": "angry",
    "fear": "fearful",
    "fearful": "fearful",
    "disgust": "disgust",
    "surprise": "surprise",
}

DROP = {"neutral", "none", "other"}


def onehot(emo6: str):
    y = [0] * 6
    y[EMO6.index(emo6)] = 1
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meld_root", required=True)
    ap.add_argument("--features_root", required=True, help="data/features/meld")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split", default="test", choices=["train", "dev", "test", "val"])
    ap.add_argument("--require_audio", action="store_true")
    ap.add_argument("--require_video", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_meld_split_df(args.meld_root, args.split)

    kept = 0
    dropped = 0
    missing_av = 0

    out_path = os.path.join(args.out_dir, f"meld_{args.split}_emo6_multimodal.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            emo = str(r["emotion"]).strip().lower()
            if emo in DROP:
                dropped += 1
                continue
            if emo not in MAP:
                dropped += 1
                continue

            emo6 = MAP[emo]
            utt_id = r["utt_id"]
            text = r["text"]

            apath = os.path.join(args.features_root, "audio", f"{utt_id}.pt")
            vpath = os.path.join(args.features_root, "video", f"{utt_id}.pt")

            if args.require_audio and (not os.path.exists(apath)):
                missing_av += 1
                continue
            if args.require_video and (not os.path.exists(vpath)):
                missing_av += 1
                continue

            rec = {
                "utt_id": utt_id,
                "split": args.split,
                "text": text,
                "audio_path": apath if os.path.exists(apath) else None,
                "video_path": vpath if os.path.exists(vpath) else None,
                "label": onehot(emo6),
                "emo6": emo6,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[OK] Wrote: {out_path}")
    print(f"     kept={kept} dropped(neutral/unknown)={dropped} missing_required_av={missing_av}")
    print(f"     emo6 order: {EMO6}")


if __name__ == "__main__":
    main()
