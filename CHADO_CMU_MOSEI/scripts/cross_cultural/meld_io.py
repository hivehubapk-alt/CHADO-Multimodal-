import os
import glob
import pandas as pd


DEFAULT_LABEL_COLS = ["Emotion", "emotion"]
DEFAULT_TEXT_COLS = ["Utterance", "utterance", "text"]
DEFAULT_DID_COLS = ["Dialogue_ID", "dialogue_id", "DialogueID"]
DEFAULT_UID_COLS = ["Utterance_ID", "utterance_id", "UtteranceID"]


def _find_first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def find_meld_csv(meld_root: str, split: str) -> str:
    """
    Tries common MELD naming patterns:
      - train_sent_emo.csv / dev_sent_emo.csv / test_sent_emo.csv
      - MELD.*.csv variants
    """
    split = split.lower()
    candidates = []
    # Common official names
    if split == "train":
        candidates += [
            os.path.join(meld_root, "train_sent_emo.csv"),
            os.path.join(meld_root, "train_sent_emo_dya.csv"),
        ]
    elif split in ("dev", "val", "valid", "validation"):
        candidates += [
            os.path.join(meld_root, "dev_sent_emo.csv"),
            os.path.join(meld_root, "val_sent_emo.csv"),
        ]
    elif split == "test":
        candidates += [os.path.join(meld_root, "test_sent_emo.csv")]

    # Sometimes nested
    candidates += [os.path.join(meld_root, "**", f"{split}_sent_emo*.csv")]
    candidates += [os.path.join(meld_root, "**", f"*{split}*emo*.csv")]
    candidates += [os.path.join(meld_root, "**", "*.csv")]

    # Expand globs; pick first that has required columns
    for pat in candidates:
        for p in glob.glob(pat, recursive=True):
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            cols = set(df.columns)
            if any(c in cols for c in DEFAULT_LABEL_COLS) and any(c in cols for c in DEFAULT_TEXT_COLS):
                return p

    raise FileNotFoundError(f"Could not locate MELD csv for split={split} under {meld_root}")


def load_meld_split_df(meld_root: str, split: str) -> pd.DataFrame:
    csv_path = find_meld_csv(meld_root, split)
    df = pd.read_csv(csv_path)

    # Normalize column names
    def pick(colset):
        for c in colset:
            if c in df.columns:
                return c
        return None

    c_label = pick(DEFAULT_LABEL_COLS)
    c_text = pick(DEFAULT_TEXT_COLS)
    c_did = pick(DEFAULT_DID_COLS)
    c_uid = pick(DEFAULT_UID_COLS)

    if c_label is None or c_text is None or c_did is None or c_uid is None:
        raise ValueError(
            f"CSV {csv_path} missing required cols. "
            f"Have={df.columns.tolist()} need label/text/dialogue_id/utterance_id"
        )

    out = pd.DataFrame({
        "dialogue_id": df[c_did].astype(int),
        "utterance_id": df[c_uid].astype(int),
        "text": df[c_text].astype(str),
        "emotion": df[c_label].astype(str),
    })
    out["utt_id"] = out.apply(lambda r: f"dia{r.dialogue_id}_utt{r.utterance_id}", axis=1)
    return out


def find_media_file(meld_root: str, split: str, utt_id: str, kind: str) -> str | None:
    """
    kind: 'audio' expects wav; 'video' expects mp4
    Searches typical MELD.Raw layouts:
      - {split}/audio/*.wav, {split}/video/*.mp4
      - audio/{split}/*.wav, videos/{split}/*.mp4
      - any nested file matching utt_id
    """
    split_norm = "dev" if split.lower() in ("val", "valid", "validation") else split.lower()
    ext = ".wav" if kind == "audio" else ".mp4"

    # Common locations
    patterns = [
        os.path.join(meld_root, split_norm, kind, f"{utt_id}{ext}"),
        os.path.join(meld_root, split_norm, f"{kind}s", f"{utt_id}{ext}"),
        os.path.join(meld_root, kind, split_norm, f"{utt_id}{ext}"),
        os.path.join(meld_root, f"{kind}s", split_norm, f"{utt_id}{ext}"),
        os.path.join(meld_root, "**", split_norm, "**", f"{utt_id}{ext}"),
        os.path.join(meld_root, "**", f"{utt_id}{ext}"),
    ]
    for pat in patterns:
        hits = glob.glob(pat, recursive=True)
        if hits:
            return hits[0]
    return None
