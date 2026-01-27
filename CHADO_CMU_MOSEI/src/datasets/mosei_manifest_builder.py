import os
import json
import argparse
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple
def _to_jsonable(x):
    """
    Convert SDK/numpy/torch objects into JSON-serializable Python types.
    Handles: numpy arrays/scalars, lists/tuples, dicts, and SDK wrappers.
    """
    if x is None:
        return None

    # Basic JSON-native
    if isinstance(x, (str, int, float, bool)):
        return x

    # Dict
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}

    # List/Tuple
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]

    # Numpy
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
    except Exception:
        pass

    # Torch
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
    except Exception:
        pass

    # SDK objects / unknown: best-effort string
    try:
        # sometimes SDK objects expose "tolist" or "values"
        if hasattr(x, "tolist"):
            return x.tolist()
        if hasattr(x, "values"):
            return _to_jsonable(x.values)
    except Exception:
        pass

    return str(x)


def _require_mmsdk():
    try:
        from mmsdk import mmdatasdk  # type: ignore
        return mmdatasdk
    except Exception as e:
        raise RuntimeError(
            "Could not import CMU Multimodal SDK.\n"
            "You already installed cmu-multimodal-sdk; ensure you are in the correct env.\n"
            f"Original error: {repr(e)}"
        )


def _paths(data_root: Path) -> Dict[str, Path]:
    base = data_root / "CMU-MOSEI"
    p = {
        "labels": base / "labels" / "CMU_MOSEI_Labels.csd",
        "words": base / "languages" / "CMU_MOSEI_TimestampedWords.csd",
        "covarep": base / "acoustics" / "CMU_MOSEI_COVAREP.csd",
        "facet": base / "visuals" / "CMU_MOSEI_VisualFacet42.csd",
        "openface": base / "visuals" / "CMU_MOSEI_VisualOpenFace2.csd",
    }
    missing = [k for k, v in p.items() if not v.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing expected files: {missing}\n"
            f"Checked under: {base}"
        )
    return p


def _load_csd(mmdatasdk, csd_path: Path):
    # Each .csd loaded as a separate dataset component
    return mmdatasdk.mmdataset({csd_path.stem: str(csd_path)})


def _safe_get(ds, comp: str, seg_id: str):
    try:
        return ds[comp][seg_id]
    except Exception:
        return None


def _words_to_text(words_entry) -> str:
    if words_entry is None:
        return ""
    feats = words_entry.get("features", None)
    if feats is None:
        return ""
    toks = []
    try:
        for t in feats:
            s = str(t)
            if s and s != "nan":
                toks.append(s)
    except Exception:
        return ""
    return " ".join(toks)


def _try_sdk_folds(mmdatasdk) -> Tuple[List[str], List[str], List[str]] | None:
    """
    Tries to retrieve standard CMU-MOSEI folds if the SDK version exposes them.
    If not available, return None and we will use deterministic hashing split.
    """
    # Try common attributes across versions
    candidates = []

    # Some versions expose cmu_mosei.standard_folds
    try:
        folds = mmdatasdk.cmu_mosei.standard_folds  # type: ignore
        candidates.append(folds)
    except Exception:
        pass

    # Some expose dataset.standard_folds["CMU_MOSEI"]
    try:
        folds = mmdatasdk.dataset.standard_folds.get("CMU_MOSEI")  # type: ignore
        if folds:
            candidates.append(folds)
    except Exception:
        pass

    # Some expose dataset.get_standard_folds("CMU_MOSEI")
    try:
        folds = mmdatasdk.dataset.get_standard_folds("CMU_MOSEI")  # type: ignore
        if folds:
            candidates.append(folds)
    except Exception:
        pass

    for folds in candidates:
        try:
            train = folds["train"]
            val = folds.get("valid", folds.get("val"))
            test = folds["test"]
            if train and val and test:
                return list(train), list(val), list(test)
        except Exception:
            continue

    return None


def _deterministic_split(ids: List[str], seed: str = "CHADO_MOSEI_V1") -> Tuple[List[str], List[str], List[str]]:
    """
    Deterministic 80/10/10 split via stable hash.
    This is only used if SDK folds aren't available.
    """
    def bucket(s: str) -> float:
        h = hashlib.sha1((seed + "::" + s).encode("utf-8")).hexdigest()
        # map first 8 hex chars to [0,1)
        v = int(h[:8], 16) / float(16**8)
        return v

    train, val, test = [], [], []
    for sid in ids:
        r = bucket(sid)
        if r < 0.80:
            train.append(sid)
        elif r < 0.90:
            val.append(sid)
        else:
            test.append(sid)
    return train, val, test


def build_manifests(
    data_root: Path,
    out_dir: Path,
    visual_key: str = "facet",
) -> None:
    mmdatasdk = _require_mmsdk()
    paths = _paths(data_root)

    if visual_key not in ("facet", "openface"):
        raise ValueError("visual_key must be one of: facet, openface")
    visual_path = paths[visual_key]

    # Load each component
    ds_labels = _load_csd(mmdatasdk, paths["labels"])
    ds_words = _load_csd(mmdatasdk, paths["words"])
    ds_audio = _load_csd(mmdatasdk, paths["covarep"])
    ds_visual = _load_csd(mmdatasdk, visual_path)

    labels_comp = paths["labels"].stem
    words_comp = paths["words"].stem
    audio_comp = paths["covarep"].stem
    visual_comp = visual_path.stem

    # Segment IDs from labels (most complete)
    seg_ids = [s for s in ds_labels[labels_comp].keys() if isinstance(s, str)]
    seg_ids_set = set(seg_ids)

    folds = _try_sdk_folds(mmdatasdk)
    if folds is None:
        print("[WARN] SDK standard folds not found; using deterministic 80/10/10 split.")
        train_ids, val_ids, test_ids = _deterministic_split(seg_ids)
    else:
        train_ids, val_ids, test_ids = folds
        # keep only those present in this package
        train_ids = [s for s in train_ids if s in seg_ids_set]
        val_ids = [s for s in val_ids if s in seg_ids_set]
        test_ids = [s for s in test_ids if s in seg_ids_set]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_train = out_dir / "mosei_train.jsonl"
    out_val = out_dir / "mosei_val.jsonl"
    out_test = out_dir / "mosei_test.jsonl"

    comps_ref = {
        "labels": str(paths["labels"]),
        "words": str(paths["words"]),
        "audio": str(paths["covarep"]),
        "visual": str(visual_path),
    }

    def write_split(split_name: str, ids: List[str], fp: Path):
        n_text = n_audio = n_video = 0
        with fp.open("w", encoding="utf-8") as f:
            for sid in ids:
                lab = _safe_get(ds_labels, labels_comp, sid)
                wrd = _safe_get(ds_words, words_comp, sid)
                aud = _safe_get(ds_audio, audio_comp, sid)
                vis = _safe_get(ds_visual, visual_comp, sid)

                if wrd is not None: n_text += 1
                if aud is not None: n_audio += 1
                if vis is not None: n_video += 1

                label_feats = None
                if lab is not None:
                    # Some SDK builds return non-serializable wrappers; convert aggressively.
                    label_feats = _to_jsonable(lab.get("features", None))


                rec: Dict[str, Any] = {
                    "utt_id": sid,
                    "split": split_name,
                    "text_raw": _words_to_text(wrd),

                    "csd": comps_ref,

                    "has_text": wrd is not None,
                    "has_audio": aud is not None,
                    "has_video": vis is not None,

                    "label_raw": label_feats,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"[OK] wrote {split_name}: {fp} (n={len(ids)})")
        print(f"     availability: text={n_text}, audio={n_audio}, video={n_video}")

    write_split("train", train_ids, out_train)
    write_split("val", val_ids, out_val)
    write_split("test", test_ids, out_test)

    print("\n[Summary]")
    print("train:", len(train_ids))
    print("val  :", len(val_ids))
    print("test :", len(test_ids))
    print("visual used:", visual_key, "->", visual_path.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="/home/tahirahmad/CHADO_CMU_MOSEI/data/raw/cmu-mosei")
    ap.add_argument("--out_dir", type=str, default="/home/tahirahmad/CHADO_CMU_MOSEI/data/manifests")
    ap.add_argument("--visual", type=str, default="facet", choices=["facet", "openface"])
    args = ap.parse_args()

    build_manifests(
        data_root=Path(args.data_root),
        out_dir=Path(args.out_dir),
        visual_key=args.visual,
    )


if __name__ == "__main__":
    main()
