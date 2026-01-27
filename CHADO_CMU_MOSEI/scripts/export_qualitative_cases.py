import os
import sys
import re
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import importlib.util
import inspect

# ---------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EMO_NAMES = ["happy", "sad", "angry", "fearful", "disgust", "surprise"]


def load_cfg_yaml(cfg_path: str):
    import yaml
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _safe_call(fn, cfg, ablation):
    """
    Try calling a function with a few common signatures.
    Returns output if succeeds, else None.
    """
    trials = [
        {"cfg": cfg, "ablation": ablation},
        {"cfg": cfg, "mode": ablation},
        {"cfg": cfg},
        {"config": cfg, "ablation": ablation},
        {"config": cfg},
    ]
    for kwargs in trials:
        try:
            return fn(**kwargs)
        except TypeError:
            continue
        except Exception:
            continue
    return None


def try_build_model_from_trainpy(cfg, ablation: str):
    """
    Your logs show model builder exists at src/train/train.py::build_model.
    We import it directly.
    """
    import src.train.train as train_mod
    if not hasattr(train_mod, "build_model"):
        raise RuntimeError("Expected build_model in src.train.train but not found.")
    model = train_mod.build_model(cfg=cfg, ablation=ablation)
    return model


def try_build_loaders_from_trainpy(cfg, ablation: str):
    """
    Robust fallback: discover loader-building function from src/train/train.py
    by scanning callables with 'loader/dataload/dataset' in name and trying to call them.
    Accepts dict(train/val/test) or tuple/list of loaders.
    """
    import src.train.train as train_mod

    # Preferred common names first (if present)
    preferred_names = [
        "build_dataloaders",
        "build_loaders",
        "get_dataloaders",
        "make_dataloaders",
        "make_loaders",
        "build_data",
        "build_dataset",
        "build_datasets",
        "get_loaders",
    ]

    for name in preferred_names:
        fn = getattr(train_mod, name, None)
        if callable(fn):
            out = _safe_call(fn, cfg, ablation)
            if out is not None:
                loaders = normalize_loaders(out)
                if loaders is not None:
                    print(f"[OK] Dataloaders from src/train/train.py::{name}")
                    return loaders

    # Otherwise scan all callables in train.py and try likely ones
    candidates = []
    for name, obj in vars(train_mod).items():
        if not callable(obj):
            continue
        if re.search(r"(loader|dataload|dataset)", name, re.IGNORECASE):
            candidates.append((name, obj))

    # deterministic order
    candidates.sort(key=lambda x: x[0])

    for name, fn in candidates:
        out = _safe_call(fn, cfg, ablation)
        if out is None:
            continue
        loaders = normalize_loaders(out)
        if loaders is not None:
            print(f"[OK] Dataloaders discovered from src/train/train.py::{name}")
            return loaders

    raise RuntimeError(
        "Could not discover any loader-building function in src/train/train.py. "
        "Please check your train.py for the function that creates train/val/test loaders "
        "and rename it to one of: build_dataloaders/build_loaders/get_dataloaders, "
        "or tell me its exact function name."
    )


def normalize_loaders(out):
    """
    Normalize output into dict with keys train/val/test when possible.
    Acceptable inputs:
      - dict with train/val/test (or dev)
      - tuple/list of length 3 (train,val,test)
    """
    if isinstance(out, dict):
        keys = set(out.keys())
        # common variants
        if "train" in keys and ("val" in keys or "dev" in keys) and "test" in keys:
            if "val" not in out and "dev" in out:
                out["val"] = out["dev"]
            return out
        # Sometimes returned as {"train":..., "test":...} only
        if "train" in keys and "test" in keys:
            return out

    if isinstance(out, (tuple, list)) and len(out) == 3:
        return {"train": out[0], "val": out[1], "test": out[2]}

    return None


@torch.no_grad()
def main():
    cfg_path = "src/configs/chado_mosei_emo6.yaml"
    ckpt_path = "outputs/checkpoints/TAV_best.pt"
    out_csv = "outputs/qualitative_cases.csv"
    ablation = "TAV"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not Path(cfg_path).exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    cfg = load_cfg_yaml(cfg_path)

    # Build model + loaders from your actual training module
    model = try_build_model_from_trainpy(cfg, ablation=ablation)
    loaders = try_build_loaders_from_trainpy(cfg, ablation=ablation)

    test_loader = loaders.get("test", None)
    if test_loader is None:
        # fallback: use val if no test loader key
        test_loader = loaders.get("val", None)
    if test_loader is None:
        raise RuntimeError(f"No test/val loader found. Loader keys: {list(loaders.keys())}")

    # Load checkpoint
    state = torch.load(ckpt_path, map_location="cpu")
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(sd, strict=False)

    model.to(device)
    model.eval()

    rows = []
    os.makedirs("outputs", exist_ok=True)

    print("[INFO] Exporting qualitative cases from TEST split...")

    for batch in tqdm(test_loader):
        # Ensure expected keys
        if "text" not in batch or "label" not in batch:
            raise RuntimeError(f"Batch missing text/label keys. Found keys: {list(batch.keys())}")

        text = batch["text"]
        labels = batch["label"]

        # Move tensors to device
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)

        logits = model(batch)
        probs = torch.sigmoid(logits)

        # Labels may be tensor on CPU originally, move-safe
        labels_cpu = labels.detach().cpu()
        probs_cpu = probs.detach().cpu()

        for i in range(len(text)):
            true_vec = labels_cpu[i].numpy().tolist()
            prob_vec = probs_cpu[i].numpy().tolist()
            pred_vec = [1 if p > 0.5 else 0 for p in prob_vec]

            true_names = [EMO_NAMES[j] for j, t in enumerate(true_vec) if t == 1]
            pred_names = [EMO_NAMES[j] for j, t in enumerate(pred_vec) if t == 1]

            rows.append({
                "text": text[i],
                "true_labels": ", ".join(true_names),
                "pred_labels": ", ".join(pred_names),
                "pred_probs": [round(float(p), 4) for p in prob_vec],
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[DONE] Saved: {out_csv}")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
