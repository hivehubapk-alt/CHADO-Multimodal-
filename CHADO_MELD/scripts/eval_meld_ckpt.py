#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import importlib
import inspect
import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import yaml
except Exception:
    yaml = None

try:
    from sklearn.metrics import f1_score
except Exception as e:
    raise RuntimeError("Please install scikit-learn in chado_meld env: pip install scikit-learn") from e


def _load_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("pyyaml not found. Install: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get(d: Dict[str, Any], key: str, default=None):
    return d[key] if key in d else default


def _extract_state_dict(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    # common patterns in your runs
    if isinstance(ckpt_obj, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "module"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
        # sometimes checkpoint itself is a state_dict
        # heuristic: values are tensors
        if any(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj
    raise RuntimeError("Could not extract a state_dict from checkpoint. Keys={}".format(
        list(ckpt_obj.keys()) if isinstance(ckpt_obj, dict) else type(ckpt_obj)
    ))


def _find_first_module_class(module_name: str) -> type:
    """
    Import module and return the first nn.Module subclass defined in it
    (excluding imported base classes).
    """
    mod = importlib.import_module(module_name)
    candidates = []
    for name, obj in vars(mod).items():
        if inspect.isclass(obj) and issubclass(obj, nn.Module):
            # prefer classes defined in this module
            if getattr(obj, "__module__", "") == module_name:
                candidates.append(obj)
    if not candidates:
        # fallback: any Module class
        for name, obj in vars(mod).items():
            if inspect.isclass(obj) and issubclass(obj, nn.Module):
                candidates.append(obj)
    if not candidates:
        raise RuntimeError(f"No nn.Module classes found inside {module_name}")
    # Prefer names containing "Baseline" or "CHADO" if multiple exist
    def score(c):
        n = c.__name__.lower()
        s = 0
        if "chado" in n: s += 5
        if "baseline" in n: s += 4
        if "trimodal" in n: s += 3
        return s
    candidates = sorted(candidates, key=score, reverse=True)
    return candidates[0]


def _instantiate_with_signature(cls: type, cfg: Dict[str, Any]) -> nn.Module:
    """
    Instantiate model by matching ctor signature against cfg keys.
    Supports:
      - __init__(self, cfg)
      - __init__(self, model_cfg, data_cfg)
      - __init__(self, **kwargs) with keys from cfg['model'] etc.
    """
    sig = inspect.signature(cls.__init__)
    params = list(sig.parameters.values())[1:]  # drop self

    model_cfg = _get(cfg, "model", {}) or {}
    data_cfg = _get(cfg, "data", {}) or {}
    chado_cfg = _get(cfg, "chado", {}) or {}

    # Try direct patterns first
    try:
        if len(params) == 1:
            p0 = params[0].name
            if p0 in ["cfg", "config"]:
                return cls(cfg)
            if p0 in ["model_cfg", "model", "mcfg"]:
                return cls(model_cfg)
    except Exception:
        pass

    try:
        if len(params) == 2:
            names = [p.name for p in params]
            if names[0] in ["cfg", "config"] and names[1] in ["cfg", "config"]:
                return cls(cfg, cfg)
            if names[0] in ["model_cfg", "model"] and names[1] in ["data_cfg", "data"]:
                return cls(model_cfg, data_cfg)
    except Exception:
        pass

    # Generic kwargs: take from merged dict
    merged = {}
    merged.update(data_cfg)
    merged.update(model_cfg)
    merged.update({f"chado_{k}": v for k, v in chado_cfg.items()})

    kwargs = {}
    for p in params:
        if p.kind in (p.VAR_KEYWORD,):
            # can pass everything
            kwargs = merged
            break
        if p.name in merged:
            kwargs[p.name] = merged[p.name]

    try:
        return cls(**kwargs)
    except Exception:
        # last fallback: try passing cfg only
        try:
            return cls(cfg)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate {cls.__name__}. "
                f"Try editing this script to match your model ctor. "
                f"Got error: {e}"
            )


def _build_dataset(cfg: Dict[str, Any], split: str):
    # You confirmed these exist:
    from src.data.meld_dataset import MeldDataset

    data_cfg = cfg.get("data", {})
    csv_key = f"{split}_csv"
    split_csv = data_cfg.get(csv_key) or data_cfg.get("test_csv")

    if not split_csv or not os.path.exists(split_csv):
        raise FileNotFoundError(f"split_csv not found: {split_csv}")

    # signature-safe init
    sig = inspect.signature(MeldDataset.__init__)
    params = list(sig.parameters.keys())[1:]  # drop self

    # Known fields from your CSV header:
    # ['utt_id','dialogue_id','utterance_id','speaker','text','emotion','video_path','audio_path', ...]
    label_col = data_cfg.get("label_col", "emotion")
    text_col = data_cfg.get("text_col", "text")
    audio_path_col = data_cfg.get("audio_path_col", "audio_path")
    video_path_col = data_cfg.get("video_path_col", "video_path")

    # minimal kwargs; only pass what ctor accepts
    kwargs = {}
    for k, v in [
        ("csv_path", split_csv),
        ("csv", split_csv),
        ("path", split_csv),
        ("label_col", label_col),
        ("text_col", text_col),
        ("audio_path_col", audio_path_col),
        ("video_path_col", video_path_col),
        ("num_frames", data_cfg.get("num_frames", 8)),
        ("frame_size", data_cfg.get("frame_size", 224)),
        ("sample_rate", data_cfg.get("sample_rate", 16000)),
        ("max_audio_seconds", data_cfg.get("max_audio_seconds", 6.0)),
    ]:
        if k in params:
            kwargs[k] = v

    ds = MeldDataset(**kwargs)
    return ds, split_csv, label_col


@torch.no_grad()
def _forward_logits(model: nn.Module, batch: Dict[str, Any], device: torch.device) -> torch.Tensor:
    """
    Supports common forward signatures used in your repo:
      model(text_input=..., audio_wave=..., video_frames=..., modality_mask=None)
    and several fallbacks.
    """
    model.eval()

    # move tensors to device
    def to_dev(x):
        if torch.is_tensor(x):
            return x.to(device, non_blocking=True)
        return x

    batch = {k: to_dev(v) for k, v in batch.items()}

    # common keys produced by meld_dataset.collate_meld
    text = batch.get("text_input", None) or batch.get("text", None) or batch.get("text_tokens", None)
    audio = batch.get("audio_wave", None) or batch.get("audio", None) or batch.get("audio_wav", None)
    video = batch.get("video_frames", None) or batch.get("video", None) or batch.get("frames", None)

    # If text is a HuggingFace mapping, it must be dict-like.
    # If missing, keep None and let model handle / or we will fail clearly.
    # Try signature-based call
    try:
        return model(text_input=text, audio_wave=audio, video_frames=video, modality_mask=None)[0]
    except Exception:
        pass

    # Some models return logits directly
    try:
        out = model(text_input=text, audio_wave=audio, video_frames=video)
        return out[0] if isinstance(out, (tuple, list)) else out
    except Exception:
        pass

    # Fallback: positional
    try:
        out = model(text, audio, video)
        return out[0] if isinstance(out, (tuple, list)) else out
    except Exception as e:
        raise RuntimeError(
            "Model forward failed. Batch keys={} ; text={} audio={} video={}. "
            "Likely collate outputs don't match your model forward."
            .format(list(batch.keys()), type(text), type(audio), type(video))
        ) from e


@torch.no_grad()
def evaluate(cfg_path: str, ckpt_path: str, split: str, batch_size: int, num_workers: int) -> Tuple[float, float]:
    cfg = _load_yaml(cfg_path)
    ds, split_csv, label_col = _build_dataset(cfg, split)

    # collate exists here (you already grepped it)
    from src.data.meld_dataset import collate_meld

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_meld,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Choose model module by ckpt name (baseline vs chado); both are supported
    module_try = []
    low = os.path.basename(ckpt_path).lower()
    if "chado" in low:
        module_try = ["src.models.chado_trimodal", "src.models.baseline_trimodal"]
    else:
        module_try = ["src.models.baseline_trimodal", "src.models.chado_trimodal"]

    last_err = None
    model = None
    for mn in module_try:
        try:
            cls = _find_first_module_class(mn)
            model = _instantiate_with_signature(cls, cfg)
            break
        except Exception as e:
            last_err = e
            continue
    if model is None:
        raise RuntimeError(f"Could not build model from {module_try}. Last error: {last_err}")

    model.to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = _extract_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[LOAD] ckpt={ckpt_path}")
    print(f"[LOAD] missing={len(missing)} unexpected={len(unexpected)} (strict=False)")

    all_y = []
    all_p = []

    for batch in loader:
        # label key from dataset/colllate
        y = batch.get("label", None)
        if y is None:
            # sometimes "labels"
            y = batch.get("labels", None)
        if y is None:
            raise RuntimeError(f"Batch has no label/labels. Keys={list(batch.keys())}")

        logits = _forward_logits(model, batch, device)
        pred = torch.argmax(logits, dim=-1)

        all_y.append(y.detach().cpu().numpy())
        all_p.append(pred.detach().cpu().numpy())

    y = np.concatenate(all_y).astype(int)
    p = np.concatenate(all_p).astype(int)

    acc = float((y == p).mean())
    wf1 = float(f1_score(y, p, average="weighted"))
    return acc, wf1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=6)
    args = ap.parse_args()

    acc, wf1 = evaluate(args.config, args.ckpt, args.split, args.batch_size, args.num_workers)
    print(f"\n=== MELD ({args.split}) ===")
    print(f"Acc={acc*100:.2f} | WF1={wf1*100:.2f}")


if __name__ == "__main__":
    main()
