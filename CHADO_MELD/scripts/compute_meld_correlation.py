#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import inspect
import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
import yaml

from src.data.meld_dataset import (
    MeldDataset,
    collate_meld,
    EMO_ORDER_7,
    build_label_map_from_order,
)
from src.models.chado_trimodal import CHADOTrimodal


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_state_dict(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "module"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
        if any(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj
    raise RuntimeError("Cannot extract state_dict from checkpoint format.")


def _build_meld_dataset_from_signature(cfg: Dict[str, Any], csv_path: str) -> MeldDataset:
    """
    Build MeldDataset using *your* repo's signature safely:
    - Fill parameters by name
    - Respect positional-only parameters if present
    - Never mix positional+keyword for same field (prevents 'multiple values' errors)
    """
    data = cfg["data"]
    model_cfg = cfg["model"]

    label_map = build_label_map_from_order(EMO_ORDER_7)

    # Values we can provide (based on your confirmed CSV header + yaml)
    provided = {
        "csv_path": csv_path,
        "text_col": data.get("text_col", "text"),
        "label_col": data.get("label_col", "emotion"),   # MUST be 'emotion' for your CSV
        "audio_path_col": data.get("audio_path_col", "audio_path"),
        "video_path_col": data.get("video_path_col", "video_path"),
        "utt_id_col": data.get("utt_id_col", "utt_id"),
        "num_frames": int(data.get("num_frames", 8)),
        "frame_size": int(data.get("frame_size", 224)),
        "sample_rate": int(data.get("sample_rate", 16000)),
        "max_audio_seconds": float(data.get("max_audio_seconds", 6.0)),
        "label_map": label_map,
        "use_text": bool(model_cfg.get("use_text", True)),
        "use_audio": bool(model_cfg.get("use_audio", True)),
        "use_video": bool(model_cfg.get("use_video", True)),
        "text_model_name": model_cfg.get("text_model_name", "roberta-base"),
    }

    sig = inspect.signature(MeldDataset.__init__)
    params = list(sig.parameters.values())[1:]  # drop self

    # Build positional-only args in correct order
    pos_args = []
    kwargs = {}

    for p in params:
        name = p.name

        # Only feed what the signature actually accepts
        if name not in provided:
            # allow optional params to use default
            continue

        if p.kind == inspect.Parameter.POSITIONAL_ONLY:
            pos_args.append(provided[name])
        else:
            kwargs[name] = provided[name]

    return MeldDataset(*pos_args, **kwargs)


def build_dataset(cfg: Dict[str, Any], split: str) -> Tuple[MeldDataset, str]:
    data = cfg["data"]
    csv_path = data.get(f"{split}_csv", None) or data.get("test_csv", None)
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found for split={split}: {csv_path}")

    ds = _build_meld_dataset_from_signature(cfg, csv_path)
    return ds, csv_path


def ambiguity_from_probs(probs: np.ndarray, mode: str) -> np.ndarray:
    if mode == "entropy":
        return -(probs * np.log(probs + 1e-12)).sum(axis=1)
    if mode == "1-max":
        return 1.0 - probs.max(axis=1)
    if mode == "margin":
        top2 = np.partition(probs, -2, axis=1)[:, -2:]
        p1 = np.max(top2, axis=1)
        p2 = np.min(top2, axis=1)
        return 1.0 - (p1 - p2)
    raise ValueError(f"Unknown ambiguity mode: {mode}")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--ambiguity", default="entropy", choices=["entropy", "1-max", "margin"])
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    ds, csv_path = build_dataset(cfg, args.split)
    print(f"[DATA] split_csv={csv_path}")
    print(f"[DATA] label_col_used={cfg['data'].get('label_col', 'emotion')}")
    print(f"[DATA] text_model_name={cfg['model'].get('text_model_name', 'roberta-base')}")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=lambda b: collate_meld(
            b,
            ds.tokenizer,
            cfg["model"].get("use_text", True),
            cfg["model"].get("use_audio", True),
            cfg["model"].get("use_video", True),
        ),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CHADOTrimodal(
        text_model_name=cfg["model"]["text_model_name"],
        audio_model_name=cfg["model"]["audio_model_name"],
        video_model_name=cfg["model"]["video_model_name"],
        num_classes=cfg["data"]["num_classes"],
        proj_dim=cfg["model"]["proj_dim"],
        dropout=cfg["model"]["dropout"],
        use_text=cfg["model"]["use_text"],
        use_audio=cfg["model"]["use_audio"],
        use_video=cfg["model"]["use_video"],
        use_gated_fusion=cfg["model"]["use_gated_fusion"],
        use_causal=cfg.get("chado", {}).get("use_causal", False),
        use_hyperbolic=cfg.get("chado", {}).get("use_hyperbolic", False),
        use_transport=cfg.get("chado", {}).get("use_transport", False),
        use_refinement=cfg.get("chado", {}).get("use_refinement", False),
    ).to(device)
    model.eval()

    ckpt_obj = torch.load(args.ckpt, map_location="cpu")
    sd = extract_state_dict(ckpt_obj)
    missing, unexpected = model.load_state_dict(sd, strict=False)

    print(f"[LOAD] ckpt={args.ckpt}")
    print(f"[LOAD] missing={len(missing)} unexpected={len(unexpected)} (strict=False)")

    all_probs = []
    all_labels = []

    for batch in loader:
        labels = batch.labels.to(device)

        text = {k: v.to(device) for k, v in batch.text_input.items()} if batch.text_input else None
        audio = batch.audio_wave.to(device) if batch.audio_wave is not None else None
        video = batch.video_frames.to(device) if batch.video_frames is not None else None

        logits, _, _ = model(text_input=text, audio_wave=audio, video_frames=video, modality_mask=None)
        probs = torch.softmax(logits, dim=1)

        all_probs.append(probs.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0).astype(int)

    preds = probs.argmax(axis=1)
    error = (preds != labels).astype(float)  # 0/1

    amb = ambiguity_from_probs(probs, args.ambiguity)

    pr, pp = pearsonr(amb, error)
    sr, sp = spearmanr(amb, error)

    print("\n=== MELD Correlation (ambiguity vs error) ===")
    print(f"Ambiguity metric: {args.ambiguity}")
    print(f"Pearson r={pr:.4f} | p={pp:.4g}")
    print(f"Spearman ρ={sr:.4f} | p={sp:.4g}")

    if args.out_csv:
        import pandas as pd
        df = pd.DataFrame({"ambiguity": amb, "error": error})
        df.to_csv(args.out_csv, index=False)
        print(f"[OK] Saved per-sample CSV → {args.out_csv}")


if __name__ == "__main__":
    main()
