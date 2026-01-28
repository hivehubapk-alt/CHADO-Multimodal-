#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import yaml
import inspect
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from src.data.meld_dataset import (
    MeldDataset,
    collate_meld,
    EMO_ORDER_7,
    build_label_map_from_order,
)
from src.models.chado_trimodal import CHADOTrimodal


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _extract_state_dict(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    # Common checkpoint wrappers
    if isinstance(ckpt_obj, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "module"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
        # If it already looks like a raw state_dict
        if any(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj
    raise TypeError("Unsupported checkpoint format: cannot extract state_dict.")


def _load_into_chado_base(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    """
    Your CHADOTrimodal state_dict keys are like: base.text_encoder...
    Baseline ckpt keys are like: text_encoder...
    So we prefix baseline keys with 'base.' to match.
    """
    if any(k.startswith("base.") for k in state_dict.keys()):
        sd = state_dict
    else:
        sd = {("base." + k): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[LOAD] missing={len(missing)} unexpected={len(unexpected)} (strict=False)")


def _build_meld_dataset(cfg: Dict[str, Any], split: str, label_map: Dict[str, int]) -> MeldDataset:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    split_key = f"{split}_csv"
    if split_key not in data_cfg:
        raise KeyError(f"Config missing data.{split_key}")

    csv_path = data_cfg[split_key]

    # Your CSV uses 'emotion' (NOT 'label')
    label_col = data_cfg.get("label_col", "emotion")
    text_col = data_cfg.get("text_col", "text")
    audio_path_col = data_cfg.get("audio_path_col", "audio_path")
    video_path_col = data_cfg.get("video_path_col", "video_path")
    utt_id_col = data_cfg.get("utt_id_col", "utt_id")

    num_frames = data_cfg.get("num_frames", 8)
    frame_size = data_cfg.get("frame_size", 224)
    sample_rate = data_cfg.get("sample_rate", 16000)
    max_audio_seconds = float(data_cfg.get("max_audio_seconds", 6.0))

    use_text = bool(model_cfg.get("use_text", True))
    use_audio = bool(model_cfg.get("use_audio", True))
    use_video = bool(model_cfg.get("use_video", True))
    text_model_name = model_cfg.get("text_model_name", "roberta-base")

    # Build kwargs ONLY for arguments that exist in your MeldDataset.__init__
    sig = inspect.signature(MeldDataset.__init__)
    params = set(sig.parameters.keys())

    kwargs = {}
    if "csv_path" in params: kwargs["csv_path"] = csv_path
    if "text_col" in params: kwargs["text_col"] = text_col
    if "label_col" in params: kwargs["label_col"] = label_col
    if "audio_path_col" in params: kwargs["audio_path_col"] = audio_path_col
    if "video_path_col" in params: kwargs["video_path_col"] = video_path_col
    if "utt_id_col" in params: kwargs["utt_id_col"] = utt_id_col

    if "num_frames" in params: kwargs["num_frames"] = num_frames
    if "frame_size" in params: kwargs["frame_size"] = frame_size
    if "sample_rate" in params: kwargs["sample_rate"] = sample_rate
    if "max_audio_seconds" in params: kwargs["max_audio_seconds"] = max_audio_seconds

    if "label_map" in params: kwargs["label_map"] = label_map
    if "text_model_name" in params: kwargs["text_model_name"] = text_model_name

    if "use_text" in params: kwargs["use_text"] = use_text
    if "use_audio" in params: kwargs["use_audio"] = use_audio
    if "use_video" in params: kwargs["use_video"] = use_video

    print(f"[DATA] split_csv={csv_path}")
    print(f"[DATA] label_col_used={label_col}")
    print(f"[DATA] text_model_name={text_model_name}")

    return MeldDataset(**kwargs)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--report", action="store_true", help="print full per-class report")
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    label_map = build_label_map_from_order(EMO_ORDER_7)
    id2label = {v: k for k, v in label_map.items()}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset + loader
    ds = _build_meld_dataset(cfg, args.split, label_map)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_meld(
            b,
            ds.tokenizer,
            cfg["model"]["use_text"],
            cfg["model"]["use_audio"],
            cfg["model"]["use_video"],
        ),
    )

    # Model (CHADO wrapper with CHADO parts disabled for baseline eval)
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
        use_causal=False,
        use_hyperbolic=False,
        use_transport=False,
        use_refinement=False,
    ).to(device)
    model.eval()

    ckpt_obj = torch.load(args.ckpt, map_location="cpu")
    sd = _extract_state_dict(ckpt_obj)

    print(f"[LOAD] ckpt={args.ckpt}")
    _load_into_chado_base(model, sd)

    # Predict
    y_true, y_pred = [], []
    for batch in loader:
        labels = batch.labels.to(device)
        text = {k: v.to(device) for k, v in batch.text_input.items()} if batch.text_input is not None else None
        audio = batch.audio_wave.to(device) if batch.audio_wave is not None else None
        video = batch.video_frames.to(device) if batch.video_frames is not None else None

        logits, _, _ = model(text_input=text, audio_wave=audio, video_frames=video, modality_mask=None)
        pred = torch.argmax(logits, dim=1)

        y_true.append(labels.detach().cpu().numpy())
        y_pred.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # Metrics (Weighted to match your "WF1" convention)
    acc = accuracy_score(y_true, y_pred)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    print("\n===== MELD METRICS (split: %s) =====" % args.split)
    print(f"Accuracy           : {acc:.4f}")
    print(f"Precision (weighted): {p_w:.4f}")
    print(f"Recall (weighted)   : {r_w:.4f}")
    print(f"F1 (weighted)       : {f1_w:.4f}")
    print(f"Precision (macro)   : {p_m:.4f}")
    print(f"Recall (macro)      : {r_m:.4f}")
    print(f"F1 (macro)          : {f1_m:.4f}")

    if args.report:
        print("\n--- Per-class report ---")
        target_names = [id2label[i] for i in range(len(id2label))]
        print(classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0))

        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix (rows=true, cols=pred):")
        print(cm)


if __name__ == "__main__":
    main()
