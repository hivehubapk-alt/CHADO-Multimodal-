# scripts/eval_meld_ckpt_metrics.py
#!/usr/bin/env python3
import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader

from src.data.meld_dataset import MeldDataset, collate_meld, build_label_map_from_order, EMO_ORDER_7
from src.models.chado_trimodal import CHADOTrimodal


def build_dataset(cfg, split: str, label_map):
    csv_path = cfg["data"][f"{split}_csv"]
    label_col = cfg["data"].get("label_col", "emotion")
    if label_col == "label":
        label_col = "emotion"

    ds = MeldDataset(
        csv_path,
        cfg["model"]["text_model_name"],
        label_map,
        cfg["data"]["text_col"],
        label_col,
        cfg["data"]["audio_path_col"],
        cfg["data"]["video_path_col"],
        cfg["data"]["utt_id_col"],
        int(cfg["data"]["num_frames"]),
        int(cfg["data"]["frame_size"]),
        int(cfg["data"]["sample_rate"]),
        float(cfg["data"]["max_audio_seconds"]),
        bool(cfg["model"]["use_text"]),
        bool(cfg["model"]["use_audio"]),
        bool(cfg["model"]["use_video"]),
        int(cfg["train"].get("seed", 42)),
    )
    return ds


@torch.no_grad()
def eval_metrics(model, loader, device, amp: bool):
    model.eval()
    num_classes = loader.dataset.num_classes if hasattr(loader.dataset, "num_classes") else 7

    tp = torch.zeros(num_classes, device=device)
    fp = torch.zeros(num_classes, device=device)
    fn = torch.zeros(num_classes, device=device)
    total = 0
    correct = 0

    for batch in loader:
        labels = batch.labels.to(device)
        text = {k: v.to(device) for k, v in batch.text_input.items()} if batch.text_input else None
        audio = batch.audio_wave.to(device) if batch.audio_wave is not None else None
        video = batch.video_frames.to(device) if batch.video_frames is not None else None

        with torch.cuda.amp.autocast(enabled=amp):
            out = model(text_input=text, audio_wave=audio, video_frames=video, modality_mask=None, compute_losses=False)
            logits = out.logits

        preds = logits.argmax(dim=-1)
        total += labels.numel()
        correct += (preds == labels).sum().item()

        for c in range(num_classes):
            tp[c] += ((preds == c) & (labels == c)).sum()
            fp[c] += ((preds == c) & (labels != c)).sum()
            fn[c] += ((preds != c) & (labels == c)).sum()

    acc = correct / max(1, total)
    precision_c = tp / (tp + fp).clamp(min=1.0)
    recall_c = tp / (tp + fn).clamp(min=1.0)
    f1_c = (2 * precision_c * recall_c) / (precision_c + recall_c).clamp(min=1e-8)

    support = tp + fn
    denom = support.sum().clamp(min=1.0)

    precision_w = (precision_c * support).sum() / denom
    recall_w = (recall_c * support).sum() / denom
    f1_w = (f1_c * support).sum() / denom

    return float(acc), float(precision_w.item()), float(recall_w.item()), float(f1_w.item())


def load_ckpt(model, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    model.load_state_dict(sd, strict=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=6)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_map = build_label_map_from_order(EMO_ORDER_7)
    ds = build_dataset(cfg, args.split, label_map)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_meld(b, ds.tokenizer, cfg["model"]["use_text"], cfg["model"]["use_audio"], cfg["model"]["use_video"]),
    )

    offline = bool(int(os.environ.get("TRANSFORMERS_OFFLINE", "0"))) or bool(int(os.environ.get("HF_HUB_OFFLINE", "0")))

    model = CHADOTrimodal(
        text_model_name=cfg["model"]["text_model_name"],
        audio_model_name=cfg["model"]["audio_model_name"],
        video_model_name=cfg["model"]["video_model_name"],
        num_classes=int(cfg["data"]["num_classes"]),
        proj_dim=int(cfg["model"]["proj_dim"]),
        dropout=float(cfg["model"]["dropout"]),
        use_text=True, use_audio=True, use_video=True,
        use_gated_fusion=bool(cfg["model"].get("use_gated_fusion", True)),
        use_causal=bool(cfg.get("chado", {}).get("use_causal", True)),
        use_hyperbolic=bool(cfg.get("chado", {}).get("use_hyperbolic", True)),
        use_transport=bool(cfg.get("chado", {}).get("use_transport", True)),
        use_refinement=bool(cfg.get("chado", {}).get("use_refinement", True)),
        local_files_only=offline,
    ).to(device)

    load_ckpt(model, args.ckpt)
    amp = bool(cfg["train"].get("amp", True))
    acc, pw, rw, f1w = eval_metrics(model, loader, device, amp)

    print("Accuracy              :", f"{acc:.4f}")
    print("Precision (weighted)  :", f"{pw:.4f}")
    print("Recall (weighted)     :", f"{rw:.4f}")
    print("F1 (weighted)         :", f"{f1w:.4f}")


if __name__ == "__main__":
    main()
