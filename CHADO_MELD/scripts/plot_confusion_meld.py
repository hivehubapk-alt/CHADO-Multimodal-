import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import inspect

from torch.utils.data import DataLoader
from torch.amp import autocast

from src.data.meld_dataset import (
    MeldDataset,
    collate_meld,
    build_label_map_from_order,
    EMO_ORDER_7,
)

from src.models.baseline_trimodal import TriModalBaseline


def _unwrap_state_dict(ckpt_obj):
    """
    Supports:
      - raw state_dict (dict[str, Tensor])
      - bundled checkpoints with keys like: {"model": state_dict, "epoch":..., ...}
      - {"state_dict": state_dict}
    Also strips "module." prefix if present.
    """
    if not isinstance(ckpt_obj, dict):
        raise TypeError("Checkpoint is not a dict.")

    # Common wrappers
    if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
        sd = ckpt_obj["model"]
    elif "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
        sd = ckpt_obj["state_dict"]
    else:
        # assume it's already a state_dict
        sd = ckpt_obj

    # Strip DDP prefix
    cleaned = {}
    for k, v in sd.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        cleaned[nk] = v
    return cleaned


def build_dataset(cfg, csv_path, label_map):
    sig = inspect.signature(MeldDataset.__init__)
    accepted = set(sig.parameters.keys())

    # Include text_model_name ONLY if dataset requires it
    kwargs = {
        "csv_path": csv_path,
        "text_col": cfg["data"]["text_col"],
        "label_col": cfg["data"]["label_col"],
        "audio_path_col": cfg["data"]["audio_path_col"],
        "video_path_col": cfg["data"]["video_path_col"],
        "utt_id_col": cfg["data"]["utt_id_col"],
        "num_frames": cfg["data"]["num_frames"],
        "frame_size": cfg["data"]["frame_size"],
        "sample_rate": cfg["data"]["sample_rate"],
        "max_audio_seconds": cfg["data"]["max_audio_seconds"],
        "label_map": label_map,
        "use_text": cfg["model"]["use_text"],
        "use_audio": cfg["model"]["use_audio"],
        "use_video": cfg["model"]["use_video"],
        "text_model_name": cfg["model"]["text_model_name"],
    }

    kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    ds = MeldDataset(**kwargs)

    if not hasattr(ds, "tokenizer"):
        raise AttributeError("MeldDataset has no attribute 'tokenizer'. collate_meld expects ds.tokenizer.")

    return ds


@torch.no_grad()
def main():
    cfg = yaml.safe_load(open("configs/baseline_meld.yaml"))

    device = torch.device("cuda")
    label_map = build_label_map_from_order(EMO_ORDER_7)

    # For axis labels: stable order = EMO_ORDER_7 (your repo’s canonical order)
    class_names = list(EMO_ORDER_7)

    # Dataset
    test_ds = build_dataset(cfg, cfg["data"]["test_csv"], label_map)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["train"]["batch_size_per_gpu"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        drop_last=False,
        collate_fn=lambda b: collate_meld(
            b,
            test_ds.tokenizer,
            cfg["model"]["use_text"],
            cfg["model"]["use_audio"],
            cfg["model"]["use_video"],
        ),
    )

    # Model
    model = TriModalBaseline(
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
    ).to(device)

    ckpt_path = "runs/baseline_trimodal_meld_best.pt"
    ckpt_obj = torch.load(ckpt_path, map_location="cpu")
    state_dict = _unwrap_state_dict(ckpt_obj)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        # Not fatal, but print so you can verify
        print("[WARN] load_state_dict strict=False")
        if missing:
            print(f"[WARN] missing keys: {len(missing)} (showing up to 20)")
            print(missing[:20])
        if unexpected:
            print(f"[WARN] unexpected keys: {len(unexpected)} (showing up to 20)")
            print(unexpected[:20])

    model.eval()

    # Confusion matrix (raw counts)
    num_classes = cfg["data"]["num_classes"]
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for batch in test_loader:
        labels = batch.labels.to(device)

        text = {k: v.to(device) for k, v in batch.text_input.items()} if batch.text_input else None
        audio = batch.audio_wave.to(device) if batch.audio_wave is not None else None
        video = batch.video_frames.to(device) if batch.video_frames is not None else None

        with autocast("cuda", enabled=cfg["train"]["amp"]):
            logits, _, _ = model(
                text_input=text,
                audio_wave=audio,
                video_frames=video,
                modality_mask=None,
            )

        preds = torch.argmax(logits, dim=-1)

        t = labels.detach().cpu().numpy()
        p = preds.detach().cpu().numpy()
        for ti, pi in zip(t, p):
            cm[int(ti), int(pi)] += 1

    os.makedirs("analysis", exist_ok=True)
    np.save("analysis/confusion_matrix.npy", cm)

    # Normalize row-wise for visualization (each row sums to 1)
    row_sums = np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    cm_norm = cm / row_sums

    # Plot
    plt.figure(figsize=(8.5, 7.5))
    im = plt.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, fraction=0.046)

    plt.xticks(range(num_classes), class_names, rotation=45, ha="right")
    plt.yticks(range(num_classes), class_names)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("MELD Confusion Matrix (Normalized)")

    # Annotate with normalized values
    for i in range(num_classes):
        for j in range(num_classes):
            v = cm_norm[i, j]
            plt.text(
                j, i, f"{v:.2f}",
                ha="center", va="center",
                fontsize=8,
                color="white" if v > 0.5 else "black",
            )

    plt.tight_layout()
    out_png = "analysis/confusion_matrix_meld.png"
    plt.savefig(out_png, dpi=300)
    plt.close()

    print("Saved:")
    print(" - analysis/confusion_matrix.npy")
    print(f" - {out_png}")
    print("Note: This uses checkpoint:", ckpt_path)


if __name__ == "__main__":
    main()
