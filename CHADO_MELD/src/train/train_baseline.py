import os
import yaml
import random
import inspect
import time
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup

from src.data.meld_dataset import (
    MeldDataset,
    collate_meld,
    build_label_map_from_order,
    EMO_ORDER_7,
)
from src.models.baseline_trimodal import TriModalBaseline


# ---------------- DDP ----------------

def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def get_rank():
    return dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ddp_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def ddp_all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    """All-reduce sum over ranks. x must be on GPU."""
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


# ---------------- Helpers ----------------

def safe_makedirs(p: str):
    os.makedirs(p, exist_ok=True)


def to_device_text(batch_text_input: Optional[Dict[str, torch.Tensor]], device):
    if batch_text_input is None:
        return None
    return {k: v.to(device, non_blocking=True) for k, v in batch_text_input.items()}


def compute_weighted_f1_from_confmat(confmat: np.ndarray) -> float:
    """
    Weighted F1 using confusion matrix:
      - per-class precision/recall/F1
      - weights = support (row sum)
    """
    eps = 1e-12
    support = confmat.sum(axis=1).astype(np.float64)
    total = support.sum() + eps

    tp = np.diag(confmat).astype(np.float64)
    fp = confmat.sum(axis=0).astype(np.float64) - tp
    fn = confmat.sum(axis=1).astype(np.float64) - tp

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)

    w = support / total
    return float((w * f1).sum())


def classwise_accuracy_from_confmat(confmat: np.ndarray) -> List[float]:
    eps = 1e-12
    support = confmat.sum(axis=1).astype(np.float64) + eps
    correct = np.diag(confmat).astype(np.float64)
    return [float(c / s) for c, s in zip(correct, support)]


def accuracy_from_confmat(confmat: np.ndarray) -> float:
    total = confmat.sum()
    if total <= 0:
        return 0.0
    return float(np.diag(confmat).sum() / total)


# ---------------- Signature-safe dataset builder ----------------

def build_meld_dataset(cfg, split_csv, label_map):
    """
    Create MeldDataset WITHOUT hardcoding keyword names that may differ in your repo.
    We filter kwargs based on MeldDataset.__init__ signature.
    """
    sig = inspect.signature(MeldDataset.__init__)
    accepted = set(sig.parameters.keys())  # includes "self"

    data_cfg = cfg.get("data", {})

    text_col = data_cfg.get("text_col", "text")
    label_col = data_cfg.get("label_col", "emotion")
    audio_col = data_cfg.get("audio_path_col", "audio_path")
    video_col = data_cfg.get("video_path_col", "video_path")
    utt_id_col = data_cfg.get("utt_id_col", "utt_id")

    candidates = {
        "csv_path": split_csv,
        "text_col": text_col,
        "label_col": label_col,
        "audio_path_col": audio_col,
        "video_path_col": video_col,
        "utt_id_col": utt_id_col,
        "num_frames": data_cfg["num_frames"],
        "frame_size": data_cfg["frame_size"],
        "sample_rate": data_cfg["sample_rate"],
        "max_audio_seconds": data_cfg["max_audio_seconds"],
        "label_map": label_map,
        "use_text": cfg["model"]["use_text"],
        "use_audio": cfg["model"]["use_audio"],
        "use_video": cfg["model"]["use_video"],
        # alternate names some repos use (kept but will be filtered by signature)
        "text_model_name": cfg["model"]["text_model_name"],
        "hf_model_name": cfg["model"]["text_model_name"],
        "bert_name": cfg["model"]["text_model_name"],
    }

    kwargs = {k: v for k, v in candidates.items() if k in accepted}

    required = [
        p.name for p in sig.parameters.values()
        if p.name != "self"
        and p.default is inspect._empty
        and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    missing = [r for r in required if r not in kwargs]
    if missing:
        if is_main():
            print("[FATAL] MeldDataset.__init__ signature requires:", required)
            print("[FATAL] We provided keys:", sorted(list(kwargs.keys())))
            print("[FATAL] Missing required keys:", missing)
            print("[FATAL] Full signature:", sig)
        raise TypeError(f"MeldDataset init missing required keys: {missing}")

    ds = MeldDataset(**kwargs)

    if not hasattr(ds, "tokenizer"):
        raise AttributeError(
            "MeldDataset instance has no attribute 'tokenizer'. "
            "Your collate_meld expects ds.tokenizer."
        )

    return ds


# ---------------- Class weights (stable) ----------------

def compute_class_weights_sqrtinv(train_ds, num_classes: int, device: torch.device, alpha: float = 0.5):
    """
    Stable class weights:
      w_c = (1 / freq_c^alpha), alpha=0.5 (sqrt inverse freq)
      normalize by mean so average weight ~1
    This avoids extreme weights that can destroy generalization.
    """
    # Gather labels from dataset in a safe way: use __getitem__ once per index would be heavy.
    # Most MeldDataset implementations store a dataframe or list. We try common patterns:
    labels = None
    if hasattr(train_ds, "df"):
        col = getattr(train_ds, "label_col", "emotion")
        labels = train_ds.df[col].tolist()
    elif hasattr(train_ds, "labels"):
        labels = list(train_ds.labels)
    elif hasattr(train_ds, "y"):
        labels = list(train_ds.y)

    if labels is None:
        # Fallback: sample lightweight by reading CSV again if dataset exposes csv_path
        if hasattr(train_ds, "csv_path"):
            import pandas as pd
            df = pd.read_csv(train_ds.csv_path)
            col = getattr(train_ds, "label_col", "emotion")
            labels = df[col].tolist()
        else:
            raise RuntimeError("Cannot compute class weights: dataset does not expose labels/df/csv_path.")

    # Map string labels to ids using label_map if present
    label_map = getattr(train_ds, "label_map", None)
    y = []
    for lab in labels:
        if isinstance(lab, str):
            if label_map is None:
                raise RuntimeError("Found string labels but dataset has no label_map.")
            y.append(int(label_map[lab]))
        else:
            y.append(int(lab))

    counts = np.bincount(np.array(y, dtype=np.int64), minlength=num_classes).astype(np.float64)
    freq = counts / max(counts.sum(), 1.0)
    freq = np.clip(freq, 1e-12, None)

    w = (1.0 / (freq ** alpha))
    w = w / w.mean()

    wt = torch.tensor(w, dtype=torch.float32, device=device)
    return wt


# ---------------- Freeze / Unfreeze ----------------

# def set_requires_grad(module: torch.nn.Module, flag: bool):
#     for p in module.parameters():
#         p.requires_grad = flag
def set_requires_grad(module, flag: bool):
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = flag


# def freeze_unfreeze_by_epoch(model_wrapped, epoch: int, cfg: dict):
#     """
#     Freeze encoders for first N epochs to stabilize training + improve generalization.
#     """
#     m = model_wrapped.module if hasattr(model_wrapped, "module") else model_wrapped
#     ft = int(cfg["train"].get("freeze_text_epochs", 0))
#     fa = int(cfg["train"].get("freeze_audio_epochs", 0))
#     fv = int(cfg["train"].get("freeze_video_epochs", 0))

#     # Best-effort attribute names (match typical baseline_trimodal)
#     if hasattr(m, "text_encoder"):
#         set_requires_grad(m.text_encoder, epoch >= ft)
#     if hasattr(m, "audio_encoder"):
#         set_requires_grad(m.audio_encoder, epoch >= fa)
#     if hasattr(m, "video_encoder"):
#         set_requires_grad(m.video_encoder, epoch >= fv)
def freeze_unfreeze_by_epoch(model, epoch, cfg):
    """
    Safe for ablations where encoders may be None.
    Works for both DDP-wrapped and non-DDP models.
    """
    m = model.module if hasattr(model, "module") else model

    ft = int(cfg["train"].get("freeze_text_epochs", 0))
    fa = int(cfg["train"].get("freeze_audio_epochs", 0))
    fv = int(cfg["train"].get("freeze_video_epochs", 0))

    # freeze if epoch < freeze_*_epochs, unfreeze otherwise
    set_requires_grad(getattr(m, "text_encoder", None), epoch >= ft)
    set_requires_grad(getattr(m, "audio_encoder", None), epoch >= fa)
    set_requires_grad(getattr(m, "video_encoder", None), epoch >= fv)


# ---------------- Optimizer (differential LR) ----------------

def build_optimizer(model_wrapped, cfg: dict):
    wd = float(cfg["train"]["weight_decay"])
    lr_default = float(cfg["train"].get("lr", 2e-5))
    lr_enc = float(cfg["train"].get("lr_encoders", lr_default))
    lr_head = float(cfg["train"].get("lr_head", lr_default))

    m = model_wrapped.module if hasattr(model_wrapped, "module") else model_wrapped

    head_params = []
    enc_params = []

    for n, p in m.named_parameters():
        if not p.requires_grad:
            continue
        nlow = n.lower()
        if any(k in nlow for k in ["classifier", "proj", "fusion", "gate", "head"]):
            head_params.append(p)
        else:
            enc_params.append(p)

    opt = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": lr_enc},
            {"params": head_params, "lr": lr_head},
        ],
        weight_decay=wd,
    )
    if is_main():
        print(f"[OPT] lr_encoders={lr_enc:.2e} lr_head={lr_head:.2e} weight_decay={wd}")
        print(f"[OPT] params enc={len(enc_params)} head={len(head_params)}")
    return opt


# ---------------- Evaluation ----------------

@torch.no_grad()
def evaluate(model_wrapped, loader, device, num_classes: int, amp: bool) -> Tuple[float, float, np.ndarray]:
    """
    Returns: (accuracy, weighted_f1, confmat)
    """
    model_wrapped.eval()
    confmat = np.zeros((num_classes, num_classes), dtype=np.int64)

    for batch in loader:
        labels = batch.labels.to(device, non_blocking=True)

        text = to_device_text(batch.text_input, device) if batch.text_input else None
        audio = batch.audio_wave.to(device, non_blocking=True) if batch.audio_wave is not None else None
        video = batch.video_frames.to(device, non_blocking=True) if batch.video_frames is not None else None

        with autocast("cuda", enabled=amp):
            logits, _, _ = model_wrapped(
                text_input=text,
                audio_wave=audio,
                video_frames=video,
                modality_mask=None,
            )
            preds = torch.argmax(logits, dim=-1)

        # Update local confmat
        p = preds.detach().cpu().numpy().astype(np.int64)
        y = labels.detach().cpu().numpy().astype(np.int64)
        for yi, pi in zip(y, p):
            confmat[yi, pi] += 1

    # DDP: reduce confusion matrix
    if dist.is_available() and dist.is_initialized():
        cm_t = torch.tensor(confmat, device=device, dtype=torch.long)
        ddp_all_reduce_sum(cm_t)
        confmat = cm_t.detach().cpu().numpy()

    acc = accuracy_from_confmat(confmat)
    f1w = compute_weighted_f1_from_confmat(confmat)
    return acc, f1w, confmat


# ---------------- MAIN ----------------

def main():
    setup_ddp()
    rank = get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank)

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    seed_all(int(cfg["train"]["seed"]) + rank)

    # Label map fixed to EMO_ORDER_7
    label_map = build_label_map_from_order(EMO_ORDER_7)
    num_classes = int(cfg["data"]["num_classes"])

    # ---------------- DATASETS ----------------
    train_ds = build_meld_dataset(cfg, cfg["data"]["train_csv"], label_map)
    val_ds = build_meld_dataset(cfg, cfg["data"]["val_csv"], label_map)
    test_ds = build_meld_dataset(cfg, cfg["data"]["test_csv"], label_map)

    # Samplers
    train_sampler = DistributedSampler(train_ds, shuffle=True) if dist.is_available() and dist.is_initialized() else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if dist.is_available() and dist.is_initialized() else None
    test_sampler = DistributedSampler(test_ds, shuffle=False) if dist.is_available() and dist.is_initialized() else None

    # Loaders
    bs = int(cfg["train"]["batch_size_per_gpu"])
    nw = int(cfg["train"]["num_workers"])

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda b: collate_meld(
            b,
            train_ds.tokenizer,
            cfg["model"]["use_text"],
            cfg["model"]["use_audio"],
            cfg["model"]["use_video"],
        ),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        sampler=val_sampler,
        shuffle=False,
        num_workers=max(2, nw // 2),
        pin_memory=True,
        drop_last=False,
        collate_fn=lambda b: collate_meld(
            b,
            val_ds.tokenizer,
            cfg["model"]["use_text"],
            cfg["model"]["use_audio"],
            cfg["model"]["use_video"],
        ),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=bs,
        sampler=test_sampler,
        shuffle=False,
        num_workers=max(2, nw // 2),
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

    # ---------------- MODEL ----------------
    model = TriModalBaseline(
        text_model_name=cfg["model"]["text_model_name"],
        audio_model_name=cfg["model"]["audio_model_name"],
        video_model_name=cfg["model"]["video_model_name"],
        num_classes=num_classes,
        proj_dim=int(cfg["model"]["proj_dim"]),
        dropout=float(cfg["model"]["dropout"]),
        use_text=bool(cfg["model"]["use_text"]),
        use_audio=bool(cfg["model"]["use_audio"]),
        use_video=bool(cfg["model"]["use_video"]),
        use_gated_fusion=bool(cfg["model"]["use_gated_fusion"]),
    ).to(device)

    # Explicitly ensure gradient checkpointing is OFF (DDP-safe)
    for m in model.modules():
        if hasattr(m, "gradient_checkpointing"):
            try:
                m.gradient_checkpointing = False
            except Exception:
                pass
        if hasattr(m, "gradient_checkpointing_disable"):
            try:
                m.gradient_checkpointing_disable()
            except Exception:
                pass

    if dist.is_available() and dist.is_initialized():
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    # ---------------- LOSS SETTINGS ----------------
    use_class_weights = bool(cfg["train"].get("use_class_weights", False))
    label_smoothing = float(cfg["train"].get("label_smoothing", 0.0))
    loss_type = str(cfg["train"].get("loss_type", "weighted_ce")).lower()

    # Stable weights
    ce_weight = None
    if use_class_weights:
        ce_weight = compute_class_weights_sqrtinv(train_ds, num_classes, device=device, alpha=0.5)
        if is_main():
            print("[INFO] Using sqrt-inv class weights:", ce_weight.detach().cpu().tolist())

    # ---------------- OPTIM / SCHED ----------------
    optimizer = build_optimizer(model, cfg)

    total_steps = int(cfg["train"]["epochs"]) * len(train_loader)
    warmup_steps = int(total_steps * float(cfg["train"]["warmup_ratio"]))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    amp = bool(cfg["train"]["amp"])
    scaler = GradScaler("cuda", enabled=amp)

    grad_accum = int(cfg["train"].get("grad_accum_steps", 1))
    max_grad_norm = float(cfg["train"].get("max_grad_norm", 1.0))

    # ---------------- LOGGING / CKPT ----------------
    run_name = cfg.get("logging", {}).get("run_name", "baseline_trimodal_meld")
    out_dir = cfg.get("logging", {}).get("out_dir", "./runs")
    safe_makedirs(out_dir)
    best_path = os.path.join(out_dir, f"{run_name}_best.pt")

    early_pat = int(cfg["train"].get("early_stop_patience", 8))
    best_val_f1w = -1.0
    bad_epochs = 0

    # ---------------- TRAIN LOOP ----------------
    for epoch in range(int(cfg["train"]["epochs"])):
        t0 = time.time()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Freeze/unfreeze per epoch
        freeze_unfreeze_by_epoch(model, epoch, cfg)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        n_steps = 0

        # Track train accuracy quickly (approx, on-the-fly)
        train_correct = 0
        train_total = 0

        for it, batch in enumerate(train_loader):
            labels = batch.labels.to(device, non_blocking=True)

            text = to_device_text(batch.text_input, device) if batch.text_input else None
            audio = batch.audio_wave.to(device, non_blocking=True) if batch.audio_wave is not None else None
            video = batch.video_frames.to(device, non_blocking=True) if batch.video_frames is not None else None

            with autocast("cuda", enabled=amp):
                logits, _, _ = model(
                    text_input=text,
                    audio_wave=audio,
                    video_frames=video,
                    modality_mask=None,
                )

                if loss_type in ("weighted_ce", "ce", "cross_entropy"):
                    loss = F.cross_entropy(
                        logits,
                        labels,
                        weight=ce_weight,
                        label_smoothing=label_smoothing,
                    )
                else:
                    # Keep default CE if unknown
                    loss = F.cross_entropy(
                        logits,
                        labels,
                        weight=ce_weight,
                        label_smoothing=label_smoothing,
                    )

                loss = loss / max(1, grad_accum)

            scaler.scale(loss).backward()

            # Train acc estimate
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                train_correct += int((preds == labels).sum().item())
                train_total += int(labels.numel())

            if (it + 1) % grad_accum == 0:
                # Unscale + clip
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            running_loss += float(loss.item()) * max(1, grad_accum)
            n_steps += 1

        # DDP reduce train stats
        if dist.is_available() and dist.is_initialized():
            tc = torch.tensor([train_correct, train_total], device=device, dtype=torch.long)
            ddp_all_reduce_sum(tc)
            train_correct, train_total = int(tc[0].item()), int(tc[1].item())

        train_acc = (train_correct / max(train_total, 1)) if train_total > 0 else 0.0
        avg_loss = running_loss / max(n_steps, 1)

        # Eval
        val_acc, val_f1w, _ = evaluate(model, val_loader, device, num_classes, amp=amp)

        if is_main():
            dt = time.time() - t0
            print(f"[epoch {epoch}] loss={avg_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f} val_f1w={val_f1w:.4f} time={dt:.1f}s")

        # Save best by val_f1w
        improved = val_f1w > best_val_f1w + 1e-6
        if improved:
            best_val_f1w = val_f1w
            bad_epochs = 0
            if is_main():
                state = {
                    "epoch": epoch,
                    "cfg": cfg,
                    "model": (model.module.state_dict() if hasattr(model, "module") else model.state_dict()),
                    "best_val_f1w": best_val_f1w,
                }
                torch.save(state, best_path)
                print(f"[OK] Saved best checkpoint: {best_path} (val_f1w={best_val_f1w:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= early_pat:
                if is_main():
                    print(f"[EARLY STOP] No val_f1w improvement for {early_pat} epochs.")
                break

    # ---------------- TEST (best checkpoint) ----------------
    ddp_barrier()
    if is_main():
        print("[OK] Loaded best checkpoint for test:", best_path)

    ckpt = torch.load(best_path, map_location="cpu")
    state = ckpt["model"]

    # Load to non-wrapped model
    m = model.module if hasattr(model, "module") else model
    m.load_state_dict(state, strict=True)
    ddp_barrier()

    test_acc, test_f1w, confmat = evaluate(model, test_loader, device, num_classes, amp=amp)

    # Print final metrics
    if is_main():
        print("\n===== FINAL TEST RESULTS =====")
        print(f"Weighted Test Accuracy : {test_acc:.4f}")
        print(f"Weighted Test F1       : {test_f1w:.4f}\n")

        cw_acc = classwise_accuracy_from_confmat(confmat)
        print("Class-wise Accuracy:")
        for i, a in enumerate(cw_acc):
            name = EMO_ORDER_7[i] if i < len(EMO_ORDER_7) else str(i)
            print(f"  {i:02d} ({name:<8}): {a:.4f}")

        print("\nConfusion Matrix (rows=true, cols=pred):")
        print(confmat)
        print("\n✅ TRAINING COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_ddp()
