# src/train/train_chado.py
import os
import json
import time
import yaml
import math
import random
import inspect
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup

from src.data.meld_dataset import MeldDataset, collate_meld, build_label_map_from_order, EMO_ORDER_7
from src.models.chado_trimodal import CHADOTrimodal


# -------------------- DDP utils --------------------
def ddp_init():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return True
    return False

def ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0

def rank():
    return dist.get_rank() if dist.is_initialized() else 0

def world():
    return dist.get_world_size() if dist.is_initialized() else 1

def seed_all(seed: int):
    random.seed(seed + rank())
    np.random.seed(seed + rank())
    torch.manual_seed(seed + rank())
    torch.cuda.manual_seed_all(seed + rank())


# -------------------- ckpt helpers --------------------
def _is_state_dict(obj: Any) -> bool:
    if not isinstance(obj, dict) or len(obj) == 0:
        return False
    nt = sum(torch.is_tensor(v) for v in obj.values())
    return nt >= max(1, int(0.8 * len(obj)))

def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if _is_state_dict(ckpt):
        return ckpt
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "module", "weights"]:
            if k in ckpt and _is_state_dict(ckpt[k]):
                return ckpt[k]
        # one-level deeper
        for v in ckpt.values():
            if _is_state_dict(v):
                return v
            if isinstance(v, dict):
                for vv in v.values():
                    if _is_state_dict(vv):
                        return vv
    raise ValueError("Could not find a valid state_dict in checkpoint.")

def _strip_prefix(key: str) -> str:
    for p in ["module.", "model.", "net."]:
        if key.startswith(p):
            return key[len(p):]
    return key

def load_baseline_into_chado(model: CHADOTrimodal, baseline_ckpt_path: str):
    """
    Loads TriModalBaseline weights into CHADOTrimodal.base.* safely.
    Works even if baseline ckpt keys are saved without 'base.' prefix.
    """
    ckpt = torch.load(baseline_ckpt_path, map_location="cpu")
    sd = _extract_state_dict(ckpt)
    cleaned = {_strip_prefix(k): v for k, v in sd.items()}

    model_sd = model.state_dict()
    keys = set(model_sd.keys())

    mapped = {}
    for k, v in cleaned.items():
        # direct
        if k in keys:
            mapped[k] = v
            continue
        # baseline -> chado wrapper
        kb = "base." + k
        if kb in keys:
            mapped[kb] = v
            continue

    missing, unexpected = model.load_state_dict(mapped, strict=False)

    if is_main():
        print(f"[LOAD] baseline_ckpt={baseline_ckpt_path}")
        print(f"[LOAD] extracted_keys={len(sd)} mapped={len(mapped)}")
        print(f"[LOAD] missing={len(missing)} unexpected={len(unexpected)}")
        if len(mapped) == 0:
            sample = list(cleaned.keys())[:25]
            print("[LOAD] sample_ckpt_keys:", sample)

    if len(mapped) == 0:
        raise RuntimeError("0 keys matched while loading baseline into CHADO.")


# -------------------- dataset builder (exact signature) --------------------
def build_meld_dataset(cfg: Dict[str, Any], csv_path: str, label_map: Dict[str, int]) -> MeldDataset:
    """
    Matches your printed signature exactly:
    (csv_path, text_model_name, label_map, text_col, label_col, audio_path_col, video_path_col,
     utt_id_col, num_frames, frame_size, sample_rate, max_audio_seconds, use_text, use_audio, use_video, seed)
    """
    d = cfg["data"]
    m = cfg["model"]

    # IMPORTANT: your CSV header shows label column is 'emotion', not 'label'
    label_col = d.get("label_col", "emotion")
    if label_col == "label":
        # auto-fix common mistake
        label_col = "emotion"

    ds = MeldDataset(
        csv_path,
        m["text_model_name"],
        label_map,
        d["text_col"],
        label_col,
        d["audio_path_col"],
        d["video_path_col"],
        d["utt_id_col"],
        int(d["num_frames"]),
        int(d["frame_size"]),
        int(d["sample_rate"]),
        float(d["max_audio_seconds"]),
        bool(m["use_text"]),
        bool(m["use_audio"]),
        bool(m["use_video"]),
        int(cfg["train"].get("seed", 42)),
    )
    if not hasattr(ds, "tokenizer"):
        raise RuntimeError("MeldDataset must expose ds.tokenizer for collate_meld().")
    return ds


# -------------------- metrics --------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> Dict[str, float]:
    model.eval()

    # infer #classes
    num_classes = loader.dataset.num_classes if hasattr(loader.dataset, "num_classes") else 7
    tp = torch.zeros(num_classes, device=device)
    fp = torch.zeros(num_classes, device=device)
    fn = torch.zeros(num_classes, device=device)
    total = torch.tensor(0.0, device=device)
    correct = torch.tensor(0.0, device=device)

    for batch in loader:
        labels = batch.labels.to(device)

        text = {k: v.to(device) for k, v in batch.text_input.items()} if batch.text_input else None
        audio = batch.audio_wave.to(device) if batch.audio_wave is not None else None
        video = batch.video_frames.to(device) if batch.video_frames is not None else None

        with autocast(enabled=amp):
            out = model(text_input=text, audio_wave=audio, video_frames=video, modality_mask=None, compute_losses=False)
            logits = out.logits

        preds = logits.argmax(dim=-1)
        total += labels.numel()
        correct += (preds == labels).sum()

        for c in range(num_classes):
            tp[c] += ((preds == c) & (labels == c)).sum()
            fp[c] += ((preds == c) & (labels != c)).sum()
            fn[c] += ((preds != c) & (labels == c)).sum()

    if dist.is_initialized():
        dist.all_reduce(tp); dist.all_reduce(fp); dist.all_reduce(fn)
        dist.all_reduce(total); dist.all_reduce(correct)

    acc = (correct / total.clamp(min=1.0)).item()

    precision_c = tp / (tp + fp).clamp(min=1.0)
    recall_c = tp / (tp + fn).clamp(min=1.0)
    f1_c = (2 * precision_c * recall_c) / (precision_c + recall_c).clamp(min=1e-8)

    support = tp + fn
    denom = support.sum().clamp(min=1.0)

    precision_w = (precision_c * support).sum() / denom
    recall_w = (recall_c * support).sum() / denom
    f1_w = (f1_c * support).sum() / denom

    return {
        "acc": float(acc),
        "precision_w": float(precision_w.item()),
        "recall_w": float(recall_w.item()),
        "f1_w": float(f1_w.item()),
    }


# -------------------- training --------------------
def main():
    ddp = ddp_init()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))

    # reproducibility
    seed_all(int(cfg["train"].get("seed", 42)))

    # speed/stability
    torch.backends.cudnn.benchmark = True

    # offline huggingface option
    offline = bool(int(os.environ.get("TRANSFORMERS_OFFLINE", "0"))) or bool(int(os.environ.get("HF_HUB_OFFLINE", "0")))
    local_files_only = offline

    label_map = build_label_map_from_order(EMO_ORDER_7)

    # ---- data
    train_ds = build_meld_dataset(cfg, cfg["data"]["train_csv"], label_map)
    val_ds   = build_meld_dataset(cfg, cfg["data"]["val_csv"], label_map)
    test_ds  = build_meld_dataset(cfg, cfg["data"]["test_csv"], label_map)

    def make_loader(ds, shuffle: bool, drop_last: bool):
        sampler = DistributedSampler(ds, shuffle=shuffle) if dist.is_initialized() else None
        return DataLoader(
            ds,
            batch_size=int(cfg["train"]["batch_size_per_gpu"]),
            sampler=sampler,
            shuffle=(sampler is None and shuffle),
            num_workers=int(cfg["train"].get("num_workers", 6)),
            pin_memory=True,
            drop_last=drop_last,
            collate_fn=lambda b: collate_meld(
                b, ds.tokenizer,
                bool(cfg["model"]["use_text"]),
                bool(cfg["model"]["use_audio"]),
                bool(cfg["model"]["use_video"]),
            ),
        ), sampler

    train_loader, train_sampler = make_loader(train_ds, shuffle=True, drop_last=True)
    val_loader,   _            = make_loader(val_ds, shuffle=False, drop_last=False)
    test_loader,  _            = make_loader(test_ds, shuffle=False, drop_last=False)

    # ---- model
    model = CHADOTrimodal(
        text_model_name=cfg["model"]["text_model_name"],
        audio_model_name=cfg["model"]["audio_model_name"],
        video_model_name=cfg["model"]["video_model_name"],
        num_classes=int(cfg["data"]["num_classes"]),
        proj_dim=int(cfg["model"]["proj_dim"]),
        dropout=float(cfg["model"]["dropout"]),
        use_text=bool(cfg["model"]["use_text"]),
        use_audio=bool(cfg["model"]["use_audio"]),
        use_video=bool(cfg["model"]["use_video"]),
        use_gated_fusion=bool(cfg["model"].get("use_gated_fusion", True)),

        use_causal=bool(cfg["chado"]["use_causal"]),
        use_hyperbolic=bool(cfg["chado"]["use_hyperbolic"]),
        use_transport=bool(cfg["chado"]["use_transport"]),
        use_refinement=bool(cfg["chado"]["use_refinement"]),

        w_mad=float(cfg["chado"].get("w_mad", 1.0)),
        w_ot=float(cfg["chado"].get("w_transport", 1.0)),
        w_hyp=float(cfg["chado"].get("w_hyperbolic", 1.0)),
        w_causal=float(cfg["chado"].get("w_causal", 1.0)),
        w_refine=float(cfg["chado"].get("w_refine", 1.0)),

        hyp_c=float(cfg["chado"].get("curvature", 1.0)),
        ot_eps=float(cfg["chado"].get("ot_eps", 0.05)),
        ot_iters=int(cfg["chado"].get("ot_iters", 30)),
        mad_mode=str(cfg["chado"].get("mad_mode", "entropy")),

        causal_drop_prob=float(cfg["chado"].get("causal_drop_prob", 0.33)),
        causal_mode=str(cfg["chado"].get("causal_mode", "kl")),
        causal_temp=float(cfg["chado"].get("causal_temp", 1.0)),

        local_files_only=local_files_only,
    ).to(device)

    # load baseline weights into CHADO wrapper (strong init)
    baseline_ckpt = cfg.get("chado", {}).get("eval_from_baseline_ckpt", "") or cfg.get("chado", {}).get("baseline_ckpt", "")
    if baseline_ckpt:
        load_baseline_into_chado(model, baseline_ckpt)

    # DDP wrapper (STATIC GRAPH prevents “marked ready twice” for fixed topology)
    if dist.is_initialized():
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            static_graph=True,
        )

    # ---- optimization
    lr = float(cfg["train"].get("lr", 2e-5))
    wd = float(cfg["train"].get("weight_decay", 0.01))
    epochs = int(cfg["train"].get("epochs", 10))
    amp = bool(cfg["train"].get("amp", True))

    # train all params (you can freeze encoders later if needed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    steps_per_epoch = len(train_loader)
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_ratio = float(cfg["train"].get("warmup_ratio", 0.06))
    warmup_steps = int(total_steps * warmup_ratio)

    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scaler = GradScaler(enabled=amp)

    # ---- output
    out_dir = cfg.get("logging", {}).get("out_dir", "runs")
    run_name = cfg.get("logging", {}).get("run_name", "chado_run")
    run_dir = os.path.join(out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    if is_main():
        with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    # ---- training loop
    best_val_f1 = -1.0
    best_path = os.path.join(run_dir, "best.pt")
    last_path = os.path.join(run_dir, "last.pt")

    ce = nn.CrossEntropyLoss()

    global_step = 0
    t0 = time.time()

    for ep in range(epochs):
        if dist.is_initialized() and train_sampler is not None:
            train_sampler.set_epoch(ep)

        model.train()
        running_loss = 0.0
        seen = 0

        for batch in train_loader:
            labels = batch.labels.to(device)

            text = {k: v.to(device) for k, v in batch.text_input.items()} if batch.text_input else None
            audio = batch.audio_wave.to(device) if batch.audio_wave is not None else None
            video = batch.video_frames.to(device) if batch.video_frames is not None else None

            opt.zero_grad(set_to_none=True)

            with autocast(enabled=amp):
                out = model(text_input=text, audio_wave=audio, video_frames=video, modality_mask=None, compute_losses=True)
                logits = out.logits
                loss_cls = ce(logits, labels)
                loss = loss_cls + out.total_chado_loss

            # SINGLE backward (avoids reentrant backward)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

            running_loss += float(loss.detach().item()) * labels.size(0)
            seen += labels.size(0)
            global_step += 1

        # reduce epoch loss
        if dist.is_initialized():
            tl = torch.tensor([running_loss, seen], device=device)
            dist.all_reduce(tl)
            running_loss, seen = float(tl[0].item()), float(tl[1].item())

        train_loss = running_loss / max(1.0, seen)

        # eval
        val_metrics = evaluate(model.module if isinstance(model, DDP) else model, val_loader, device, amp)
        test_metrics = evaluate(model.module if isinstance(model, DDP) else model, test_loader, device, amp)

        if is_main():
            print(
                f"[epoch {ep}] loss={train_loss:.4f} "
                f"val_acc={val_metrics['acc']:.4f} val_f1={val_metrics['f1_w']:.4f} "
                f"test_acc={test_metrics['acc']:.4f} test_f1={test_metrics['f1_w']:.4f}"
            )

        # save last
        if is_main():
            torch.save(
                {"epoch": ep, "state_dict": (model.module if isinstance(model, DDP) else model).state_dict(), "cfg": cfg},
                last_path,
            )

        # save best by val_f1
        if val_metrics["f1_w"] > best_val_f1:
            best_val_f1 = val_metrics["f1_w"]
            if is_main():
                torch.save(
                    {"epoch": ep, "state_dict": (model.module if isinstance(model, DDP) else model).state_dict(), "cfg": cfg,
                     "val": val_metrics, "test": test_metrics},
                    best_path,
                )
                with open(os.path.join(run_dir, "metrics_best.json"), "w", encoding="utf-8") as f:
                    json.dump({"val": val_metrics, "test": test_metrics}, f, indent=2)

    if is_main():
        dt = time.time() - t0
        print(f"[DONE] best_val_f1={best_val_f1:.4f} time={dt/60:.1f} min")
        print(f"[CKPT] best={best_path}")
        print(f"[CKPT] last={last_path}")


if __name__ == "__main__":
    import argparse
    try:
        main()
    finally:
        ddp_cleanup()

# import os
# import yaml
# import time
# import random
# import inspect
# import numpy as np
# import torch
# import torch.distributed as dist
# import torch.nn.functional as F

# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data import DataLoader, DistributedSampler
# from torch.amp import autocast, GradScaler
# from transformers import get_cosine_schedule_with_warmup

# from src.data.meld_dataset import (
#     MeldDataset,
#     collate_meld,
#     build_label_map_from_order,
#     EMO_ORDER_7,
# )
# from src.models.chado_trimodal import CHADOTrimodal

# # ============================================================
# # DDP UTILITIES
# # ============================================================

# def setup_ddp():
#     if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
#         dist.init_process_group("nccl")
#         torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

# def cleanup_ddp():
#     if dist.is_available() and dist.is_initialized():
#         dist.destroy_process_group()

# def is_main():
#     return (not dist.is_initialized()) or dist.get_rank() == 0

# def seed_all(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


# # ============================================================
# # DATASET BUILDER (SIGNATURE SAFE)
# # ============================================================

# def build_meld_dataset(cfg, csv_path, label_map):
#     sig = inspect.signature(MeldDataset.__init__)
#     accepted = set(sig.parameters.keys())

#     text_model_name = cfg.get("model", {}).get("text_model_name", None)

#     candidates = {
#         "csv_path": csv_path,
#         "text_col": cfg["data"]["text_col"],
#         "label_col": cfg["data"]["label_col"],
#         "audio_path_col": cfg["data"]["audio_path_col"],
#         "video_path_col": cfg["data"]["video_path_col"],
#         "utt_id_col": cfg["data"]["utt_id_col"],
#         "num_frames": cfg["data"]["num_frames"],
#         "frame_size": cfg["data"]["frame_size"],
#         "sample_rate": cfg["data"]["sample_rate"],
#         "max_audio_seconds": cfg["data"]["max_audio_seconds"],
#         "label_map": label_map,
#         "use_text": cfg["model"]["use_text"],
#         "use_audio": cfg["model"]["use_audio"],
#         "use_video": cfg["model"]["use_video"],
#         "text_model_name": text_model_name,
#         "tokenizer_name": text_model_name,
#         "hf_model_name": text_model_name,
#         "bert_name": text_model_name,
#     }

#     kwargs = {k: v for k, v in candidates.items() if (k in accepted and v is not None)}

#     required = [
#         p.name for p in sig.parameters.values()
#         if p.name != "self" and p.default is inspect._empty
#     ]
#     missing = [k for k in required if k not in kwargs]
#     if missing:
#         if is_main():
#             print("[FATAL] MeldDataset signature:", sig)
#             print("[FATAL] Provided keys:", sorted(kwargs.keys()))
#             print("[FATAL] Missing required keys:", missing)
#         raise TypeError(f"MeldDataset missing required args: {missing}")

#     ds = MeldDataset(**kwargs)
#     if not hasattr(ds, "tokenizer"):
#         raise RuntimeError("MeldDataset must expose ds.tokenizer for collate_meld().")
#     return ds


# # ============================================================
# # METRICS (Acc + weighted P/R/F1)
# # ============================================================

# @torch.no_grad()
# def evaluate_full(model, loader, device, amp):
#     model.eval()
#     total, correct = 0, 0

#     num_classes = loader.dataset.num_classes if hasattr(loader.dataset, "num_classes") else 7
#     tp = torch.zeros(num_classes, device=device)
#     fp = torch.zeros(num_classes, device=device)
#     fn = torch.zeros(num_classes, device=device)

#     for batch in loader:
#         labels = batch.labels.to(device)
#         text = {k: v.to(device) for k, v in batch.text_input.items()} if batch.text_input else None
#         audio = batch.audio_wave.to(device) if batch.audio_wave is not None else None
#         video = batch.video_frames.to(device) if batch.video_frames is not None else None

#         with autocast("cuda", enabled=amp):
#             logits, _, _ = model(
#                 text_input=text,
#                 audio_wave=audio,
#                 video_frames=video,
#                 modality_mask=None,
#             )

#         preds = logits.argmax(dim=-1)
#         total += labels.numel()
#         correct += (preds == labels).sum().item()

#         for c in range(num_classes):
#             tp[c] += ((preds == c) & (labels == c)).sum()
#             fp[c] += ((preds == c) & (labels != c)).sum()
#             fn[c] += ((preds != c) & (labels == c)).sum()

#     if dist.is_initialized():
#         dist.all_reduce(tp)
#         dist.all_reduce(fp)
#         dist.all_reduce(fn)
#         t = torch.tensor([total, correct], device=device, dtype=torch.float32)
#         dist.all_reduce(t)
#         total, correct = int(t[0].item()), int(t[1].item())

#     acc = correct / max(1, total)
#     precision_c = tp / torch.clamp(tp + fp, min=1.0)
#     recall_c = tp / torch.clamp(tp + fn, min=1.0)
#     f1_c = 2 * precision_c * recall_c / torch.clamp(precision_c + recall_c, min=1e-8)
#     support = tp + fn

#     # Weighted (by support)
#     w = support / torch.clamp(support.sum(), min=1.0)
#     prec_w = (precision_c * w).sum()
#     rec_w = (recall_c * w).sum()
#     f1_w = (f1_c * w).sum()

#     return float(acc), float(prec_w.item()), float(rec_w.item()), float(f1_w.item())


# # ============================================================
# # BASELINE → CHADO SAFE LOADER
# # ============================================================

# from collections import OrderedDict

# def _is_state_dict(obj):
#     if not isinstance(obj, (dict, OrderedDict)):
#         return False
#     if len(obj) == 0:
#         return False
#     n_tensors = sum(torch.is_tensor(v) for v in obj.values())
#     return n_tensors >= max(1, int(0.8 * len(obj)))

# def _extract_state_dict(ckpt):
#     if _is_state_dict(ckpt):
#         return ckpt
#     if isinstance(ckpt, dict):
#         for k in ["state_dict", "model_state_dict", "model", "net", "module", "weights"]:
#             if k in ckpt and _is_state_dict(ckpt[k]):
#                 return ckpt[k]
#         for v in ckpt.values():
#             if _is_state_dict(v):
#                 return v
#             if isinstance(v, dict):
#                 for vv in v.values():
#                     if _is_state_dict(vv):
#                         return vv
#     raise ValueError("Could not locate a valid state_dict inside the checkpoint file.")

# def _strip_prefix(k: str, prefixes):
#     for p in prefixes:
#         if k.startswith(p):
#             return k[len(p):]
#     return k

# def load_baseline_into_chado(model, ckpt_path):
#     ckpt = torch.load(ckpt_path, map_location="cpu")
#     sd = _extract_state_dict(ckpt)

#     cleaned = {}
#     for k, v in sd.items():
#         k = _strip_prefix(k, ["module.", "model.", "net."])
#         cleaned[k] = v

#     model_sd = model.state_dict()
#     model_keys = set(model_sd.keys())

#     mapped = {}
#     for k, v in cleaned.items():
#         if k in model_keys:
#             mapped[k] = v
#             continue
#         k_base = "base." + k
#         if k_base in model_keys:
#             mapped[k_base] = v
#             continue

#         aliases = [
#             ("text_model.", "text_encoder."),
#             ("audio_model.", "audio_encoder."),
#             ("video_model.", "video_encoder."),
#         ]
#         kk = k
#         for a, b in aliases:
#             if kk.startswith(a):
#                 kk = b + kk[len(a):]
#         if ("base." + kk) in model_keys:
#             mapped["base." + kk] = v
#             continue
#         if kk in model_keys:
#             mapped[kk] = v
#             continue

#     missing, unexpected = model.load_state_dict(mapped, strict=False)

#     if is_main():
#         print(f"[LOAD] baseline ckpt: {ckpt_path}")
#         print(f"[LOAD] extracted state_dict keys: {len(sd)}")
#         sk = list(cleaned.keys())[:15]
#         print("[LOAD] sample ckpt keys:", sk)
#         print(f"[LOAD] matched keys: {len(mapped)}")
#         print(f"[LOAD] missing keys: {len(missing)} (expected CHADO heads)")
#         print(f"[LOAD] unexpected keys: {len(unexpected)}")

#     if len(mapped) == 0:
#         raise RuntimeError("[FATAL] 0 keys matched when loading baseline into CHADO.")


# # ============================================================
# # TRAIN LOOP
# # ============================================================

# def main():
#     setup_ddp()
#     local_rank = int(os.environ.get("LOCAL_RANK", 0))
#     device = torch.device("cuda", local_rank)

#     import argparse
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--config", required=True)
#     args = ap.parse_args()

#     cfg = yaml.safe_load(open(args.config))
#     seed_all(cfg["train"]["seed"] + (dist.get_rank() if dist.is_initialized() else 0))

#     label_map = build_label_map_from_order(EMO_ORDER_7)

#     train_ds = build_meld_dataset(cfg, cfg["data"]["train_csv"], label_map)
#     val_ds   = build_meld_dataset(cfg, cfg["data"]["val_csv"], label_map)
#     test_ds  = build_meld_dataset(cfg, cfg["data"]["test_csv"], label_map)

#     train_sampler = DistributedSampler(train_ds, shuffle=True) if dist.is_initialized() else None
#     val_sampler   = DistributedSampler(val_ds, shuffle=False) if dist.is_initialized() else None
#     test_sampler  = DistributedSampler(test_ds, shuffle=False) if dist.is_initialized() else None

#     def _collate(ds):
#         return lambda b: collate_meld(
#             b, ds.tokenizer,
#             cfg["model"]["use_text"],
#             cfg["model"]["use_audio"],
#             cfg["model"]["use_video"],
#         )

#     train_loader = DataLoader(
#         train_ds,
#         batch_size=cfg["train"]["batch_size_per_gpu"],
#         sampler=train_sampler,
#         shuffle=train_sampler is None,
#         num_workers=cfg["train"]["num_workers"],
#         pin_memory=True,
#         drop_last=True,
#         collate_fn=_collate(train_ds),
#     )
#     val_loader = DataLoader(
#         val_ds,
#         batch_size=cfg["train"]["batch_size_per_gpu"],
#         sampler=val_sampler,
#         shuffle=False,
#         num_workers=cfg["train"]["num_workers"],
#         pin_memory=True,
#         drop_last=False,
#         collate_fn=_collate(val_ds),
#     )
#     test_loader = DataLoader(
#         test_ds,
#         batch_size=cfg["train"]["batch_size_per_gpu"],
#         sampler=test_sampler,
#         shuffle=False,
#         num_workers=cfg["train"]["num_workers"],
#         pin_memory=True,
#         drop_last=False,
#         collate_fn=_collate(test_ds),
#     )

#     model = CHADOTrimodal(
#         text_model_name=cfg["model"]["text_model_name"],
#         audio_model_name=cfg["model"]["audio_model_name"],
#         video_model_name=cfg["model"]["video_model_name"],
#         num_classes=cfg["data"]["num_classes"],
#         proj_dim=cfg["model"]["proj_dim"],
#         dropout=cfg["model"]["dropout"],
#         use_text=cfg["model"]["use_text"],
#         use_audio=cfg["model"]["use_audio"],
#         use_video=cfg["model"]["use_video"],
#         use_gated_fusion=cfg["model"]["use_gated_fusion"],
#         use_causal=cfg["chado"]["use_causal"],
#         use_hyperbolic=cfg["chado"]["use_hyperbolic"],
#         use_transport=cfg["chado"]["use_transport"],
#         use_refinement=cfg["chado"]["use_refinement"],
#     ).to(device)
#     try:
#         _disable_gc(model)
#     except Exception as e:
#         if is_main():
#             print(f"[WARN] _disable_gc failed: {e}")
#     # Load baseline weights for initialization
#     chado_cfg = cfg.get("chado", {})
#     baseline_ckpt = (
#         chado_cfg.get("eval_from_baseline_ckpt")
#         or chado_cfg.get("init_from_baseline_ckpt")
#         or chado_cfg.get("baseline_ckpt")
#         or ""
#     )
#     if baseline_ckpt:
#         load_baseline_into_chado(model, baseline_ckpt)

#     if dist.is_initialized():
#         ddp_kwargs = dict(device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

#         # PyTorch >= 2.0 supports static_graph arg in many builds; if not, fallback to _set_static_graph()
#         try:
#             model = DDP(model, **ddp_kwargs, static_graph=True)
#         except TypeError:
#             model = DDP(model, **ddp_kwargs)
#             try:
#                 model._set_static_graph()
#             except Exception:
#                 pass


#     # Optimizer / scheduler
#     lr = float(cfg["train"]["lr"])
#     wd = float(cfg["train"].get("weight_decay", 0.0))
#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

#     epochs = int(cfg["train"]["epochs"])
#     steps_per_epoch = max(1, len(train_loader))
#     total_steps = steps_per_epoch * epochs

#     warmup_ratio = float(cfg["train"].get("warmup_ratio", 0.0))
#     warmup_steps = int(total_steps * warmup_ratio)

#     scheduler = get_cosine_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=warmup_steps,
#         num_training_steps=total_steps
#     )

#     amp = bool(cfg["train"].get("amp", True))
#     scaler = GradScaler(enabled=amp)

#     # Output dir
#     out_dir = cfg.get("logging", {}).get("out_dir", "runs")
#     run_name = cfg.get("logging", {}).get("run_name", "chado_run")
#     run_dir = os.path.join(out_dir, run_name)
#     if is_main():
#         os.makedirs(run_dir, exist_ok=True)
#         with open(os.path.join(run_dir, "config.yaml"), "w") as f:
#             yaml.safe_dump(cfg, f, sort_keys=False)

#     best_val_f1 = -1.0
#     best_path = os.path.join(run_dir, "best.pt")
#     last_path = os.path.join(run_dir, "last.pt")

#     # TRAIN
#     for ep in range(epochs):
#         if train_sampler is not None:
#             train_sampler.set_epoch(ep)

#         model.train()
#         t0 = time.time()
#         running_loss = 0.0

#         for step, batch in enumerate(train_loader):
#             labels = batch.labels.to(device)
#             text = {k: v.to(device) for k, v in batch.text_input.items()} if batch.text_input else None
#             audio = batch.audio_wave.to(device) if batch.audio_wave is not None else None
#             video = batch.video_frames.to(device) if batch.video_frames is not None else None

#             optimizer.zero_grad(set_to_none=True)

#             with autocast("cuda", enabled=amp):
#                 logits, _, _ = model(
#                     text_input=text,
#                     audio_wave=audio,
#                     video_frames=video,
#                     modality_mask=None,
#                 )
#                 loss = F.cross_entropy(logits, labels)

#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#             scheduler.step()

#             running_loss += float(loss.item())

#         # EVAL (val)
#         val_acc, val_prec, val_rec, val_f1 = evaluate_full(model, val_loader, device, amp)
#         if is_main():
#             dt = time.time() - t0
#             print(f"[epoch {ep}] loss={running_loss/max(1,len(train_loader)):.4f} "
#                   f"val_acc={val_acc:.4f} val_prec={val_prec:.4f} val_rec={val_rec:.4f} val_wf1={val_f1:.4f} "
#                   f"time={dt:.1f}s")

#         # Save last
#         if is_main():
#             sd = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
#             torch.save({"state_dict": sd, "epoch": ep}, last_path)

#         # Save best by val WF1
#         if val_f1 > best_val_f1:
#             best_val_f1 = val_f1
#             if is_main():
#                 sd = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
#                 torch.save({"state_dict": sd, "epoch": ep, "best_val_wf1": best_val_f1}, best_path)

#     # FINAL TEST using best
#     if is_main():
#         print("===== FINAL (BEST) EVAL =====")
#         print(f"[BEST] val_wf1={best_val_f1:.4f} ckpt={best_path}")

#     # Load best weights back (all ranks)
#     best_obj = torch.load(best_path, map_location="cpu")
#     best_sd = best_obj["state_dict"] if isinstance(best_obj, dict) and "state_dict" in best_obj else best_obj
#     if isinstance(model, DDP):
#         model.module.load_state_dict(best_sd, strict=False)
#     else:
#         model.load_state_dict(best_sd, strict=False)

#     test_acc, test_prec, test_rec, test_f1 = evaluate_full(model, test_loader, device, amp)
#     if is_main():
#         print(f"[TEST] acc={test_acc:.4f} prec={test_prec:.4f} rec={test_rec:.4f} wf1={test_f1:.4f}")
#         with open(os.path.join(run_dir, "metrics.txt"), "w") as f:
#             f.write(f"val_best_wf1={best_val_f1:.6f}\n")
#             f.write(f"test_acc={test_acc:.6f}\n")
#             f.write(f"test_prec={test_prec:.6f}\n")
#             f.write(f"test_rec={test_rec:.6f}\n")
#             f.write(f"test_wf1={test_f1:.6f}\n")
# def _disable_gc(m):
#     # Transformers-style gradient checkpointing
#     if hasattr(m, "gradient_checkpointing_disable"):
#         try:
#             m.gradient_checkpointing_disable()
#         except Exception:
#             pass
#     # Some repos store flag
#     if hasattr(m, "gradient_checkpointing"):
#         try:
#             m.gradient_checkpointing = False
#         except Exception:
#             pass

# # Disable checkpointing everywhere possible
# #_disable_gc(model)
# for attr in ["text_encoder", "audio_encoder", "video_encoder"]:
#     if hasattr(model, attr):
#         _disable_gc(getattr(model, attr))
# # If CHADO stores baseline under model.base.*
# if hasattr(model, "base"):
#     for attr in ["text_encoder", "audio_encoder", "video_encoder"]:
#         if hasattr(model.base, attr):
#             _disable_gc(getattr(model.base, attr))


# if __name__ == "__main__":
#     try:
#         main()
#     finally:
#         cleanup_ddp()

