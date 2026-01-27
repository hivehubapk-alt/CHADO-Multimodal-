#!/usr/bin/env python3
import os
import argparse
import warnings
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoTokenizer, AutoImageProcessor
from transformers.utils import logging as hf_logging

from chado_lib.utils.ddp import ddp_setup, ddp_cleanup, ddp_barrier, ddp_concat_all_gather
from chado_lib.utils.seed import set_all_seeds
from chado_lib.utils.io import load_yaml, expand_env, mkdir
from chado_lib.data.iemocap_dataset import IEMOCAPTriModal
from chado_lib.data.collate import collate_fn
from chado_lib.models.chado import CHADO
from chado_lib.models.components import (
    disentanglement_loss, causal_dropout, hyperbolic_reg_loss,
    sinkhorn_ot_cost, MADState, mad_loss
)
from chado_lib.metrics.classification import accuracy, macro_f1

import matplotlib.pyplot as plt

LABELS = ["neu", "hap", "ang", "sad"]

def ablation_to_modalities(ablation: str):
    if ablation == "tri": return True, True, True
    if ablation == "text": return True, False, False
    if ablation == "audio": return False, True, False
    if ablation == "vision": return False, False, True
    if ablation == "ta": return True, True, False
    if ablation == "tv": return True, False, True
    if ablation == "av": return False, True, True
    raise ValueError(f"Unknown ablation={ablation}")

@torch.no_grad()
def evaluate_ddp(model, loader, device, num_classes=4, return_logits=False):
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")

    total_loss_local = torch.zeros((), device=device, dtype=torch.float32)
    n_local = torch.zeros((), device=device, dtype=torch.long)

    preds_local, gold_local = [], []
    probs_local = [] if return_logits else None

    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(device, non_blocking=True)

        logits, _ = model(batch["input_ids"], batch["attention_mask"], batch["wav"], batch["pixel_values"])
        loss = ce(logits, batch["label"])

        pred = torch.argmax(logits, dim=-1)
        total_loss_local += loss.detach().float()
        n_local += batch["label"].numel()

        preds_local.append(pred.detach())
        gold_local.append(batch["label"].detach())

        if return_logits:
            probs_local.append(torch.softmax(logits.detach().float(), dim=-1))

    preds_local = torch.cat(preds_local, dim=0) if preds_local else torch.empty((0,), device=device, dtype=torch.long)
    gold_local  = torch.cat(gold_local,  dim=0) if gold_local  else torch.empty((0,), device=device, dtype=torch.long)

    preds_all = ddp_concat_all_gather(preds_local).detach().cpu()
    gold_all  = ddp_concat_all_gather(gold_local).detach().cpu()

    # all-reduce loss/count via gather semantics
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        loss_all = total_loss_local.clone()
        n_all = n_local.clone()
        torch.distributed.all_reduce(loss_all, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(n_all, op=torch.distributed.ReduceOp.SUM)
    else:
        loss_all, n_all = total_loss_local, n_local

    loss_avg = (loss_all / torch.clamp(n_all.float(), min=1.0)).item()
    acc = accuracy(preds_all, gold_all)
    f1  = macro_f1(preds_all, gold_all, num_classes)

    if return_logits:
        probs_local = torch.cat(probs_local, dim=0) if probs_local else torch.empty((0, num_classes), device=device)
        probs_all = ddp_concat_all_gather(probs_local).detach().cpu().numpy()
        return loss_avg, acc, f1, preds_all.numpy(), gold_all.numpy(), probs_all

    return loss_avg, acc, f1

def save_train_val_curve(out_dir, hist):
    if not hist["epoch"]:
        return
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Train vs Val accuracy (your requested style)
    fig = plt.figure(figsize=(7,4))
    ax = fig.add_subplot(111)
    ax.plot(hist["epoch"], hist["train_acc"], label="Train Accuracy")
    ax.plot(hist["epoch"], hist["val_acc"], label="Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training vs Validation Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "train_vs_val_accuracy.png"), dpi=220)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--config", required=True, help="YAML config (iemocap_tri.yaml style).")
    ap.add_argument("--save_reports", action="store_true")

    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    warnings.filterwarnings("ignore")
    hf_logging.set_verbosity_error()

    cfg = load_yaml(args.config)
    cfg = expand_env(cfg)

    ddp, rank, local_rank, world = ddp_setup()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    try:
        if rank == 0:
            mkdir(args.out_dir)

        set_all_seeds(int(cfg["seed"]) + rank)

        # Tokenizer / image processor
        tok = AutoTokenizer.from_pretrained(cfg["models"]["text_model"], use_fast=False)
        img_proc = AutoImageProcessor.from_pretrained(cfg["models"]["vision_model"], use_fast=False)

        train_ds = IEMOCAPTriModal(args.train_csv, tok, img_proc,
                                  max_text_len=cfg["data"]["max_text_len"],
                                  audio_sec=cfg["data"]["audio_sec"],
                                  n_frames=cfg["data"]["n_frames"])
        val_ds = IEMOCAPTriModal(args.val_csv, tok, img_proc,
                                max_text_len=cfg["data"]["max_text_len"],
                                audio_sec=cfg["data"]["audio_sec"],
                                n_frames=cfg["data"]["n_frames"])
        test_ds = IEMOCAPTriModal(args.test_csv, tok, img_proc,
                                 max_text_len=cfg["data"]["max_text_len"],
                                 audio_sec=cfg["data"]["audio_sec"],
                                 n_frames=cfg["data"]["n_frames"])

        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True) if ddp else None
        val_sampler   = DistributedSampler(val_ds, shuffle=False, drop_last=False) if ddp else None
        test_sampler  = DistributedSampler(test_ds, shuffle=False, drop_last=False) if ddp else None

        bs = int(cfg["batch_size"])
        train_loader = DataLoader(train_ds, batch_size=bs, sampler=train_sampler,
                                  shuffle=(train_sampler is None), num_workers=4,
                                  pin_memory=True, collate_fn=collate_fn, drop_last=True)
        val_loader   = DataLoader(val_ds, batch_size=bs, sampler=val_sampler,
                                  shuffle=False, num_workers=4, pin_memory=True,
                                  collate_fn=collate_fn, drop_last=False)
        test_loader  = DataLoader(test_ds, batch_size=bs, sampler=test_sampler,
                                  shuffle=False, num_workers=4, pin_memory=True,
                                  collate_fn=collate_fn, drop_last=False)

        ablation = cfg["chado"]["ablation"]
        use_text, use_audio, use_video = ablation_to_modalities(ablation)

        model = CHADO(
            cfg["models"]["text_model"],
            cfg["models"]["audio_model"],
            cfg["models"]["vision_model"],
            num_classes=4,
            use_text=use_text, use_audio=use_audio, use_video=use_video
        ).to(device)

        if ddp:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
            model._set_static_graph()

        # Optim
        enc_params, head_params = [], []
        for n, p in model.named_parameters():
            if ("head" in n or "fuser" in n or "disent" in n):
                head_params.append(p)
            else:
                enc_params.append(p)

        optim = torch.optim.AdamW(
            [
                {"params": enc_params, "lr": float(cfg["weights"]["lr_enc"])},
                {"params": head_params, "lr": float(cfg["weights"]["lr_head"])},
            ],
            weight_decay=float(cfg["weights"]["weight_decay"])
        )

        ce = nn.CrossEntropyLoss()
        amp_enabled = bool(cfg.get("amp", True))
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        best_val_acc = -1.0
        mad_state = MADState()

        hist = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}

        epochs = int(cfg["epochs"])
        aux_warmup = int(cfg["weights"]["aux_warmup_epochs"])

        for epoch in range(1, epochs + 1):
            if ddp:
                train_sampler.set_epoch(epoch)
            model.train()

            total_loss, n_seen = 0.0, 0
            correct_local = 0
            total_local = 0

            ramp = 0.0 if epoch <= aux_warmup else 1.0

            for step, batch in enumerate(train_loader, start=1):
                for k in batch:
                    batch[k] = batch[k].to(device, non_blocking=True)

                optim.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    logits, h = model(batch["input_ids"], batch["attention_mask"], batch["wav"], batch["pixel_values"])
                    loss_ce = ce(logits, batch["label"])

                # train-acc local
                pred = torch.argmax(logits.detach(), dim=-1)
                correct_local += (pred == batch["label"]).sum().item()
                total_local += batch["label"].numel()

                aux = torch.zeros((), device=device, dtype=torch.float32)

                # AUX terms in fp32
                with torch.cuda.amp.autocast(enabled=False):
                    h32 = h.float()
                    disent = model.module.disent(h32) if hasattr(model, "module") else model.disent(h32)
                    aux = aux + 0.05 * disentanglement_loss(disent)

                    if cfg["chado"]["use_causal"]:
                        h_c = causal_dropout(h32, float(cfg["hyper"]["causal_drop_p"]))
                        head = model.module.head if hasattr(model, "module") else model.head
                        logits_c = head(h_c).float()
                        aux = aux + (float(cfg["weights"]["w_causal"]) * ramp) * ce(logits_c, batch["label"]).float()

                    if cfg["chado"]["use_hyperbolic"]:
                        aux = aux + (float(cfg["weights"]["w_hyp"]) * ramp) * hyperbolic_reg_loss(disent, c=float(cfg["hyper"]["hyp_c"]))

                    if cfg["chado"]["use_ot"]:
                        idx = torch.randperm(disent.size(0), device=disent.device)
                        aux = aux + (float(cfg["weights"]["w_ot"]) * ramp) * sinkhorn_ot_cost(
                            disent, disent[idx],
                            eps=float(cfg["hyper"]["ot_eps"]),
                            iters=int(cfg["hyper"]["ot_iters"])
                        )

                    if cfg["chado"]["use_refine"] and int(cfg["chado"]["refine_steps"]) > 0:
                        head = model.module.head if hasattr(model, "module") else model.head
                        for _ in range(int(cfg["chado"]["refine_steps"])):
                            h32 = h32 + 0.10 * torch.tanh(h32)
                        logits_r = head(h32).float()
                        aux = aux + (0.05 * ramp) * ce(logits_r, batch["label"]).float()

                    if cfg["chado"]["use_mad"]:
                        with torch.no_grad():
                            cur_mean = logits.detach().float().mean(dim=0)
                            if mad_state.ema_teacher_logits is None:
                                mad_state.ema_teacher_logits = cur_mean
                            else:
                                mad_state.ema_teacher_logits = (
                                    float(cfg["weights"]["mad_ema"]) * mad_state.ema_teacher_logits
                                    + (1.0 - float(cfg["weights"]["mad_ema"])) * cur_mean
                                )
                        aux = aux + (float(cfg["weights"]["mad_w"]) * ramp) * mad_loss(
                            logits.float(),
                            mad_state.ema_teacher_logits,
                            temp=float(cfg["weights"]["mad_temp"])
                        )

                loss = loss_ce.float() + aux

                if not torch.isfinite(loss).all():
                    if rank == 0:
                        print(f"[WARN] non-finite loss at epoch={epoch} step={step}, skipping.")
                    optim.zero_grad(set_to_none=True)
                    continue

                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()

                total_loss += float(loss.detach()) * batch["label"].size(0)
                n_seen += batch["label"].size(0)

                if rank == 0 and (step % 50 == 0):
                    print(f"[epoch {epoch:02d} step {step:04d}] train_loss={(total_loss/max(n_seen,1)):.4f}")

            ddp_barrier(ddp)

            # Gather train accuracy across ranks
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                t_corr = torch.tensor([correct_local], device=device, dtype=torch.long)
                t_tot  = torch.tensor([total_local], device=device, dtype=torch.long)
                torch.distributed.all_reduce(t_corr, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(t_tot,  op=torch.distributed.ReduceOp.SUM)
                train_acc = (t_corr.float() / torch.clamp(t_tot.float(), min=1.0)).item()
            else:
                train_acc = correct_local / max(1, total_local)

            val_loss, val_acc, val_f1 = evaluate_ddp(model, val_loader, device, num_classes=4)

            if rank == 0:
                train_loss_epoch = total_loss / max(n_seen, 1)
                print(f"[epoch {epoch:02d}] val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

                hist["epoch"].append(epoch)
                hist["train_loss"].append(train_loss_epoch)
                hist["val_loss"].append(val_loss)
                hist["train_acc"].append(train_acc)
                hist["val_acc"].append(val_acc)
                hist["val_f1"].append(val_f1)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                    ckpt_path = os.path.join(args.out_dir, "best.pt")
                    torch.save({"epoch": epoch, "val_acc": val_acc, "state_dict": state, "config": cfg}, ckpt_path)
                    print(f"[OK] Saved best checkpoint: {ckpt_path}")

        ddp_barrier(ddp)

        # Load best & test
        ckpt_path = os.path.join(args.out_dir, "best.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu")

        if hasattr(model, "module"):
            model.module.load_state_dict(ckpt["state_dict"], strict=True)
        else:
            model.load_state_dict(ckpt["state_dict"], strict=True)

        ddp_barrier(ddp)

        test_loss, test_acc, test_f1 = evaluate_ddp(model, test_loader, device, num_classes=4)

        if rank == 0:
            print(f"\n[TEST] loss={test_loss:.4f} acc={test_acc:.4f} f1={test_f1:.4f}")

            # Save history for ICML curves
            if args.save_reports:
                out_rep = os.path.join(args.out_dir, "reports")
                mkdir(out_rep)
                with open(os.path.join(out_rep, "history.json"), "w") as f:
                    json.dump(hist, f, indent=2)
                save_train_val_curve(out_rep, hist)

    finally:
        ddp_cleanup()

if __name__ == "__main__":
    main()
