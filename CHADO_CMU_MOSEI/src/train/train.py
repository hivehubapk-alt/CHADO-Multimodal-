# import os
# import csv
# import argparse
# import warnings
# from typing import Any, Dict, Tuple

# import torch
# import torch.distributed as dist
# from torch.utils.data import DataLoader, DistributedSampler

# from src.chado.config import load_yaml
# from src.chado.objective import ChadoObjective
# from src.chado.metrics import compute_acc_wf1
# from src.chado.trainer_utils import tune_thresholds, apply_thresholds
# from src.chado.calibration import temperature_scale_logits
# from src.chado.report import multilabel_prf_report, format_report

# from src.chado.causal import CausalDisentangler
# from src.chado.self_correction import SelfCorrectionHead

# from src.datasets.mosei_dataset import MoseiCSDDataset, mosei_collate_fn
# from src.models.baseline_fusion import BaselineFusion


# EMO_NAMES = ["happy", "sad", "angry", "fearful", "disgust", "surprise"]


# def suppress_warnings():
#     warnings.filterwarnings("ignore")
#     os.environ["PYTHONWARNINGS"] = "ignore"
#     os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "OFF")
#     os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")


# def seed_all(seed: int):
#     import random
#     import numpy as np
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


# def is_main() -> bool:
#     return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


# def ddp_setup(enabled: bool) -> bool:
#     if not enabled:
#         return False
#     if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
#         dist.init_process_group(backend="nccl")
#         torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
#         return True
#     return False


# def _gather_object(py_obj):
#     if not dist.is_available() or not dist.is_initialized():
#         return [py_obj]
#     obj_list = [None for _ in range(dist.get_world_size())]
#     dist.all_gather_object(obj_list, py_obj)
#     return obj_list


# def _broadcast_tensor(t: torch.Tensor, src: int = 0) -> torch.Tensor:
#     if dist.is_available() and dist.is_initialized():
#         dist.broadcast(t, src=src)
#     return t


# def _broadcast_model_params(model, src: int = 0):
#     if not dist.is_available() or not dist.is_initialized():
#         return
#     m = model.module if hasattr(model, "module") else model
#     for p in m.parameters():
#         dist.broadcast(p.data, src=src)


# def deep_set(cfg: Dict[str, Any], dotted_key: str, value: Any):
#     keys = dotted_key.split(".")
#     cur = cfg
#     for k in keys[:-1]:
#         if k not in cur or not isinstance(cur[k], dict):
#             cur[k] = {}
#         cur = cur[k]
#     cur[keys[-1]] = value


# def parse_value(v: str):
#     vv = v.strip().lower()
#     if vv in ("true", "false"):
#         return vv == "true"
#     try:
#         if "." in vv:
#             return float(v)
#         return int(v)
#     except Exception:
#         return v


# @torch.no_grad()
# def compute_pos_neg_rank0(train_manifest: str, ablation: str, label_thr: float):
#     ds = MoseiCSDDataset(train_manifest, ablation=ablation, label_thr=label_thr)
#     loader = DataLoader(ds, batch_size=256, num_workers=0, shuffle=False, collate_fn=mosei_collate_fn)
#     pos = torch.zeros((6,), dtype=torch.long)
#     neg = torch.zeros((6,), dtype=torch.long)
#     for batch in loader:
#         y = (batch["label"] > 0.5).long()
#         pos += y.sum(dim=0)
#         neg += (1 - y).sum(dim=0)
#     pos = torch.clamp(pos, min=1)
#     neg = torch.clamp(neg, min=1)
#     pos_weight = (neg.float() / pos.float())
#     prior_bias = torch.log(pos.float() / neg.float())
#     return pos_weight, prior_bias, pos, neg


# def forward_model(model, batch: Dict[str, Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
#     inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
#     out = model(inputs)
#     if isinstance(out, (tuple, list)) and len(out) == 2:
#         logits, z = out
#         return logits, z
#     logits = out
#     return logits, logits


# @torch.no_grad()
# def collect_logits_and_labels(model, loader, device) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Returns logits, y on CPU aggregated across ranks.
#     """
#     model.eval()
#     all_logits, all_y = [], []
#     for batch in loader:
#         y = batch["label"].to(device)
#         logits, _ = forward_model(model, batch, device)
#         all_logits.append(logits.detach().cpu())
#         all_y.append(y.detach().cpu())

#     logits_local = torch.cat(all_logits, dim=0) if all_logits else torch.zeros((0, 6))
#     y_local = torch.cat(all_y, dim=0) if all_y else torch.zeros((0, 6))

#     gl = _gather_object(logits_local)
#     gy = _gather_object(y_local)

#     if dist.is_available() and dist.is_initialized():
#         logits = torch.cat(gl, dim=0)
#         y = torch.cat(gy, dim=0)
#     else:
#         logits, y = gl[0], gy[0]

#     return logits, y


# def build_model(cfg: Dict[str, Any], ablation: str) -> torch.nn.Module:
#     mcfg = cfg["model"]
#     use_audio = ablation in ("TA", "TAV") and bool(mcfg.get("use_audio", True))
#     use_video = ablation in ("TV", "TAV") and bool(mcfg.get("use_video", True))
#     model = BaselineFusion(
#         num_classes=6,
#         d_model=int(mcfg.get("d_model", 256)),
#         use_audio=use_audio,
#         use_video=use_video,
#         text_model=str(mcfg.get("text_model", "roberta-base")),
#         max_text_len=int(cfg["data"].get("max_text_len", 96)),
#         modality_dropout=float(mcfg.get("modality_dropout", 0.1)),
#     )
#     return model


# def build_optimizer(cfg: Dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
#     ocfg = cfg["optim"]
#     lr_backbone = float(ocfg.get("lr_backbone", 2e-5))
#     lr_head = float(ocfg.get("lr_head", 1e-3))
#     wd = float(ocfg.get("weight_decay", 1e-2))

#     m = model.module if hasattr(model, "module") else model
#     backbone_params = list(m.text_enc.model.parameters())
#     head_params = [p for n, p in m.named_parameters() if not n.startswith("text_enc.model.")]

#     opt = torch.optim.AdamW(
#         [{"params": backbone_params, "lr": lr_backbone},
#          {"params": head_params, "lr": lr_head}],
#         weight_decay=wd
#     )
#     return opt


# def main():
#     suppress_warnings()

#     ap = argparse.ArgumentParser()
#     ap.add_argument("--cfg", type=str, required=True)
#     ap.add_argument("--ablation", type=str, required=True, choices=["T", "TA", "TV", "TAV"])
#     ap.add_argument("--override", action="append", default=[], help="Override config key: a.b.c=value")

#     # NEW: cross-env / LOEO manifest overrides + output routing
#     ap.add_argument("--train_manifest", type=str, default=None)
#     ap.add_argument("--val_manifest", type=str, default=None)
#     ap.add_argument("--test_manifest", type=str, default=None)
#     ap.add_argument("--out_root", type=str, default=None, help="Override output root dir (default: cfg.experiment.output_root or ./outputs)")
#     ap.add_argument("--run_tag", type=str, default="", help="Optional tag appended to output folder")

#     args = ap.parse_args()

#     cfg = load_yaml(args.cfg)
#     for ov in args.override:
#         if "=" not in ov:
#             continue
#         k, v = ov.split("=", 1)
#         deep_set(cfg, k.strip(), parse_value(v))

#     seed_all(int(cfg["experiment"].get("seed", 42)))
#     ddp = ddp_setup(bool(cfg["experiment"].get("ddp", True)))
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     proj = cfg["experiment"]["project_root"]

#     # NEW: use CLI-provided manifests (LOEO), else config, else defaults
#     train_manifest = args.train_manifest or cfg.get("data", {}).get("train_manifest") or f"{proj}/data/manifests/mosei_train.jsonl"
#     val_manifest   = args.val_manifest   or cfg.get("data", {}).get("val_manifest")   or f"{proj}/data/manifests/mosei_val.jsonl"
#     test_manifest  = args.test_manifest  or cfg.get("data", {}).get("test_manifest")  or f"{proj}/data/manifests/mosei_test.jsonl"

#     label_thr = float(cfg["data"].get("label_thr", 0.0))
#     batch_size = int(cfg["data"].get("batch_size", 16))
#     num_workers = int(cfg["data"].get("num_workers", 4))

#     # NEW: fold-safe output root (avoid overwriting)
#     base_out = args.out_root or cfg.get("experiment", {}).get("output_root", "outputs")
#     if args.run_tag:
#         base_out = os.path.join(base_out, args.run_tag)

#     # Higher grid helps reduce FP (and works with acc-driven tuner)
#     thr_grid = list(cfg["data"].get("pred_thr_grid", [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]))

#     eval_test_each_epoch = bool(cfg["experiment"].get("eval_test_each_epoch", True))

#     train_ds = MoseiCSDDataset(train_manifest, ablation=args.ablation, label_thr=label_thr)
#     val_ds = MoseiCSDDataset(val_manifest, ablation=args.ablation, label_thr=label_thr)
#     test_ds = MoseiCSDDataset(test_manifest, ablation=args.ablation, label_thr=label_thr)

#     train_sampler = DistributedSampler(train_ds, shuffle=True) if ddp else None
#     val_sampler = DistributedSampler(val_ds, shuffle=False) if ddp else None
#     test_sampler = DistributedSampler(test_ds, shuffle=False) if ddp else None

#     train_loader = DataLoader(
#         train_ds, batch_size=batch_size, num_workers=num_workers,
#         sampler=train_sampler, shuffle=(train_sampler is None),
#         collate_fn=mosei_collate_fn, pin_memory=True, drop_last=False
#     )
#     val_loader = DataLoader(
#         val_ds, batch_size=batch_size, num_workers=num_workers,
#         sampler=val_sampler, shuffle=False,
#         collate_fn=mosei_collate_fn, pin_memory=True, drop_last=False
#     )
#     test_loader = DataLoader(
#         test_ds, batch_size=batch_size, num_workers=num_workers,
#         sampler=test_sampler, shuffle=False,
#         collate_fn=mosei_collate_fn, pin_memory=True, drop_last=False
#     )

#     model = build_model(cfg, args.ablation).to(device)
#     if ddp:
#         model = torch.nn.parallel.DistributedDataParallel(
#             model, device_ids=[int(os.environ["LOCAL_RANK"])], find_unused_parameters=True
#         )
#     m = model.module if hasattr(model, "module") else model

#     # imbalance + prior bias
#     prior_on = bool(cfg.get("chado", {}).get("prior_bias", {}).get("enable", True))
#     if (not ddp) or dist.get_rank() == 0:
#         pos_weight, prior_bias, pos, neg = compute_pos_neg_rank0(train_manifest, args.ablation, label_thr)
#         if is_main():
#             print("pos:", pos.tolist())
#             print("neg:", neg.tolist())
#             print("pos_weight:", [round(float(x), 4) for x in pos_weight.tolist()])
#             if prior_on:
#                 print("prior bias:", [round(float(x), 4) for x in prior_bias.tolist()])
#         pos_weight = pos_weight.to(device)
#         if prior_on:
#             with torch.no_grad():
#                 m.classifier.bias.copy_(prior_bias.to(device))
#     else:
#         pos_weight = torch.ones((6,), dtype=torch.float32, device=device)

#     pos_weight = _broadcast_tensor(pos_weight, src=0)
#     if ddp:
#         dist.barrier()
#         _broadcast_model_params(model, src=0)
#         dist.barrier()

#     # optional modules
#     d_model = int(cfg["model"].get("d_model", 256))
#     causal_on = bool(cfg.get("chado", {}).get("causal", {}).get("enable", False))
#     causal_lambda = float(cfg.get("chado", {}).get("causal", {}).get("lambda", 0.05))
#     selfcorr_on = bool(cfg.get("chado", {}).get("self_correction", {}).get("enable", False))

#     causal = CausalDisentangler(d_model=d_model).to(device) if causal_on else None
#     selfcorr = SelfCorrectionHead(d_model=d_model, num_classes=6).to(device) if selfcorr_on else None

#     # objective flags
#     ch = cfg.get("chado", {})
#     mad = ch.get("mad", {})
#     hyp = ch.get("hyperbolic", {})
#     ot = ch.get("ot", {})

#     objective = ChadoObjective(
#         pos_weight=pos_weight,
#         mad_enable=bool(mad.get("enable", True)),
#         mad_lambda=float(mad.get("lambda", 0.5)),
#         mad_gamma=float(mad.get("gamma", 1.0)),
#         hyp_enable=bool(hyp.get("enable", False)),
#         hyp_lambda=float(hyp.get("lambda", 0.0)),
#         hyp_c=float(hyp.get("c", 1.0)),
#         ot_enable=bool(ot.get("enable", False)),
#         ot_lambda=float(ot.get("lambda", 0.0)),
#         ot_sigma=float(ot.get("perturb_sigma", 0.05)),
#         ot_eps=float(ot.get("sinkhorn_eps", 0.05)),
#         ot_iters=int(ot.get("sinkhorn_iters", 30)),
#         causal_enable=causal_on,
#         causal_lambda=causal_lambda
#     ).to(device)

#     opt = build_optimizer(cfg, model)
#     lr_head = float(cfg["optim"].get("lr_head", 1e-3))
#     if causal_on:
#         opt.add_param_group({"params": causal.parameters(), "lr": lr_head})
#     if selfcorr_on:
#         opt.add_param_group({"params": selfcorr.parameters(), "lr": lr_head})

#     epochs = int(cfg["optim"].get("epochs", 10))
#     grad_clip = float(cfg["optim"].get("grad_clip", 1.0))

#     # curve logs
#     curve = {
#         "epoch": [],
#         "train_loss": [],
#         "val_acc": [],
#         "val_wf1": [],
#         "test_acc": [],
#         "test_wf1": [],
#         "T": [],
#     }

#     best_val_wf1 = -1.0
#     best_state = None
#     best_thr = torch.full((6,), 0.5, dtype=torch.float32)
#     best_T = 1.0

#     for epoch in range(1, epochs + 1):
#         if ddp and train_sampler is not None:
#             train_sampler.set_epoch(epoch)

#         model.train()
#         if causal_on:
#             causal.train()
#         if selfcorr_on:
#             selfcorr.train()

#         total_loss = 0.0
#         n = 0

#         for batch in train_loader:
#             y = batch["label"].to(device)
#             logits1, z = forward_model(model, batch, device)

#             z_shared, z_spec = (None, None)
#             if causal_on:
#                 z_shared, z_spec = causal(z)
#                 logits1 = m.classifier(z_shared)

#             logits = logits1
#             if selfcorr_on:
#                 logits = selfcorr(z_shared if (causal_on and z_shared is not None) else z, logits1)

#             loss = objective(logits, y, z, z_shared=z_shared, z_spec=z_spec)

#             opt.zero_grad(set_to_none=True)
#             loss.backward()

#             if grad_clip and grad_clip > 0:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
#                 if causal_on:
#                     torch.nn.utils.clip_grad_norm_(causal.parameters(), grad_clip)
#                 if selfcorr_on:
#                     torch.nn.utils.clip_grad_norm_(selfcorr.parameters(), grad_clip)

#             opt.step()

#             bs = y.size(0)
#             total_loss += float(loss.item()) * bs
#             n += bs

#         tr_loss = total_loss / max(1, n)

#         # ---------- validation ----------
#         val_logits, val_y = collect_logits_and_labels(model, val_loader, device)
#         T = float(temperature_scale_logits(val_logits, val_y))
#         val_logits_cal = val_logits / T
#         val_probs = torch.sigmoid(val_logits_cal)

#         # acc-driven threshold tuning
#         thr_vec = tune_thresholds(val_logits_cal, val_y, grid=thr_grid, objective="acc")

#         val_pred = apply_thresholds(val_probs, thr_vec)
#         val_true = (val_y > 0.5).int()
#         val_acc, val_wf1, _ = compute_acc_wf1(val_true, val_pred)

#         test_acc = test_wf1 = None
#         if eval_test_each_epoch:
#             test_logits, test_y = collect_logits_and_labels(model, test_loader, device)
#             test_logits_cal = test_logits / T
#             test_probs = torch.sigmoid(test_logits_cal)
#             test_pred = apply_thresholds(test_probs, thr_vec.detach().cpu())
#             test_true = (test_y > 0.5).int()
#             test_acc, test_wf1, _ = compute_acc_wf1(test_true, test_pred)

#         if is_main():
#             print(f"[E{epoch:02d}] loss={tr_loss:.4f}  val_Acc={val_acc*100:.2f}% val_WF1={val_wf1*100:.2f}  T={T:.3f}")
#             if eval_test_each_epoch:
#                 print(f"          test_Acc={test_acc*100:.2f}% test_WF1={test_wf1*100:.2f}")
#             print("  thr:", [round(float(x), 3) for x in thr_vec.tolist()])

#         # update curves (rank0 only)
#         if (not ddp) or dist.get_rank() == 0:
#             curve["epoch"].append(epoch)
#             curve["train_loss"].append(tr_loss)
#             curve["val_acc"].append(float(val_acc))
#             curve["val_wf1"].append(float(val_wf1))
#             curve["T"].append(float(T))
#             if eval_test_each_epoch:
#                 curve["test_acc"].append(float(test_acc))
#                 curve["test_wf1"].append(float(test_wf1))
#             else:
#                 curve["test_acc"].append(None)
#                 curve["test_wf1"].append(None)

#             # best by val WF1
#             if val_wf1 > best_val_wf1:
#                 best_val_wf1 = float(val_wf1)
#                 best_state = m.state_dict()
#                 best_thr = thr_vec.clone()
#                 best_T = float(T)

#         if ddp:
#             dist.barrier()
#             best_t = torch.tensor([best_val_wf1], device=device, dtype=torch.float32)
#             dist.broadcast(best_t, src=0)
#             best_val_wf1 = float(best_t.item())
#             dist.barrier()

#     # ---------- load best weights ----------
#     if ddp:
#         dist.barrier()
#         if dist.get_rank() == 0 and best_state is not None:
#             m.load_state_dict(best_state)
#         dist.barrier()
#         _broadcast_model_params(model, src=0)
#         dist.barrier()
#     else:
#         if best_state is not None:
#             m.load_state_dict(best_state)

#     # broadcast thresholds + temperature
#     best_thr = best_thr.to(device)
#     best_thr = _broadcast_tensor(best_thr, src=0)

#     T_tensor = torch.tensor([best_T], device=device, dtype=torch.float32)
#     T_tensor = _broadcast_tensor(T_tensor, src=0)
#     best_T = float(T_tensor.item())

#     # ---------- save checkpoint (rank0 only) ----------
#     if is_main() and best_state is not None:
#         ckpt_dir = os.path.join(base_out, "checkpoints")
#         os.makedirs(ckpt_dir, exist_ok=True)
#         ckpt_path = os.path.join(ckpt_dir, f"{args.ablation}_best.pt")
#         torch.save(best_state, ckpt_path)
#         print(f"[CKPT] saved: {ckpt_path}")

#     # ---------- save curves (rank0 only) ----------
#     if is_main():
#         log_dir = os.path.join(base_out, "logs")
#         os.makedirs(log_dir, exist_ok=True)
#         pt_path = os.path.join(log_dir, f"{args.ablation}_curves.pt")
#         torch.save(curve, pt_path)

#         csv_path = os.path.join(log_dir, f"{args.ablation}_curves.csv")
#         with open(csv_path, "w", newline="", encoding="utf-8") as f:
#             w = csv.writer(f)
#             w.writerow(["epoch", "train_loss", "val_acc", "val_wf1", "test_acc", "test_wf1", "T"])
#             for i in range(len(curve["epoch"])):
#                 w.writerow([
#                     curve["epoch"][i],
#                     curve["train_loss"][i],
#                     curve["val_acc"][i],
#                     curve["val_wf1"][i],
#                     curve["test_acc"][i],
#                     curve["test_wf1"][i],
#                     curve["T"][i],
#                 ])

#         print(f"[CURVE] saved: {pt_path}")
#         print(f"[CURVE] saved: {csv_path}")

#     # ---------- final TEST ----------
#     test_logits, test_y = collect_logits_and_labels(model, test_loader, device)
#     test_logits_cal = test_logits / best_T
#     test_probs = torch.sigmoid(test_logits_cal)
#     test_pred = apply_thresholds(test_probs, best_thr.detach().cpu())

#     test_true = (test_y > 0.5).int()
#     test_acc, test_wf1, conf = compute_acc_wf1(test_true, test_pred)

#     if is_main():
#         pos_rate = test_pred.float().mean(dim=0)
#         print(f"\n[TEST] Acc={test_acc*100:.2f}%  WF1={test_wf1*100:.2f}  T={best_T:.3f}")
#         print("Tuned thresholds:", [float(x) for x in best_thr.detach().cpu().tolist()])
#         print("test_pred_pos_rate:", [round(float(x), 3) for x in pos_rate.tolist()])
#         print("test_prob_mean:", [round(float(x), 3) for x in test_probs.mean(dim=0).tolist()])

#         print("\nPer-emotion confusion counts: [TP, FP, FN, TN]")
#         for i, n in enumerate(EMO_NAMES):
#             tp, fp, fn, tn = conf[i].tolist()
#             print(f"  {n:8s}: TP={tp:5d} FP={fp:5d} FN={fn:5d} TN={tn:5d}")

#         rows, summary, _ = multilabel_prf_report(test_true, test_pred, class_names=EMO_NAMES)
#         print(format_report(rows, summary))
#         print(f"\nWeighted Avg F1: {summary['weighted_avg']['f1']*100:.2f}")

#     if ddp:
#         dist.destroy_process_group()


# if __name__ == "__main__":
#     main()
import os
import csv
import argparse
import warnings
from typing import Any, Dict, Tuple, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from src.chado.config import load_yaml
from src.chado.objective import ChadoObjective
from src.chado.metrics import compute_acc_wf1
from src.chado.trainer_utils import tune_thresholds, apply_thresholds
from src.chado.calibration import temperature_scale_logits
from src.chado.report import multilabel_prf_report, format_report

from src.chado.causal import CausalDisentangler
from src.chado.self_correction import SelfCorrectionHead

from src.datasets.mosei_dataset import MoseiCSDDataset, mosei_collate_fn
from src.models.baseline_fusion import BaselineFusion

EMO_NAMES = ["happy", "sad", "angry", "fearful", "disgust", "surprise"]


def suppress_warnings():
    warnings.filterwarnings("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "OFF")
    os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")


def seed_all(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_main() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def ddp_setup(enabled: bool) -> bool:
    if not enabled:
        return False
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return True
    return False


def _gather_object(py_obj):
    if not dist.is_available() or not dist.is_initialized():
        return [py_obj]
    obj_list = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(obj_list, py_obj)
    return obj_list


def _broadcast_tensor(t: torch.Tensor, src: int = 0) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(t, src=src)
    return t


def _broadcast_model_params(model, src: int = 0):
    if not dist.is_available() or not dist.is_initialized():
        return
    m = model.module if hasattr(model, "module") else model
    for p in m.parameters():
        dist.broadcast(p.data, src=src)


def deep_set(cfg: Dict[str, Any], dotted_key: str, value: Any):
    keys = dotted_key.split(".")
    cur = cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def parse_value(v: str):
    vv = v.strip().lower()
    if vv in ("true", "false"):
        return vv == "true"
    try:
        if "." in vv:
            return float(v)
        return int(v)
    except Exception:
        return v


@torch.no_grad()
def compute_pos_neg_rank0(train_manifest: str, ablation: str, label_thr: float):
    ds = MoseiCSDDataset(train_manifest, ablation=ablation, label_thr=label_thr)
    loader = DataLoader(ds, batch_size=256, num_workers=0, shuffle=False, collate_fn=mosei_collate_fn)
    pos = torch.zeros((6,), dtype=torch.long)
    neg = torch.zeros((6,), dtype=torch.long)
    for batch in loader:
        y = (batch["label"] > 0.5).long()
        pos += y.sum(dim=0)
        neg += (1 - y).sum(dim=0)
    pos = torch.clamp(pos, min=1)
    neg = torch.clamp(neg, min=1)
    pos_weight = (neg.float() / pos.float())
    prior_bias = torch.log(pos.float() / neg.float())
    return pos_weight, prior_bias, pos, neg


def forward_model(model, batch: Dict[str, Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    out = model(inputs)
    if isinstance(out, (tuple, list)) and len(out) == 2:
        logits, z = out
        return logits, z
    logits = out
    return logits, logits


@torch.no_grad()
def collect_logits_and_labels(model, loader, device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits, all_y = [], []
    for batch in loader:
        y = batch["label"].to(device)
        logits, _ = forward_model(model, batch, device)
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())

    logits_local = torch.cat(all_logits, dim=0) if all_logits else torch.zeros((0, 6))
    y_local = torch.cat(all_y, dim=0) if all_y else torch.zeros((0, 6))

    gl = _gather_object(logits_local)
    gy = _gather_object(y_local)

    if dist.is_available() and dist.is_initialized():
        logits = torch.cat(gl, dim=0)
        y = torch.cat(gy, dim=0)
    else:
        logits, y = gl[0], gy[0]
    return logits, y


def build_model(cfg: Dict[str, Any], ablation: str) -> torch.nn.Module:
    mcfg = cfg["model"]
    use_audio = ablation in ("TA", "TAV") and bool(mcfg.get("use_audio", True))
    use_video = ablation in ("TV", "TAV") and bool(mcfg.get("use_video", True))
    model = BaselineFusion(
        num_classes=6,
        d_model=int(mcfg.get("d_model", 256)),
        use_audio=use_audio,
        use_video=use_video,
        text_model=str(mcfg.get("text_model", "roberta-base")),
        max_text_len=int(cfg["data"].get("max_text_len", 96)),
        modality_dropout=float(mcfg.get("modality_dropout", 0.1)),
    )
    return model


def build_optimizer(cfg: Dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    ocfg = cfg["optim"]
    lr_backbone = float(ocfg.get("lr_backbone", 2e-5))
    lr_head = float(ocfg.get("lr_head", 1e-3))
    wd = float(ocfg.get("weight_decay", 1e-2))

    m = model.module if hasattr(model, "module") else model
    backbone_params = list(m.text_enc.model.parameters())
    head_params = [p for n, p in m.named_parameters() if not n.startswith("text_enc.model.")]

    opt = torch.optim.AdamW(
        [{"params": backbone_params, "lr": lr_backbone},
         {"params": head_params, "lr": lr_head}],
        weight_decay=wd
    )
    return opt


def _ensure_dir(p: str):
    if p:
        os.makedirs(p, exist_ok=True)


def main():
    suppress_warnings()

    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True)
    ap.add_argument("--ablation", type=str, required=True, choices=["T", "TA", "TV", "TAV"])
    ap.add_argument("--override", action="append", default=[], help="Override config key: a.b.c=value")

    # NEW: cross-env / custom manifests
    ap.add_argument("--train_manifest", type=str, default=None)
    ap.add_argument("--val_manifest", type=str, default=None)
    ap.add_argument("--test_manifest", type=str, default=None)

    # NEW: write outputs under fold-specific root
    ap.add_argument("--out_root", type=str, default="outputs")

    # NEW: allow disabling DDP even if cfg sets it
    ap.add_argument("--ddp", type=str, default=None, help="true/false to override cfg.experiment.ddp")

    args = ap.parse_args()

    cfg = load_yaml(args.cfg)
    for ov in args.override:
        if "=" not in ov:
            continue
        k, v = ov.split("=", 1)
        deep_set(cfg, k.strip(), parse_value(v))

    if args.ddp is not None:
        deep_set(cfg, "experiment.ddp", parse_value(args.ddp))

    seed_all(int(cfg["experiment"].get("seed", 42)))

    out_root = args.out_root
    ckpt_dir = os.path.join(out_root, "checkpoints")
    log_dir = os.path.join(out_root, "logs")
    _ensure_dir(ckpt_dir)
    _ensure_dir(log_dir)

    ddp = ddp_setup(bool(cfg["experiment"].get("ddp", True)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    proj = cfg["experiment"]["project_root"]

    train_manifest = args.train_manifest or f"{proj}/data/manifests/mosei_train.jsonl"
    val_manifest = args.val_manifest or f"{proj}/data/manifests/mosei_val.jsonl"
    test_manifest = args.test_manifest or f"{proj}/data/manifests/mosei_test.jsonl"

    label_thr = float(cfg["data"].get("label_thr", 0.0))
    batch_size = int(cfg["data"].get("batch_size", 16))
    num_workers = int(cfg["data"].get("num_workers", 4))

    thr_grid = list(cfg["data"].get(
        "pred_thr_grid",
        [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    ))
    eval_test_each_epoch = bool(cfg["experiment"].get("eval_test_each_epoch", True))

    train_ds = MoseiCSDDataset(train_manifest, ablation=args.ablation, label_thr=label_thr)
    val_ds = MoseiCSDDataset(val_manifest, ablation=args.ablation, label_thr=label_thr)
    test_ds = MoseiCSDDataset(test_manifest, ablation=args.ablation, label_thr=label_thr)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if ddp else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if ddp else None
    test_sampler = DistributedSampler(test_ds, shuffle=False) if ddp else None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, num_workers=num_workers,
        sampler=train_sampler, shuffle=(train_sampler is None),
        collate_fn=mosei_collate_fn, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, num_workers=num_workers,
        sampler=val_sampler, shuffle=False,
        collate_fn=mosei_collate_fn, pin_memory=True, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, num_workers=num_workers,
        sampler=test_sampler, shuffle=False,
        collate_fn=mosei_collate_fn, pin_memory=True, drop_last=False
    )

    model = build_model(cfg, args.ablation).to(device)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[int(os.environ["LOCAL_RANK"])], find_unused_parameters=True
        )
    m = model.module if hasattr(model, "module") else model

    # imbalance + prior bias
    prior_on = bool(cfg.get("chado", {}).get("prior_bias", {}).get("enable", True))
    if (not ddp) or dist.get_rank() == 0:
        pos_weight, prior_bias, pos, neg = compute_pos_neg_rank0(train_manifest, args.ablation, label_thr)
        if is_main():
            print("pos:", pos.tolist())
            print("neg:", neg.tolist())
            print("pos_weight:", [round(float(x), 4) for x in pos_weight.tolist()])
            if prior_on:
                print("prior bias:", [round(float(x), 4) for x in prior_bias.tolist()])
        pos_weight = pos_weight.to(device)
        if prior_on:
            with torch.no_grad():
                m.classifier.bias.copy_(prior_bias.to(device))
    else:
        pos_weight = torch.ones((6,), dtype=torch.float32, device=device)

    pos_weight = _broadcast_tensor(pos_weight, src=0)
    if ddp:
        dist.barrier()
        _broadcast_model_params(model, src=0)
        dist.barrier()

    d_model = int(cfg["model"].get("d_model", 256))
    causal_on = bool(cfg.get("chado", {}).get("causal", {}).get("enable", False))
    causal_lambda = float(cfg.get("chado", {}).get("causal", {}).get("lambda", 0.05))
    selfcorr_on = bool(cfg.get("chado", {}).get("self_correction", {}).get("enable", False))

    causal = CausalDisentangler(d_model=d_model).to(device) if causal_on else None
    selfcorr = SelfCorrectionHead(d_model=d_model, num_classes=6).to(device) if selfcorr_on else None

    ch = cfg.get("chado", {})
    mad = ch.get("mad", {})
    hyp = ch.get("hyperbolic", {})
    ot = ch.get("ot", {})

    objective = ChadoObjective(
        pos_weight=pos_weight,
        mad_enable=bool(mad.get("enable", True)),
        mad_lambda=float(mad.get("lambda", 0.5)),
        mad_gamma=float(mad.get("gamma", 1.0)),
        hyp_enable=bool(hyp.get("enable", False)),
        hyp_lambda=float(hyp.get("lambda", 0.0)),
        hyp_c=float(hyp.get("c", 1.0)),
        ot_enable=bool(ot.get("enable", False)),
        ot_lambda=float(ot.get("lambda", 0.0)),
        ot_sigma=float(ot.get("perturb_sigma", 0.05)),
        ot_eps=float(ot.get("sinkhorn_eps", 0.05)),
        ot_iters=int(ot.get("sinkhorn_iters", 30)),
        causal_enable=causal_on,
        causal_lambda=causal_lambda
    ).to(device)

    opt = build_optimizer(cfg, model)
    lr_head = float(cfg["optim"].get("lr_head", 1e-3))
    if causal_on:
        opt.add_param_group({"params": causal.parameters(), "lr": lr_head})
    if selfcorr_on:
        opt.add_param_group({"params": selfcorr.parameters(), "lr": lr_head})

    epochs = int(cfg["optim"].get("epochs", 10))
    grad_clip = float(cfg["optim"].get("grad_clip", 1.0))

    curve = {"epoch": [], "train_loss": [], "val_acc": [], "val_wf1": [], "test_acc": [], "test_wf1": [], "T": []}

    best_val_wf1 = -1.0
    best_state = None
    best_thr = torch.full((6,), 0.5, dtype=torch.float32)
    best_T = 1.0

    for epoch in range(1, epochs + 1):
        if ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        if causal_on:
            causal.train()
        if selfcorr_on:
            selfcorr.train()

        total_loss = 0.0
        n = 0

        for batch in train_loader:
            y = batch["label"].to(device)
            logits1, z = forward_model(model, batch, device)

            z_shared, z_spec = (None, None)
            if causal_on:
                z_shared, z_spec = causal(z)
                logits1 = m.classifier(z_shared)

            logits = logits1
            if selfcorr_on:
                logits = selfcorr(z_shared if (causal_on and z_shared is not None) else z, logits1)

            loss = objective(logits, y, z, z_shared=z_shared, z_spec=z_spec)

            opt.zero_grad(set_to_none=True)
            loss.backward()

            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if causal_on:
                    torch.nn.utils.clip_grad_norm_(causal.parameters(), grad_clip)
                if selfcorr_on:
                    torch.nn.utils.clip_grad_norm_(selfcorr.parameters(), grad_clip)

            opt.step()

            bs = y.size(0)
            total_loss += float(loss.item()) * bs
            n += bs

        tr_loss = total_loss / max(1, n)

        val_logits, val_y = collect_logits_and_labels(model, val_loader, device)
        T = float(temperature_scale_logits(val_logits, val_y))
        val_logits_cal = val_logits / T
        val_probs = torch.sigmoid(val_logits_cal)

        thr_vec = tune_thresholds(val_logits_cal, val_y, grid=thr_grid, objective="acc")
        val_pred = apply_thresholds(val_probs, thr_vec)
        val_true = (val_y > 0.5).int()
        val_acc, val_wf1, _ = compute_acc_wf1(val_true, val_pred)

        test_acc = test_wf1 = None
        if eval_test_each_epoch:
            test_logits, test_y = collect_logits_and_labels(model, test_loader, device)
            test_logits_cal = test_logits / T
            test_probs = torch.sigmoid(test_logits_cal)
            test_pred = apply_thresholds(test_probs, thr_vec.detach().cpu())
            test_true = (test_y > 0.5).int()
            test_acc, test_wf1, _ = compute_acc_wf1(test_true, test_pred)

        if is_main():
            print(f"[E{epoch:02d}] loss={tr_loss:.4f}  val_Acc={val_acc*100:.2f}% val_WF1={val_wf1*100:.2f}  T={T:.3f}")
            if eval_test_each_epoch:
                print(f"          test_Acc={test_acc*100:.2f}% test_WF1={test_wf1*100:.2f}")
            print("  thr:", [round(float(x), 3) for x in thr_vec.tolist()])

        if (not ddp) or dist.get_rank() == 0:
            curve["epoch"].append(epoch)
            curve["train_loss"].append(tr_loss)
            curve["val_acc"].append(float(val_acc))
            curve["val_wf1"].append(float(val_wf1))
            curve["T"].append(float(T))
            curve["test_acc"].append(float(test_acc) if eval_test_each_epoch else None)
            curve["test_wf1"].append(float(test_wf1) if eval_test_each_epoch else None)

            if val_wf1 > best_val_wf1:
                best_val_wf1 = float(val_wf1)
                best_state = m.state_dict()
                best_thr = thr_vec.clone()
                best_T = float(T)

        if ddp:
            dist.barrier()
            best_t = torch.tensor([best_val_wf1], device=device, dtype=torch.float32)
            dist.broadcast(best_t, src=0)
            best_val_wf1 = float(best_t.item())
            dist.barrier()

    if ddp:
        dist.barrier()
        if dist.get_rank() == 0 and best_state is not None:
            m.load_state_dict(best_state)
        dist.barrier()
        _broadcast_model_params(model, src=0)
        dist.barrier()
    else:
        if best_state is not None:
            m.load_state_dict(best_state)

    best_thr = _broadcast_tensor(best_thr.to(device), src=0)
    best_T = float(_broadcast_tensor(torch.tensor([best_T], device=device), src=0).item())

    if is_main() and best_state is not None:
        ckpt_path = os.path.join(ckpt_dir, f"{args.ablation}_best.pt")
        torch.save(best_state, ckpt_path)
        print(f"[CKPT] saved: {ckpt_path}")

    if is_main():
        pt_path = os.path.join(log_dir, f"{args.ablation}_curves.pt")
        torch.save(curve, pt_path)
        csv_path = os.path.join(log_dir, f"{args.ablation}_curves.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "val_acc", "val_wf1", "test_acc", "test_wf1", "T"])
            for i in range(len(curve["epoch"])):
                w.writerow([curve["epoch"][i], curve["train_loss"][i], curve["val_acc"][i],
                            curve["val_wf1"][i], curve["test_acc"][i], curve["test_wf1"][i], curve["T"][i]])
        print(f"[CURVE] saved: {pt_path}")
        print(f"[CURVE] saved: {csv_path}")

    test_logits, test_y = collect_logits_and_labels(model, test_loader, device)
    test_probs = torch.sigmoid((test_logits / best_T))
    test_pred = apply_thresholds(test_probs, best_thr.detach().cpu())
    test_true = (test_y > 0.5).int()
    test_acc, test_wf1, conf = compute_acc_wf1(test_true, test_pred)

    if is_main():
        pos_rate = test_pred.float().mean(dim=0)
        print(f"\n[TEST] Acc={test_acc*100:.2f}%  WF1={test_wf1*100:.2f}  T={best_T:.3f}")
        print("Tuned thresholds:", [float(x) for x in best_thr.detach().cpu().tolist()])
        print("test_pred_pos_rate:", [round(float(x), 3) for x in pos_rate.tolist()])
        print("test_prob_mean:", [round(float(x), 3) for x in test_probs.mean(dim=0).tolist()])

        print("\nPer-emotion confusion counts: [TP, FP, FN, TN]")
        for i, n in enumerate(EMO_NAMES):
            tp, fp, fn, tn = conf[i].tolist()
            print(f"  {n:8s}: TP={tp:5d} FP={fp:5d} FN={fn:5d} TN={tn:5d}")

        rows, summary, _ = multilabel_prf_report(test_true, test_pred, class_names=EMO_NAMES)
        print(format_report(rows, summary))
        print(f"\nWeighted Avg F1: {summary['weighted_avg']['f1']*100:.2f}")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
