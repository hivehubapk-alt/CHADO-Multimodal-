import os
import argparse
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from src.chado.config import load_yaml
from src.chado.metrics import compute_acc_wf1
from src.chado.trainer_utils import apply_thresholds, tune_thresholds
from src.datasets.mosei_dataset import MoseiCSDDataset, mosei_collate_fn
from src.models.baseline_fusion import BaselineFusion


def is_main():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def ddp_setup(enabled: bool):
    if not enabled:
        return False
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return True
    return False


def forward_model(model, batch, device):
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    out = model(inputs)
    if isinstance(out, (tuple, list)) and len(out) == 2:
        logits, z = out
        return logits, z
    logits = out
    return logits, logits


@torch.no_grad()
def collect_logits_labels_probs(model, loader, device):
    model.eval()
    all_logits, all_y, all_probs = [], [], []
    for batch in loader:
        y = batch["label"].to(device)
        logits, _ = forward_model(model, batch, device)
        probs = torch.sigmoid(logits)
        all_logits.append(logits.detach().cpu())
        all_probs.append(probs.detach().cpu())
        all_y.append(y.detach().cpu())
    return torch.cat(all_logits, 0), torch.cat(all_y, 0), torch.cat(all_probs, 0)


def ambiguity_score_from_probs(probs: torch.Tensor):
    u = 1.0 - 2.0 * torch.abs(probs - 0.5)
    u = torch.clamp(u, 0.0, 1.0)
    return u.mean(dim=1)


@torch.no_grad()
def measure_inference_ms(model, loader, device, n_warmup=10, n_meas=50):
    model.eval()
    it = iter(loader)

    for _ in range(n_warmup):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        _ = forward_model(model, batch, device)

    torch.cuda.synchronize()

    times = []
    for _ in range(n_meas):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        bs = batch["label"].shape[0]
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = forward_model(model, batch, device)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        times.append(ms / max(1, bs))

    return float(np.mean(times))


def build_model(cfg, ablation):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True)
    ap.add_argument("--ablation", type=str, required=True, choices=["T", "TA", "TV", "TAV"])
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--ambiguous_topk", type=float, default=0.20)
    ap.add_argument("--ddp", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.cfg)
    proj = cfg["experiment"]["project_root"]
    label_thr = float(cfg["data"].get("label_thr", 0.0))
    bs = int(cfg["data"].get("batch_size", 16))
    nw = int(cfg["data"].get("num_workers", 4))
    thr_grid = list(cfg["data"].get("pred_thr_grid", [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]))

    ddp = ddp_setup(args.ddp)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_manifest = f"{proj}/data/manifests/mosei_train.jsonl"
    val_manifest = f"{proj}/data/manifests/mosei_val.jsonl"
    test_manifest = f"{proj}/data/manifests/mosei_test.jsonl"

    val_ds = MoseiCSDDataset(val_manifest, ablation=args.ablation, label_thr=label_thr)
    test_ds = MoseiCSDDataset(test_manifest, ablation=args.ablation, label_thr=label_thr)

    val_sampler = DistributedSampler(val_ds, shuffle=False) if ddp else None
    test_sampler = DistributedSampler(test_ds, shuffle=False) if ddp else None

    val_loader = DataLoader(val_ds, batch_size=bs, num_workers=nw, sampler=val_sampler,
                            shuffle=False, collate_fn=mosei_collate_fn, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, num_workers=nw, sampler=test_sampler,
                             shuffle=False, collate_fn=mosei_collate_fn, pin_memory=True)

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    model = build_model(cfg, args.ablation).to(device)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state, strict=False)

    # tune thresholds on val
    val_logits, val_y, _ = collect_logits_labels_probs(model, val_loader, device)
    thr_vec = tune_thresholds(val_logits, val_y, grid=thr_grid)

    # test metrics
    test_logits, test_y, test_probs = collect_logits_labels_probs(model, test_loader, device)
    test_true = (test_y > 0.5).int()
    test_pred = apply_thresholds(test_probs, thr_vec)

    overall_acc, overall_wf1, _ = compute_acc_wf1(test_true, test_pred)

    # ambiguous subset (top-k by MAD uncertainty proxy)
    amb = ambiguity_score_from_probs(test_probs)
    k = int(max(1, round(len(amb) * args.ambiguous_topk)))
    idx = torch.topk(amb, k=k, largest=True).indices
    amb_acc, amb_wf1, _ = compute_acc_wf1(test_true[idx], test_pred[idx])

    inf_ms = measure_inference_ms(model, test_loader, device, n_warmup=10, n_meas=50)

    if is_main():
        print("[EXTENDED METRICS]")
        print(f"Overall: Acc={overall_acc*100:.2f}% WF1={overall_wf1*100:.2f}")
        print(f"Ambiguous(top {args.ambiguous_topk*100:.0f}%): Acc={amb_acc*100:.2f}% WF1={amb_wf1*100:.2f}")
        print("Cross-Cultural: NA (no culture metadata in this Kaggle MOSEI dump)")
        print(f"Inference: {inf_ms:.2f} ms/sample")
        print("Thresholds:", [float(x) for x in thr_vec.tolist()])


if __name__ == "__main__":
    main()
