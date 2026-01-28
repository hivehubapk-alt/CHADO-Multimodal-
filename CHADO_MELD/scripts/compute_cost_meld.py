import os
import time
import yaml
import argparse
import numpy as np
import torch
import inspect

from torch.utils.data import DataLoader
from torch.amp import autocast

from src.data.meld_dataset import (
    MeldDataset,
    collate_meld,
    build_label_map_from_order,
    EMO_ORDER_7,
)

# Prefer baseline model for ablation cost table (T/A/V/TA/TV/AV/TAV).
# If you want CHADO-cost too, see notes at the end.
from src.models.baseline_trimodal import TriModalBaseline


def count_params_m(model: torch.nn.Module) -> float:
    n = sum(p.numel() for p in model.parameters())
    return n / 1e6


def build_dataset(cfg, csv_path, label_map):
    """
    Signature-safe dataset builder (avoids the label_map double-pass issue).
    """
    sig = inspect.signature(MeldDataset.__init__)
    accepted = set(sig.parameters.keys())

    candidates = {
        "csv_path": csv_path,
        "text_col": cfg["data"].get("text_col", "text"),
        "label_col": cfg["data"].get("label_col", "emotion"),
        "audio_path_col": cfg["data"].get("audio_path_col", "audio_path"),
        "video_path_col": cfg["data"].get("video_path_col", "video_path"),
        "utt_id_col": cfg["data"].get("utt_id_col", "utt_id"),
        "num_frames": cfg["data"]["num_frames"],
        "frame_size": cfg["data"]["frame_size"],
        "sample_rate": cfg["data"]["sample_rate"],
        "max_audio_seconds": cfg["data"]["max_audio_seconds"],
        "label_map": label_map,
        "use_text": True,
        "use_audio": True,
        "use_video": True,
        "text_model_name": cfg["model"].get("text_model_name", None),
    }

    kwargs = {k: v for k, v in candidates.items() if k in accepted and v is not None}
    ds = MeldDataset(**kwargs)

    if not hasattr(ds, "tokenizer"):
        raise RuntimeError("MeldDataset must expose ds.tokenizer for collate_meld.")
    return ds


def make_loader(ds, cfg, split_name="test"):
    bs = cfg["train"].get("batch_size_per_gpu", 6)
    nw = cfg["train"].get("num_workers", 6)

    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        drop_last=False,
        collate_fn=lambda b: collate_meld(
            b,
            ds.tokenizer,
            use_text=True, use_audio=True, use_video=True
        ),
    )


def build_model_for_ablation(cfg, ablation: str, device: torch.device):
    """
    ablation in {"T","A","V","TA","TV","AV","TAV"}.
    """
    use_text = "T" in ablation
    use_audio = "A" in ablation
    use_video = "V" in ablation

    model = TriModalBaseline(
        text_model_name=cfg["model"]["text_model_name"],
        audio_model_name=cfg["model"]["audio_model_name"],
        video_model_name=cfg["model"]["video_model_name"],
        num_classes=cfg["data"]["num_classes"],
        proj_dim=cfg["model"]["proj_dim"],
        dropout=cfg["model"]["dropout"],
        use_text=use_text,
        use_audio=use_audio,
        use_video=use_video,
        use_gated_fusion=cfg["model"].get("use_gated_fusion", True),
    ).to(device)

    model.eval()
    return model


@torch.no_grad()
def benchmark_inference(
    model,
    loader,
    device: torch.device,
    amp: bool,
    runs: int,
    warmup: int,
):
    """
    Returns (ms_per_sample, peak_mem_mb)
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    # Build a fixed list of batches to avoid DataLoader jitter in timing
    batches = []
    for batch in loader:
        batches.append(batch)
        if len(batches) * loader.batch_size >= (warmup + runs) * loader.batch_size:
            break
    if len(batches) == 0:
        raise RuntimeError("Loader returned no batches (check CSV paths).")

    def move_batch(batch):
        labels = batch.labels.to(device, non_blocking=True)
        text = {k: v.to(device, non_blocking=True) for k, v in batch.text_input.items()} if batch.text_input else None
        audio = batch.audio_wave.to(device, non_blocking=True) if batch.audio_wave is not None else None
        video = batch.video_frames.to(device, non_blocking=True) if batch.video_frames is not None else None
        return labels, text, audio, video

    # Warmup (not timed)
    for i in range(min(warmup, len(batches))):
        _, text, audio, video = move_batch(batches[i])
        with autocast("cuda", enabled=amp):
            _ = model(text_input=text, audio_wave=audio, video_frames=video, modality_mask=None)
        torch.cuda.synchronize(device)

    # Timed runs
    times = []
    n_samples = 0

    for i in range(min(runs, len(batches))):
        batch = batches[i]
        bs = batch.labels.size(0)
        _, text, audio, video = move_batch(batch)

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        with autocast("cuda", enabled=amp):
            _ = model(text_input=text, audio_wave=audio, video_frames=video, modality_mask=None)

        torch.cuda.synchronize(device)
        t1 = time.perf_counter()

        times.append((t1 - t0) * 1000.0)  # ms per batch
        n_samples += bs

    ms_per_sample = (np.sum(times) / max(1, n_samples))
    peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return float(ms_per_sample), float(peak_mem_mb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="e.g., configs/baseline_meld.yaml")
    ap.add_argument("--splits", default="test", choices=["test", "val"], help="Which split to benchmark")
    ap.add_argument("--runs", type=int, default=200, help="Timed batches")
    ap.add_argument("--warmup", type=int, default=30, help="Warmup batches")
    ap.add_argument("--amp", action="store_true", help="Force AMP on (otherwise read from config)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda")

    label_map = build_label_map_from_order(EMO_ORDER_7)
    split_csv = cfg["data"]["test_csv"] if args.splits == "test" else cfg["data"]["val_csv"]

    # Dataset always built with all modalities available; ablation toggles in model
    ds = build_dataset(cfg, split_csv, label_map)
    loader = make_loader(ds, cfg, split_name=args.splits)

    amp_enabled = bool(args.amp) or bool(cfg["train"].get("amp", True))

    ablations = ["T", "A", "V", "TA", "TV", "AV", "TAV"]

    print(f"\n=== Computational Cost Comparison ({args.splits.upper()} Inference) ===")
    print(f"{'Ablation':<8} {'Params(M)':>10} {'Infer(ms/sample)':>16} {'PeakMem(MB)':>12}")

    for ab in ablations:
        model = build_model_for_ablation(cfg, ab, device)
        params_m = count_params_m(model)

        ms_per_sample, peak_mb = benchmark_inference(
            model=model,
            loader=loader,
            device=device,
            amp=amp_enabled,
            runs=args.runs,
            warmup=args.warmup,
        )

        print(f"{ab:<8} {params_m:>10.2f} {ms_per_sample:>16.3f} {peak_mb:>12.1f}")

    print()


if __name__ == "__main__":
    main()
