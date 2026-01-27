#!/usr/bin/env python3
import argparse
import time
import torch
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["config"]

    # Build four modality settings: Text, T+A, T+V, T+A+V
    from chado_lib.models.chado import CHADO

    device = torch.device(args.device)

    def build(use_text, use_audio, use_video):
        m = CHADO(cfg["models"]["text_model"], cfg["models"]["audio_model"], cfg["models"]["vision_model"],
                  num_classes=4, use_text=use_text, use_audio=use_audio, use_video=use_video).to(device)
        m.load_state_dict(ckpt["state_dict"], strict=True)
        m.eval()
        return m

    models = {
        "Text": build(True, False, False),
        "Text+Audio": build(True, True, False),
        "Text+Video": build(True, False, True),
        "Text+Audio+Video": build(True, True, True),
    }

    # Dummy inputs with correct shapes; cost comparison is relative.
    B = args.batch_size
    max_len = int(cfg["data"]["max_text_len"])
    T = int(cfg["data"]["n_frames"])
    wav_len = int(float(cfg["data"]["audio_sec"]) * 16000)

    input_ids = torch.zeros((B, max_len), dtype=torch.long, device=device)
    attention_mask = torch.ones((B, max_len), dtype=torch.long, device=device)
    wav = torch.zeros((B, wav_len), dtype=torch.float32, device=device)
    pixel_values = torch.zeros((B, T, 3, 224, 224), dtype=torch.float32, device=device)

    def measure(m):
        torch.cuda.reset_peak_memory_stats(device)
        # warmup
        for _ in range(10):
            with torch.cuda.amp.autocast(enabled=args.amp):
                _ = m(input_ids, attention_mask, wav, pixel_values)
        torch.cuda.synchronize(device)

        t0 = time.time()
        iters = 50
        for _ in range(iters):
            with torch.cuda.amp.autocast(enabled=args.amp):
                _ = m(input_ids, attention_mask, wav, pixel_values)
        torch.cuda.synchronize(device)
        t1 = time.time()

        latency_ms = (t1 - t0) * 1000.0 / iters
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        params_m = sum(p.numel() for p in m.parameters()) / 1e6
        return params_m, peak_mem_mb, latency_ms

    print("\n=== Computational Cost Comparison (printed only) ===")
    print(f"Device: {device} | batch_size={B} | AMP={args.amp}")
    print("Model\t\t\tParams(M)\tPeakMem(MB)\tLatency(ms/iter)")
    for name, m in models.items():
        p, mem, lat = measure(m)
        print(f"{name:16s}\t{p:8.2f}\t{mem:10.1f}\t{lat:12.2f}")

if __name__ == "__main__":
    main()
