import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.mosei_dataset import MoseiCSDDataset, mosei_collate_fn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="train", choices=["train","val","test"])
    ap.add_argument("--ablation", type=str, default="TA", choices=["TA","TAV"])
    ap.add_argument("--n_batches", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    manifest = f"data/manifests/mosei_{args.split}.jsonl"
    ds = MoseiCSDDataset(manifest, ablation=args.ablation, label_thr=0.0)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=mosei_collate_fn)

    all_mean_abs = []
    all_std = []
    all_nan_frac = []
    all_zero_frac = []
    all_energy = []
    all_max_abs = []

    n_seen = 0
    for bi, batch in enumerate(dl):
        if bi >= args.n_batches:
            break
        a = batch["audio"]  # [B,T,74]
        B = a.shape[0]
        n_seen += B

        a_np = a.numpy()
        nan_frac = np.isnan(a_np).mean()
        inf_frac = np.isinf(a_np).mean()

        mean_abs = np.mean(np.abs(a_np))
        std = np.std(a_np)
        max_abs = np.max(np.abs(a_np))

        # per-sample energy and all-zero detection
        energy = np.mean(np.abs(a_np), axis=(1,2))  # [B]
        # a sample is "all-zero" if max abs is ~0
        sample_max = np.max(np.abs(a_np), axis=(1,2))
        zero_frac = np.mean(sample_max < 1e-8)

        all_mean_abs.append(mean_abs)
        all_std.append(std)
        all_max_abs.append(max_abs)
        all_nan_frac.append(nan_frac + inf_frac)
        all_zero_frac.append(zero_frac)
        all_energy.append(energy)

        if bi == 0:
            print("Audio tensor shape:", tuple(a.shape))
            print("First batch energy stats:",
                  "min", float(np.min(energy)),
                  "p50", float(np.median(energy)),
                  "p90", float(np.quantile(energy, 0.90)),
                  "max", float(np.max(energy)))

    all_energy = np.concatenate(all_energy, axis=0) if len(all_energy) else np.array([])
    print("\n[Audio sanity summary]")
    print("samples checked:", n_seen)
    print("mean(|a|):", float(np.mean(all_mean_abs)))
    print("std(a):", float(np.mean(all_std)))
    print("max(|a|):", float(np.mean(all_max_abs)))
    print("nan/inf fraction:", float(np.mean(all_nan_frac)))
    print("all-zero sample fraction:", float(np.mean(all_zero_frac)))
    if all_energy.size > 0:
        print("energy percentiles:",
              "p01", float(np.quantile(all_energy, 0.01)),
              "p10", float(np.quantile(all_energy, 0.10)),
              "p50", float(np.quantile(all_energy, 0.50)),
              "p90", float(np.quantile(all_energy, 0.90)),
              "p99", float(np.quantile(all_energy, 0.99)))

if __name__ == "__main__":
    main()
