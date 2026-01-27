import argparse, os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

EMO = ["happy","sad","angry","fearful","disgust","surprise"]

def load_cache(d):
    probs  = torch.load(os.path.join(d, "test_probs.pt")).float()   # [N,C]
    labels = torch.load(os.path.join(d, "test_labels.pt")).float()
    errors = torch.load(os.path.join(d, "test_errors.pt")).float()  # [N]
    mad    = torch.load(os.path.join(d, "test_mad.pt")).float()     # [N]
    return probs, labels, errors, mad

def sinkhorn_cost(p, q, eps=0.05, iters=50):
    """
    Simple 1D Sinkhorn-like approximation for distribution shift across classes.
    Here we treat class index as a line (0..C-1) and compute transport cost.
    """
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    c = p.numel()
    x = torch.arange(c, dtype=torch.float32)
    M = (x[:,None] - x[None,:]).abs()  # [C,C] cost
    K = torch.exp(-M / eps)

    u = torch.ones(c) / c
    v = torch.ones(c) / c
    for _ in range(iters):
        u = p / (K @ v + 1e-12)
        v = q / (K.t() @ u + 1e-12)
    P = torch.diag(u) @ K @ torch.diag(v)
    return float((P * M).sum().item())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="outputs/cross_env")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out_dir", default="outputs/cross_env/analysis")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for fold in range(args.k):
        for abl in ["T","TA","TV","TAV"]:
            d = os.path.join(args.root, f"fold_{fold}", abl, "cache")
            if not os.path.isdir(d):
                continue
            probs, labels, errors, mad = load_cache(d)

            # ambiguity strata: top 20% MAD
            thr = torch.quantile(mad, 0.80).item()
            hi = (mad >= thr)
            lo = (mad < thr)

            err_rate_all = errors.mean().item()
            err_rate_hi = errors[hi].mean().item() if hi.any() else np.nan
            err_rate_lo = errors[lo].mean().item() if lo.any() else np.nan

            mad_mean = mad.mean().item()
            mad_hi = mad[hi].mean().item() if hi.any() else np.nan

            # OT distance between average predicted distro and average true label distro
            # (proxy for distributional mismatch in the held-out environment)
            p_pred = probs.mean(dim=0)
            p_true = (labels > 0.5).float().mean(dim=0)
            ot = sinkhorn_cost(p_pred, p_true, eps=0.05, iters=60)

            rows.append({
                "fold": fold, "ablation": abl,
                "mad_mean": mad_mean,
                "mad_top20_mean": mad_hi,
                "err_rate": err_rate_all,
                "err_rate_top20": err_rate_hi,
                "err_rate_low80": err_rate_lo,
                "ot_pred_to_true": ot
            })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.out_dir, "loeo_mad_ot.csv")
    df.to_csv(out_csv, index=False)
    print(f"[SAVED] {out_csv}")

    # Plot: OT vs modality (mean±std)
    for col, fname, ylabel in [
        ("mad_mean", "loeo_mad_mean.png", "Mean MAD"),
        ("ot_pred_to_true", "loeo_ot.png", "OT distance (pred ↔ true)")
    ]:
        g = df.groupby("ablation")[col].agg(["mean","std"]).reindex(["T","TA","TV","TAV"]).reset_index()
        x = np.arange(len(g))
        plt.figure(figsize=(6.5, 3.8))
        plt.bar(x, g["mean"].values, yerr=g["std"].values, capsize=4)
        plt.xticks(x, g["ablation"].values)
        plt.ylabel(ylabel)
        plt.xlabel("Modality")
        plt.grid(True, axis="y", alpha=0.25)
        outp = os.path.join(args.out_dir, fname)
        plt.tight_layout()
        plt.savefig(outp, dpi=300)
        print(f"[SAVED] {outp}")

if __name__ == "__main__":
    main()
