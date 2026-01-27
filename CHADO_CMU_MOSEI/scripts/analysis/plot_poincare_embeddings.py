import os
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt


EMO_NAMES = ["happy", "sad", "angry", "fearful", "disgust", "surprise"]


def pca_2d(X: np.ndarray) -> np.ndarray:
    """
    Simple PCA to 2D using SVD (no sklearn dependency).
    X: [N,D]
    return: [N,2]
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    # SVD on covariance-like: (N,D) -> principal directions in Vt
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:2].T  # [N,2]
    return Z


def robust_standardize(Z: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Robust scaling using median and MAD-like spread (percentile).
    """
    med = np.median(Z, axis=0, keepdims=True)
    Zc = Z - med
    scale = np.percentile(np.abs(Zc), 90, axis=0, keepdims=True) + eps
    return Zc / scale


def to_poincare_disk(Z: np.ndarray, max_r: float = 0.99) -> np.ndarray:
    """
    Map 2D Euclidean coordinates to inside Poincaré disk using smooth radial squash.
    """
    r = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    # squash radius to (0, max_r)
    r_new = np.tanh(r) * max_r
    Z_disk = Z / r * r_new
    return Z_disk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="outputs/cache", type=str)
    ap.add_argument("--out", default="outputs/analysis/plots/poincare_disk.png", type=str)
    ap.add_argument("--title", default="Poincaré Disk Embedding (size = ambiguity)", type=str)
    ap.add_argument("--max_points", default=0, type=int, help="0 = all; else subsample for clarity")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    emb_path = os.path.join(args.cache_dir, "test_embeddings.pt")
    lab_path = os.path.join(args.cache_dir, "test_labels.pt")
    mad_path = os.path.join(args.cache_dir, "test_mad.pt")

    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Missing: {emb_path}. Run cache_test_outputs.py first (must save test_embeddings.pt).")
    if not os.path.exists(lab_path):
        raise FileNotFoundError(f"Missing: {lab_path}. Run cache_test_outputs.py first (must save test_labels.pt).")
    if not os.path.exists(mad_path):
        raise FileNotFoundError(f"Missing: {mad_path}. Run cache_test_outputs.py first (must save test_mad.pt).")

    emb = torch.load(emb_path, map_location="cpu").float().numpy()   # [N,D]
    y = torch.load(lab_path, map_location="cpu").float().numpy()     # [N,6]
    mad = torch.load(mad_path, map_location="cpu").float().numpy()   # [N]

    N = emb.shape[0]
    if args.max_points and args.max_points > 0 and args.max_points < N:
        idx = np.random.RandomState(42).choice(N, size=args.max_points, replace=False)
        emb = emb[idx]
        y = y[idx]
        mad = mad[idx]

    # dominant label for coloring (multi-label -> argmax of y)
    dom = np.argmax(y, axis=1)

    # 2D projection + robust scaling + disk mapping
    Z = pca_2d(emb)
    Z = robust_standardize(Z)
    Z = to_poincare_disk(Z, max_r=0.985)

    x, y2 = Z[:, 0], Z[:, 1]

    # marker size from MAD (normalize)
    mad = np.clip(mad, 0.0, None)
    if mad.max() > 1e-12:
        s = 20.0 + 180.0 * (mad - mad.min()) / (mad.max() - mad.min() + 1e-12)
    else:
        s = np.full_like(mad, 40.0)

    # Plot
    fig = plt.figure(figsize=(8.5, 8.5))
    ax = plt.gca()

    # draw disk boundary
    theta = np.linspace(0, 2 * np.pi, 600)
    ax.plot(np.cos(theta), np.sin(theta), linewidth=2.0)

    # scatter per class for clean legend
    for k, name in enumerate(EMO_NAMES):
        m = (dom == k)
        if m.sum() == 0:
            continue
        ax.scatter(
            x[m], y2[m],
            s=s[m],
            alpha=0.75,
            edgecolors="none",
            label=name
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)

    ax.set_title(args.title, pad=14)
    ax.set_xlabel("Poincaré x (2D projection)", labelpad=10)
    ax.set_ylabel("Poincaré y (2D projection)", labelpad=10)

    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.legend(loc="upper right", frameon=True, fontsize=10)

    # small note about size encoding
    ax.text(
        -1.03, -1.12,
        "Marker size ∝ MAD ambiguity (higher = more ambiguous)",
        fontsize=10,
        ha="left",
        va="top"
    )

    plt.tight_layout()
    plt.savefig(args.out, dpi=250)
    plt.close(fig)

    print(f"[SAVED] {args.out}")


if __name__ == "__main__":
    main()
