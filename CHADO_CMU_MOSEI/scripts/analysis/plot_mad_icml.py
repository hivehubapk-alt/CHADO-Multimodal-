import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from src.chado.config import load_yaml
from src.train.train import collect_logits_and_labels, build_model
from src.datasets.mosei_dataset import MoseiCSDDataset, mosei_collate_fn
from src.chado.mad import compute_mad_scores
from torch.utils.data import DataLoader

EMO_NAMES = ["happy", "sad", "angry", "fearful", "disgust", "surprise"]


@torch.no_grad()
def forward_logits(model, batch, device):
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    out = model(inputs)
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


@torch.no_grad()
def collect_logits_and_labels_single(model, loader, device):
    model.eval()
    all_logits, all_y = [], []
    for batch in loader:
        y = batch["label"].to(device)
        logits = forward_logits(model, batch, device)
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())
    logits = torch.cat(all_logits, dim=0) if all_logits else torch.zeros((0, 6))
    y = torch.cat(all_y, dim=0) if all_y else torch.zeros((0, 6))
    return logits, y


def fmt_p(p):
    # ICML-friendly compact p-value formatting
    if p < 1e-4:
        return "<1e-4"
    return f"={p:.4f}"


def main():
    cfg_path = "src/configs/chado_mosei_emo6.yaml"
    ablation = "TAV"
    ckpt_path = f"outputs/checkpoints/{ablation}_best.pt"
    out_dir = "outputs/analysis/plots"
    os.makedirs(out_dir, exist_ok=True)

    cfg = load_yaml(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Model ----
    model = build_model(cfg, ablation).to(device)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()

    # ---- Data ----
    proj = cfg["experiment"]["project_root"]
    test_manifest = f"{proj}/data/manifests/mosei_test.jsonl"
    test_ds = MoseiCSDDataset(test_manifest, ablation=ablation, label_thr=float(cfg["data"].get("label_thr", 0.0)))
    test_loader = DataLoader(
        test_ds,
        batch_size=64,
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        collate_fn=mosei_collate_fn,
        pin_memory=True,
    )

    # ---- Logits -> probs ----
    logits, y = collect_logits_and_labels_single(model, test_loader, device)
    probs = torch.sigmoid(logits)

    # ---- MAD and per-sample error ----
    mad = compute_mad_scores(probs, gamma=float(cfg.get("chado", {}).get("mad", {}).get("gamma", 1.0))).cpu().numpy()

    pred = (probs > 0.5).int()
    true = (y > 0.5).int()

    # Error rate per sample: fraction of labels wrong (0..1)
    err = (pred != true).float().mean(dim=1).cpu().numpy()

    # ---- Correlations ----
    pear = pearsonr(mad, err)
    spear = spearmanr(mad, err)

    print("MAD vs Error")
    print("Pearson:", pear)
    print("Spearman:", spear)

    # ---- ICML-quality plot ----
    # Use hexbin to avoid overplotting; ICML-friendly sizing
    fig = plt.figure(figsize=(6.2, 4.6))
    ax = fig.add_subplot(111)

    hb = ax.hexbin(
        mad, err,
        gridsize=35,
        mincnt=1
    )
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Count")

    # Trend line (least squares)
    if len(mad) >= 2:
        m, b = np.polyfit(mad, err, 1)
        xs = np.linspace(mad.min(), mad.max(), 200)
        ax.plot(xs, m * xs + b, linewidth=2, linestyle="-")

    ax.set_xlabel("MAD ambiguity score (per sample)")
    ax.set_ylabel("Prediction error rate (fraction of labels wrong)")
    ax.set_title("MAD correlates with error on CMU-MOSEI (Test)")

    # Annotation box (ICML-style)
    text = (
        f"Pearson r={pear.statistic:.3f}, p{fmt_p(pear.pvalue)}\n"
        f"Spearman ρ={spear.statistic:.3f}, p{fmt_p(spear.pvalue)}"
    )
    ax.text(
        0.02, 0.98, text,
        transform=ax.transAxes,
        va="top", ha="left",
        bbox=dict(boxstyle="round", alpha=0.85)
    )

    ax.grid(alpha=0.25)
    fig.tight_layout()

    out_base = os.path.join(out_dir, f"{ablation}_mad_vs_error_icml")
    fig.savefig(out_base + ".png", dpi=300)
    fig.savefig(out_base + ".pdf", dpi=300)
    plt.close(fig)

    print("[SAVED]", out_base + ".png")
    print("[SAVED]", out_base + ".pdf")


if __name__ == "__main__":
    main()
