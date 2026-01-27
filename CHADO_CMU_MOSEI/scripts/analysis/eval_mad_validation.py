import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from src.chado.metrics import compute_acc_wf1
from src.chado.mad import compute_mad_scores
from src.train.train import collect_logits_and_labels, build_model
from src.datasets.mosei_dataset import MoseiCSDDataset, mosei_collate_fn
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    ckpt = "outputs/checkpoints/TAV_best.pt"

    # Load model
    from src.chado.config import load_yaml
    cfg = load_yaml("src/configs/chado_mosei_emo6.yaml")
    model = build_model(cfg, "TAV").to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    # Data
    test_ds = MoseiCSDDataset(
        f"{cfg['experiment']['project_root']}/data/manifests/mosei_test.jsonl",
        ablation="TAV"
    )
    loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=mosei_collate_fn)

    logits, y = collect_logits_and_labels(model, loader, DEVICE)
    probs = torch.sigmoid(logits)

    # MAD
    mad_scores = compute_mad_scores(probs).cpu().numpy()

    preds = (probs > 0.5).int()
    true = (y > 0.5).int()

    errors = (preds != true).float().mean(dim=1).cpu().numpy()
    confidence = probs.max(dim=1).values.cpu().numpy()

    # Correlations
    pear_mad_err = pearsonr(mad_scores, errors)
    spear_mad_err = spearmanr(mad_scores, errors)

    print("MAD vs Error")
    print("Pearson:", pear_mad_err)
    print("Spearman:", spear_mad_err)

    # Plot
    plt.figure(figsize=(6,5))
    plt.scatter(mad_scores, errors, alpha=0.4)
    plt.xlabel("MAD Score")
    plt.ylabel("Prediction Error Rate")
    plt.title("MAD vs Prediction Error")
    plt.tight_layout()
    plt.savefig("outputs/analysis/plots/mad_vs_error.png")
    plt.close()

if __name__ == "__main__":
    main()

    # Save arrays for ICML-style plotting
    import os
    os.makedirs("outputs/analysis", exist_ok=True)
    np.savez("outputs/analysis/mad_error_data.npz", mad=mad_scores, error=errors)
    print("[SAVED] outputs/analysis/mad_error_data.npz")
