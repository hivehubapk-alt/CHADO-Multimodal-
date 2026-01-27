import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.chado.metrics import compute_acc_wf1
from src.chado.mad import compute_mad_scores
from src.train.train import collect_logits_and_labels, build_model
from src.datasets.mosei_dataset import MoseiCSDDataset, mosei_collate_fn
from torch.utils.data import DataLoader
from src.chado.config import load_yaml

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    cfg = load_yaml("src/configs/chado_mosei_emo6.yaml")
    model = build_model(cfg, "TAV").to(DEVICE)
    model.load_state_dict(torch.load("outputs/checkpoints/TAV_best.pt"))
    model.eval()

    ds = MoseiCSDDataset(
        f"{cfg['experiment']['project_root']}/data/manifests/mosei_test.jsonl",
        ablation="TAV"
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=mosei_collate_fn)

    logits, y = collect_logits_and_labels(model, loader, DEVICE)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).int()
    true = (y > 0.5).int()

    mad = compute_mad_scores(probs).cpu().numpy()

    q1, q2 = np.quantile(mad, [0.33, 0.66])
    strata = {
        "Low": mad <= q1,
        "Medium": (mad > q1) & (mad <= q2),
        "High": mad > q2
    }

    results = []

    for k, mask in strata.items():
        acc, wf1, _ = compute_acc_wf1(true[mask], preds[mask])
        results.append([k, acc*100, wf1*100])

    df = pd.DataFrame(results, columns=["Ambiguity", "Accuracy", "WF1"])
    df.to_csv("outputs/analysis/tables/ambiguity_strata.csv", index=False)

    df.plot(x="Ambiguity", y=["Accuracy", "WF1"], kind="bar", figsize=(6,4))
    plt.title("Performance vs Ambiguity Level")
    plt.tight_layout()
    plt.savefig("outputs/analysis/plots/ambiguity_strata.png")

if __name__ == "__main__":
    main()
