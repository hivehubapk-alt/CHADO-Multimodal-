import time
import torch
import pandas as pd

from src.train.train import build_model
from src.chado.config import load_yaml

DEVICE = "cuda"

def measure(ablation):
    cfg = load_yaml("src/configs/chado_mosei_emo6.yaml")
    model = build_model(cfg, ablation).to(DEVICE)
    dummy = {
        "text_ids": torch.randint(0, 100, (1, 64)).to(DEVICE),
        "audio": torch.randn(1, 400, 74).to(DEVICE),
        "video": torch.randn(1, 32, 713).to(DEVICE),
    }

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        _ = model(dummy)
    torch.cuda.synchronize()

    return (time.time() - t0) / 100 * 1000

def main():
    rows = []
    for ab in ["T", "TA", "TV", "TAV"]:
        rows.append([ab, measure(ab)])

    df = pd.DataFrame(rows, columns=["Model", "Inference_ms"])
    df.to_csv("outputs/analysis/tables/compute_cost.csv", index=False)

if __name__ == "__main__":
    main()
