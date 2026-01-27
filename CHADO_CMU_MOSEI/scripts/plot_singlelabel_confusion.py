import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from src.chado.config import load_yaml
from src.chado.trainer_utils import tune_thresholds
from src.chado.calibration import temperature_scale_logits
from src.datasets.mosei_dataset import MoseiCSDDataset, mosei_collate_fn
from src.models.baseline_fusion import BaselineFusion
from torch.utils.data import DataLoader

EMO_NAMES = ["happy", "sad", "angry", "fearful", "disgust", "surprise"]


@torch.no_grad()
def collect_logits_and_labels(model, loader, device):
    model.eval()
    L, Y = [], []
    for batch in loader:
        y = batch["label"].to(device)
        inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(inputs)
        if isinstance(out, (tuple, list)):
            logits = out[0]   # CHADO returns (logits, z)
        else:
            logits = out
        L.append(logits.detach().cpu())

        Y.append(y.detach().cpu())
    return torch.cat(L, 0), torch.cat(Y, 0)


def main():
    cfg_path = "src/configs/chado_mosei_emo6.yaml"
    ckpt_path = "outputs/checkpoints/TAV_best.pt"
    out_dir = "outputs/plots_paper"
    os.makedirs(out_dir, exist_ok=True)

    cfg = load_yaml(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model (same as training)
    mcfg = cfg["model"]
    model = BaselineFusion(
        num_classes=6,
        d_model=int(mcfg.get("d_model", 256)),
        use_audio=True,
        use_video=True,
        text_model=str(mcfg.get("text_model", "roberta-base")),
        max_text_len=int(cfg["data"].get("max_text_len", 96)),
        modality_dropout=float(mcfg.get("modality_dropout", 0.1)),
    ).to(device)

    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)

    # Load TEST set
    proj = cfg["experiment"]["project_root"]
    test_ds = MoseiCSDDataset(
        f"{proj}/data/manifests/mosei_test.jsonl",
        ablation="TAV",
        label_thr=float(cfg["data"].get("label_thr", 0.0)),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg["data"].get("batch_size", 16)),
        shuffle=False,
        collate_fn=mosei_collate_fn,
    )

    # Collect logits
    logits, y = collect_logits_and_labels(model, test_loader, device)

    # Temperature calibration (same as training)
    T = float(temperature_scale_logits(logits, y))
    probs = torch.sigmoid(logits / T)

    # ---- SINGLE-LABEL CONVERSION ----
    # True label = first active label (MOSEI standard)
    y_true = torch.argmax(y, dim=1).numpy()
    y_pred = torch.argmax(probs, dim=1).numpy()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot (same color style)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, cmap="viridis")
    plt.colorbar(im)

    plt.xticks(range(6), EMO_NAMES, rotation=45)
    plt.yticks(range(6), EMO_NAMES)

    for i in range(6):
        for j in range(6):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="white")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Test)")
    plt.tight_layout()

    out_png = f"{out_dir}/TAV_singlelabel_confusion.png"
    out_pdf = f"{out_dir}/TAV_singlelabel_confusion.pdf"
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf, dpi=300)
    plt.close()

    print("[SAVED]", out_png)
    print("[SAVED]", out_pdf)


if __name__ == "__main__":
    main()
