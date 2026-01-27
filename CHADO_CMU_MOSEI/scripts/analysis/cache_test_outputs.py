import os
import argparse
from typing import Dict, Any, Tuple, List

import torch

from src.chado.config import load_yaml
from src.chado.calibration import temperature_scale_logits
from src.chado.trainer_utils import tune_thresholds, apply_thresholds
from src.datasets.mosei_dataset import MoseiCSDDataset, mosei_collate_fn
from src.chado.mad import compute_mad_scores

# reuse the same model builder from your trainer
from src.train.train import build_model


@torch.no_grad()
def collect_logits_labels_embeddings(
    model: torch.nn.Module,
    loader,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collects:
      - logits: [N, C]
      - labels: [N, C]
      - embeddings: [N, D] (latent z returned by your model/forward_model)

    IMPORTANT:
    - Your BaselineFusion in train.py returns either:
        logits
      or:
        (logits, z)
    - We handle both safely.

    This is SINGLE-GPU inference (no DDP needed for plotting).
    """
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_y: List[torch.Tensor] = []
    all_z: List[torch.Tensor] = []

    for batch in loader:
        y = batch["label"].to(device)

        # move tensors to device for forward
        inputs: Dict[str, Any] = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(device)
            else:
                inputs[k] = v

        out = model(inputs)

        if isinstance(out, (tuple, list)) and len(out) == 2:
            logits, z = out
        else:
            logits = out
            # if model does not provide embeddings, fall back to logits as a proxy
            # (Poincaré plot will still run, but it is less meaningful)
            z = logits

        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())
        all_z.append(z.detach().cpu())

    logits = torch.cat(all_logits, dim=0) if all_logits else torch.zeros((0, 6))
    labels = torch.cat(all_y, dim=0) if all_y else torch.zeros((0, 6))
    emb = torch.cat(all_z, dim=0) if all_z else torch.zeros((0, 6))

    return logits, labels, emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, type=str)
    ap.add_argument("--ablation", default="TAV", type=str, choices=["T", "TA", "TV", "TAV"])
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--out_dir", default="outputs/cache", type=str)
    ap.add_argument("--batch_size", default=64, type=int)
    ap.add_argument("--num_workers", default=4, type=int)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cfg = load_yaml(args.cfg)
    proj = cfg["experiment"]["project_root"]
    test_manifest = f"{proj}/data/manifests/mosei_test.jsonl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg, args.ablation).to(device)

    state = torch.load(args.ckpt, map_location="cpu")
    # support both raw state_dict and wrapped dicts
    if isinstance(state, dict) and any(k.startswith("text_enc") or k.startswith("classifier") for k in state.keys()):
        model.load_state_dict(state, strict=False)
    elif isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"], strict=False)
    else:
        model.load_state_dict(state, strict=False)

    model.eval()

    test_ds = MoseiCSDDataset(test_manifest, ablation=args.ablation)
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=mosei_collate_fn
    )

    # Collect logits + labels + embeddings
    logits, y, emb = collect_logits_labels_embeddings(model, test_loader, device)

    # Calibration + thresholds (same logic as your train.py)
    T = float(temperature_scale_logits(logits, y))
    probs = torch.sigmoid(logits / T)

    thr_grid = cfg["data"].get("pred_thr_grid", [0.1, 0.2, 0.3, 0.4, 0.5])
    thr = tune_thresholds(logits / T, y, grid=thr_grid, objective="acc")

    preds = apply_thresholds(probs, thr)
    errors = (preds.int() != (y > 0.5).int()).any(dim=1).int()

    mad = compute_mad_scores(probs)

    # Save cached artifacts
    torch.save(probs.cpu(), os.path.join(args.out_dir, "test_probs.pt"))
    torch.save(y.cpu(), os.path.join(args.out_dir, "test_labels.pt"))
    torch.save(errors.cpu(), os.path.join(args.out_dir, "test_errors.pt"))
    torch.save(mad.cpu(), os.path.join(args.out_dir, "test_mad.pt"))
    torch.save(emb.cpu(), os.path.join(args.out_dir, "test_embeddings.pt"))

    print("[OK] Cached:")
    print(" - test_probs.pt")
    print(" - test_labels.pt")
    print(" - test_errors.pt")
    print(" - test_mad.pt")
    print(" - test_embeddings.pt")
    print(f"Calibration T = {T:.3f}")
    print(f"Embeddings shape = {tuple(emb.shape)}")


if __name__ == "__main__":
    main()
