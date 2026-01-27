import os
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.chado.config import load_yaml
from src.train.train import build_model
from src.datasets.mosei_dataset import MoseiCSDDataset, mosei_collate_fn

ABLATIONS = ["T", "TA", "TV", "TAV"]


@torch.no_grad()
def forward_logits(model, batch, device):
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    out = model(inputs)
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def _annotate_bars(ax, xs, ys):
    # annotate bars with value text
    for x, y in zip(xs, ys):
        if np.isnan(y):
            continue
        ax.text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=10)


def plot_bar_annotated(df, col, out_base, title, ylabel):
    fig = plt.figure(figsize=(6.8, 4.2))
    ax = fig.add_subplot(111)

    x = np.arange(len(df))
    y = df[col].values.astype(float)

    ax.bar(x, y)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Ablation"].values)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)

    _annotate_bars(ax, x, y)

    fig.tight_layout()
    fig.savefig(out_base + ".png", dpi=300)
    fig.savefig(out_base + ".pdf", dpi=300)
    plt.close(fig)


def plot_barh_params(df, out_base):
    fig = plt.figure(figsize=(7.0, 4.2))
    ax = fig.add_subplot(111)

    ylabels = df["Ablation"].values
    xvals = df["Params_M"].values.astype(float)
    y = np.arange(len(ylabels))

    ax.barh(y, xvals)
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Millions of parameters")
    ax.set_title("Parameter Count Comparison")
    ax.grid(axis="x", alpha=0.25)

    for yi, xv in zip(y, xvals):
        if np.isnan(xv):
            continue
        ax.text(xv, yi, f" {xv:.2f}M", va="center", ha="left", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_base + ".png", dpi=300)
    fig.savefig(out_base + ".pdf", dpi=300)
    plt.close(fig)


def plot_line_if_possible(df, col, out_base, title, ylabel):
    # only meaningful if >=2 points
    if len(df) < 2:
        return False

    fig = plt.figure(figsize=(6.8, 4.2))
    ax = fig.add_subplot(111)

    x = np.arange(len(df))
    y = df[col].values.astype(float)

    ax.plot(x, y, marker="o", linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Ablation"].values)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)

    for xi, yi in zip(x, y):
        if np.isnan(yi):
            continue
        ax.text(xi, yi, f"{yi:.2f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_base + ".png", dpi=300)
    fig.savefig(out_base + ".pdf", dpi=300)
    plt.close(fig)
    return True


def print_table(df: pd.DataFrame):
    # Print an ICML-friendly console table
    def fmt(x, digits=2):
        if pd.isna(x):
            return "NA"
        return f"{x:.{digits}f}"

    print("\n=== Computational Cost Comparison (Test Inference) ===")
    cols = ["Ablation", "Params_M", "Infer_ms_per_sample", "PeakMem_MB", "Checkpoint"]
    d = df[cols].copy()

    # nicer formatting
    d["Params_M"] = d["Params_M"].apply(lambda v: fmt(v, 2))
    d["Infer_ms_per_sample"] = d["Infer_ms_per_sample"].apply(lambda v: fmt(v, 3))
    d["PeakMem_MB"] = d["PeakMem_MB"].apply(lambda v: fmt(v, 1))

    # column widths
    wA = max(len("Ablation"), d["Ablation"].astype(str).map(len).max())
    wP = max(len("Params(M)"), d["Params_M"].astype(str).map(len).max())
    wT = max(len("Infer(ms/sample)"), d["Infer_ms_per_sample"].astype(str).map(len).max())
    wM = max(len("PeakMem(MB)"), d["PeakMem_MB"].astype(str).map(len).max())

    header = f"{'Ablation':<{wA}}  {'Params(M)':>{wP}}  {'Infer(ms/sample)':>{wT}}  {'PeakMem(MB)':>{wM}}"
    print(header)
    print("-" * len(header))

    for _, r in d.iterrows():
        print(
            f"{str(r['Ablation']):<{wA}}  "
            f"{str(r['Params_M']):>{wP}}  "
            f"{str(r['Infer_ms_per_sample']):>{wT}}  "
            f"{str(r['PeakMem_MB']):>{wM}}"
        )

    # quick note about missing checkpoints
    missing = df[df["Checkpoint"] == "MISSING"]["Ablation"].tolist()
    if missing:
        print("\n[NOTE] Missing checkpoints for:", ", ".join(missing))
        print("       Train these ablations to get a full comparison.")


def main():
    cfg_path = "src/configs/chado_mosei_emo6.yaml"
    cfg = load_yaml(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("outputs/analysis/tables", exist_ok=True)
    os.makedirs("outputs/analysis/plots", exist_ok=True)

    out_table = "outputs/analysis/tables/compute_cost_comparison.csv"
    out_time_bar = "outputs/analysis/plots/compute_inference_time_bar"
    out_time_line = "outputs/analysis/plots/compute_inference_time_line"
    out_mem_bar = "outputs/analysis/plots/compute_peak_memory_bar"
    out_mem_line = "outputs/analysis/plots/compute_peak_memory_line"
    out_params_bar = "outputs/analysis/plots/compute_params_bar"
    out_params_barh = "outputs/analysis/plots/compute_params_barh"
    out_params_line = "outputs/analysis/plots/compute_params_line"

    proj = cfg["experiment"]["project_root"]
    test_manifest = f"{proj}/data/manifests/mosei_test.jsonl"

    label_thr = float(cfg["data"].get("label_thr", 0.0))
    batch_size = int(cfg["data"].get("batch_size", 32))
    num_workers = int(cfg["data"].get("num_workers", 4))

    rows = []
    for ab in ABLATIONS:
        ckpt = f"outputs/checkpoints/{ab}_best.pt"
        if not os.path.exists(ckpt):
            rows.append({
                "Ablation": ab,
                "Checkpoint": "MISSING",
                "Params_M": np.nan,
                "Infer_ms_per_sample": np.nan,
                "PeakMem_MB": np.nan,
            })
            continue

        ds = MoseiCSDDataset(test_manifest, ablation=ab, label_thr=label_thr)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=mosei_collate_fn,
            pin_memory=True
        )

        model = build_model(cfg, ab).to(device)
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state, strict=False)
        model.eval()

        params_m = count_params(model) / 1e6

        # inference timing
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # warmup a few steps for more stable timing (optional)
        warm = 0
        for batch in loader:
            _ = forward_logits(model, batch, device)
            warm += batch["label"].shape[0]
            if warm >= min(4 * batch_size, 128):
                break
        if device.type == "cuda":
            torch.cuda.synchronize()

        n_samples = 0
        t0 = time.perf_counter()
        for batch in loader:
            _ = forward_logits(model, batch, device)
            n_samples += batch["label"].shape[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        ms_per_sample = (t1 - t0) * 1000.0 / max(1, n_samples)
        peak_mb = np.nan
        if device.type == "cuda":
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        rows.append({
            "Ablation": ab,
            "Checkpoint": ckpt,
            "Params_M": float(params_m),
            "Infer_ms_per_sample": float(ms_per_sample),
            "PeakMem_MB": float(peak_mb),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_table, index=False)

    # print on screen
    print_table(df)

    # plotting only for available entries
    df_plot = df.dropna(subset=["Infer_ms_per_sample", "PeakMem_MB", "Params_M"]).copy()
    if len(df_plot) == 0:
        print("\n[WARN] No checkpoints found to plot (need outputs/checkpoints/{T,TA,TV,TAV}_best.pt).")
        print("[SAVED]", out_table)
        return

    # bars with numeric labels (more readable)
    plot_bar_annotated(df_plot, "Infer_ms_per_sample", out_time_bar, "Inference Time Comparison (Test)", "ms / sample")
    plot_bar_annotated(df_plot, "PeakMem_MB", out_mem_bar, "Peak GPU Memory (Inference)", "MB")
    plot_bar_annotated(df_plot, "Params_M", out_params_bar, "Parameter Count (Millions)", "Millions of params")

    # best parameter readability
    plot_barh_params(df_plot, out_params_barh)

    # line plots (only if multiple ablations exist)
    if plot_line_if_possible(df_plot, "Infer_ms_per_sample", out_time_line, "Inference Time (Line)", "ms / sample"):
        pass
    if plot_line_if_possible(df_plot, "PeakMem_MB", out_mem_line, "Peak GPU Memory (Line)", "MB"):
        pass
    if plot_line_if_possible(df_plot, "Params_M", out_params_line, "Parameter Count (Line)", "Millions of params"):
        pass

    print("\n[SAVED]", out_table)
    print("[SAVED]", out_time_bar + ".png/.pdf")
    print("[SAVED]", out_mem_bar + ".png/.pdf")
    print("[SAVED]", out_params_bar + ".png/.pdf")
    print("[SAVED]", out_params_barh + ".png/.pdf")
    if len(df_plot) >= 2:
        print("[SAVED]", out_time_line + ".png/.pdf")
        print("[SAVED]", out_mem_line + ".png/.pdf")
        print("[SAVED]", out_params_line + ".png/.pdf")


if __name__ == "__main__":
    main()
