#!/usr/bin/env python3
import os
import time
import yaml
import math
import json
import argparse
import numpy as np
import torch

from torch.utils.data import DataLoader

from src.data.meld_dataset import (
    MeldDataset,
    collate_meld,
    build_label_map_from_order,
    EMO_ORDER_7,
)
from src.models.chado_trimodal import CHADOTrimodal


def _safe_bool(x):
    return bool(x) if isinstance(x, bool) else str(x).lower() in ("1", "true", "yes", "y")


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def _extract_state_dict(ckpt_obj):
    # supports raw state_dict or {"state_dict": ...}
    if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
        return ckpt_obj["state_dict"]
    if isinstance(ckpt_obj, dict):
        # could already be state_dict
        return ckpt_obj
    raise TypeError("Checkpoint format not recognized.")


def load_baseline_into_chado_base(model: CHADOTrimodal, baseline_ckpt_path: str, verbose: bool = True):
    ckpt = torch.load(baseline_ckpt_path, map_location="cpu")
    sd = _extract_state_dict(ckpt)

    # strip module. prefix if present
    cleaned = {}
    for k, v in sd.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        cleaned[nk] = v

    # CHADO keys are base.* ; baseline keys are typically text_encoder.*, audio_encoder.*, video_encoder.*, classifier.*
    chado_sd = model.state_dict()
    mapped = {}
    matched = 0

    for ck, cv in cleaned.items():
        # direct match? (rare)
        if ck in chado_sd and chado_sd[ck].shape == cv.shape:
            mapped[ck] = cv
            matched += 1
            continue

        # try prefix into base.*
        bkey = "base." + ck
        if bkey in chado_sd and chado_sd[bkey].shape == cv.shape:
            mapped[bkey] = cv
            matched += 1

    missing, unexpected = model.load_state_dict(mapped, strict=False)

    if verbose:
        print(f"[LOAD] baseline ckpt: {baseline_ckpt_path}")
        print(f"[LOAD] extracted state_dict keys: {len(cleaned)}")
        sample = list(cleaned.keys())[:15]
        print(f"[LOAD] sample ckpt keys: {sample}")
        print(f"[LOAD] matched keys: {matched}")
        print(f"[LOAD] missing keys: {len(missing)} (expected CHADO heads)")
        print(f"[LOAD] unexpected keys: {len(unexpected)}")

    return matched


@torch.no_grad()
def infer_and_metrics(
    model,
    loader,
    device,
    amp: bool,
    subset_mask_amb: torch.Tensor | None,
    subset_mask_xc: torch.Tensor | None,
    mad_quantile: float | None,
):
    model.eval()

    all_logits = []
    all_labels = []

    # Measure inference time (exclude warmup)
    timings = []
    warmup = 5
    step = 0

    for batch in loader:
        labels = batch.labels.to(device)

        text = {k: v.to(device) for k, v in batch.text_input.items()} if batch.text_input else None
        audio = batch.audio_wave.to(device) if batch.audio_wave is not None else None
        video = batch.video_frames.to(device) if batch.video_frames is not None else None

        torch.cuda.synchronize()
        t0 = time.time()
        with torch.amp.autocast("cuda", enabled=amp):
            logits, _, _ = model(text_input=text, audio_wave=audio, video_frames=video, modality_mask=None)
        torch.cuda.synchronize()
        t1 = time.time()

        if step >= warmup:
            timings.append(t1 - t0)
        step += 1

        all_logits.append(logits.detach().float().cpu())
        all_labels.append(labels.detach().cpu())

    logits = torch.cat(all_logits, dim=0)  # [N,C]
    labels = torch.cat(all_labels, dim=0)  # [N]

    preds = torch.argmax(logits, dim=-1)

    # overall acc
    overall_acc = float((preds == labels).float().mean().item())

    # weighted F1
    num_classes = logits.shape[-1]
    tp = torch.zeros(num_classes)
    fp = torch.zeros(num_classes)
    fn = torch.zeros(num_classes)
    for c in range(num_classes):
        tp[c] = ((preds == c) & (labels == c)).sum()
        fp[c] = ((preds == c) & (labels != c)).sum()
        fn[c] = ((preds != c) & (labels == c)).sum()
    support = tp + fn
    precision = tp / torch.clamp(tp + fp, min=1.0)
    recall = tp / torch.clamp(tp + fn, min=1.0)
    f1 = 2 * precision * recall / torch.clamp(precision + recall, min=1e-8)
    f1w = float((f1 * support).sum().item() / max(1.0, float(support.sum().item())))

    # ambiguous subset
    amb_acc = None
    if subset_mask_amb is not None:
        m = subset_mask_amb.bool()
        if m.any():
            amb_acc = float((preds[m] == labels[m]).float().mean().item())

    # if ambiguous column not present, derive via MAD quantile
    if amb_acc is None and mad_quantile is not None:
        # MAD proxy: 1 - max softmax prob (higher => more ambiguous)
        probs = torch.softmax(logits, dim=-1)
        mad = 1.0 - probs.max(dim=-1).values
        thr = torch.quantile(mad, 1.0 - mad_quantile)  # top quantile
        m = mad >= thr
        if m.any():
            amb_acc = float((preds[m] == labels[m]).float().mean().item())

    # cross-cultural subset
    xc_acc = None
    if subset_mask_xc is not None:
        m = subset_mask_xc.bool()
        if m.any():
            xc_acc = float((preds[m] == labels[m]).float().mean().item())

    # inference time
    # report mean ms per batch; also convert to ms per sample
    if len(timings) == 0:
        inf_ms = float("nan")
    else:
        mean_s = float(np.mean(timings))
        # convert to ms per sample (approx)
        bs = loader.batch_size if loader.batch_size else 1
        inf_ms = (mean_s / max(1, bs)) * 1000.0

    return {
        "overall_acc": overall_acc,
        "overall_f1w": f1w,
        "amb_acc": amb_acc,
        "xc_acc": xc_acc,
        "inf_ms": inf_ms,
    }


import inspect

def build_dataset(cfg, csv_path, label_map):
    """
    Build MeldDataset safely without double-passing label_map.
    Compatible with all existing dataset signatures.
    """
    sig = inspect.signature(MeldDataset.__init__)
    accepted = set(sig.parameters.keys())

    candidates = {
        "csv_path": csv_path,
        "text_col": cfg["data"]["text_col"],
        "label_col": cfg["data"]["label_col"],
        "audio_path_col": cfg["data"]["audio_path_col"],
        "video_path_col": cfg["data"]["video_path_col"],
        "utt_id_col": cfg["data"]["utt_id_col"],
        "num_frames": cfg["data"]["num_frames"],
        "frame_size": cfg["data"]["frame_size"],
        "sample_rate": cfg["data"]["sample_rate"],
        "max_audio_seconds": cfg["data"]["max_audio_seconds"],
        "label_map": label_map,
        "use_text": cfg["model"]["use_text"],
        "use_audio": cfg["model"]["use_audio"],
        "use_video": cfg["model"]["use_video"],
        "text_model_name": cfg["model"].get("text_model_name", None),
    }

    kwargs = {k: v for k, v in candidates.items() if k in accepted and v is not None}

    ds = MeldDataset(**kwargs)

    if not hasattr(ds, "tokenizer"):
        raise RuntimeError("MeldDataset must expose tokenizer for collate_meld")

    return ds


def try_build_subset_masks(ds, subset_cols: dict):
    """
    If dataset exposes a dataframe or underlying metadata, try to build masks.
    This is repo-dependent; we handle multiple common patterns.
    """
    amb_col = subset_cols.get("ambiguous_col", None)
    xc_col = subset_cols.get("cross_cultural_col", None)

    amb_mask = None
    xc_mask = None

    # Common patterns: ds.df, ds.dataframe, ds.data
    df = None
    for attr in ("df", "dataframe", "data"):
        if hasattr(ds, attr):
            obj = getattr(ds, attr)
            # likely pandas df
            if hasattr(obj, "columns"):
                df = obj
                break

    if df is not None:
        if amb_col and amb_col in df.columns:
            amb_mask = torch.tensor(df[amb_col].astype(int).values)
        if xc_col and xc_col in df.columns:
            xc_mask = torch.tensor(df[xc_col].astype(int).values)

    return amb_mask, xc_mask


def mean_std(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if len(xs) == 0:
        return None, None
    m = float(np.mean(xs))
    s = float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
    return m, s


def format_pm(m, s, scale=100.0, decimals=1):
    if m is None:
        return "N/A"
    if s is None:
        return f"{m*scale:.{decimals}f}"
    return f"{m*scale:.{decimals}f} ± {s*scale:.{decimals}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/chado_component_ablations.yaml")
    ap.add_argument("--gpu", default="0", help="single GPU id for evaluation (e.g., 0 or 5).")
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    base_cfg = _load_yaml(cfg["base_config"])

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    out_dir = _ensure_dir(cfg["output"]["out_dir"])
    csv_path = os.path.join(out_dir, cfg["output"]["csv_name"])
    tex_path = os.path.join(out_dir, cfg["output"]["latex_name"])

    device = torch.device(cfg.get("device", "cuda"))
    torch.backends.cudnn.benchmark = True

    label_map = build_label_map_from_order(EMO_ORDER_7)

    # Build test dataset once (same ordering across runs)
    test_ds = build_dataset(base_cfg, base_cfg["data"]["test_csv"], label_map)
    amb_mask, xc_mask = try_build_subset_masks(test_ds, cfg.get("subset_columns", {}))

    loader = DataLoader(
        test_ds,
        batch_size=int(cfg.get("batch_size_eval", 12)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 6)),
        pin_memory=True,
        drop_last=False,
        collate_fn=lambda b: collate_meld(
            b, test_ds.tokenizer,
            base_cfg["model"]["use_text"], base_cfg["model"]["use_audio"], base_cfg["model"]["use_video"]
        ),
    )

    seeds = cfg["seeds"]
    mad_cfg = cfg.get("mad_ambiguous", {})
    mad_enabled = _safe_bool(mad_cfg.get("enabled", True))
    mad_q = float(mad_cfg.get("quantile", 0.20)) if mad_enabled else None

    results_rows = []

    for ab in cfg["ablations"]:
        name = ab["name"]
        setting = ab["chado"]

        overall_accs, f1ws, amb_accs, xc_accs, inf_mss = [], [], [], [], []

        for seed in seeds:
            # set seed for reproducibility
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))

            model = CHADOTrimodal(
                text_model_name=base_cfg["model"]["text_model_name"],
                audio_model_name=base_cfg["model"]["audio_model_name"],
                video_model_name=base_cfg["model"]["video_model_name"],
                num_classes=base_cfg["data"]["num_classes"],
                proj_dim=base_cfg["model"]["proj_dim"],
                dropout=base_cfg["model"]["dropout"],
                use_text=base_cfg["model"]["use_text"],
                use_audio=base_cfg["model"]["use_audio"],
                use_video=base_cfg["model"]["use_video"],
                use_gated_fusion=base_cfg["model"]["use_gated_fusion"],
                use_causal=bool(setting["use_causal"]),
                use_hyperbolic=bool(setting["use_hyperbolic"]),
                use_transport=bool(setting["use_transport"]),
                use_refinement=bool(setting["use_refinement"]),
            ).to(device)

            # Load baseline into base.* to preserve your 63% performance.
            load_baseline_into_chado_base(model, cfg["baseline_ckpt"], verbose=(seed == seeds[0]))

            # Evaluate
            metrics = infer_and_metrics(
                model=model,
                loader=loader,
                device=device,
                amp=_safe_bool(cfg.get("amp", True)),
                subset_mask_amb=amb_mask,
                subset_mask_xc=xc_mask,
                mad_quantile=mad_q,
            )

            overall_accs.append(metrics["overall_acc"])
            f1ws.append(metrics["overall_f1w"])
            amb_accs.append(metrics["amb_acc"])
            xc_accs.append(metrics["xc_acc"])
            inf_mss.append(metrics["inf_ms"])

            amb_str = "N/A" if metrics["amb_acc"] is None else f"{metrics['amb_acc']:.4f}"
            xc_str  = "N/A" if metrics["xc_acc"] is None else f"{metrics['xc_acc']:.4f}"
            inf_str = "N/A" if metrics["inf_ms"] is None else f"{metrics['inf_ms']:.2f}"

            print(
                f"[{name} seed={seed}] "
                f"acc={metrics['overall_acc']:.4f} "
                f"f1w={metrics['overall_f1w']:.4f} "
                f"amb={amb_str} "
                f"xc={xc_str} "
                f"inf_ms={inf_str}"
            )


        m_acc, s_acc = mean_std(overall_accs)
        m_f1, s_f1 = mean_std(f1ws)
        m_amb, s_amb = mean_std(amb_accs)
        m_xc, s_xc = mean_std(xc_accs)
        m_t, s_t = mean_std(inf_mss)

        row = {
            "name": name,
            "overall_acc_mean": m_acc, "overall_acc_std": s_acc,
            "overall_f1w_mean": m_f1, "overall_f1w_std": s_f1,
            "amb_acc_mean": m_amb, "amb_acc_std": s_amb,
            "xc_acc_mean": m_xc, "xc_acc_std": s_xc,
            "inf_ms_mean": m_t, "inf_ms_std": s_t,
        }
        results_rows.append(row)

    # Write CSV
    import csv
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results_rows[0].keys()))
        w.writeheader()
        for r in results_rows:
            w.writerow(r)
    print(f"[OK] Wrote CSV: {csv_path}")

    # Write LaTeX table (matches your figure columns)
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{CHADO component ablations on MELD (mean $\pm$ std over seeds).}")
    lines.append(r"\label{tab:chado_components}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Components Used & Overall Acc. & Ambiguous Cases & Cross-Cultural & Inference Time \\")
    lines.append(r"\midrule")

    def pretty_name(n):
        return {
            "causal_only": "Causal only",
            "hyperbolic_only": "Hyperbolic only",
            "transport_only": "Transport only",
            "causal_plus_hyperbolic": "Causal + Hyperbolic",
            "causal_plus_transport": "Causal + Transport",
            "hyperbolic_plus_transport": "Hyperbolic + Transport",
            "all_chado": r"\textbf{All (CHADO)}",
        }.get(n, n)

    for r in results_rows:
        name = pretty_name(r["name"])
        oa = format_pm(r["overall_acc_mean"], r["overall_acc_std"], scale=100.0, decimals=1)
        amb = format_pm(r["amb_acc_mean"], r["amb_acc_std"], scale=100.0, decimals=1)
        xc = format_pm(r["xc_acc_mean"], r["xc_acc_std"], scale=100.0, decimals=1)
        # inference time in ms already
        if r["inf_ms_mean"] is None or (isinstance(r["inf_ms_mean"], float) and math.isnan(r["inf_ms_mean"])):
            it = "N/A"
        else:
            it = f"{r['inf_ms_mean']:.0f} ms"

        # bold All (CHADO) row
        if r["name"] == "all_chado":
            oa = r"\textbf{" + oa + "}"
            amb = r"\textbf{" + amb + "}" if amb != "N/A" else amb
            xc = r"\textbf{" + xc + "}" if xc != "N/A" else xc
            it = r"\textbf{" + it + "}" if it != "N/A" else it

        lines.append(f"{name} & {oa} & {amb} & {xc} & {it} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[OK] Wrote LaTeX: {tex_path}")


if __name__ == "__main__":
    main()
