import os
import yaml
import inspect
import random
import textwrap
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from src.data.meld_dataset import (
    MeldDataset,
    collate_meld,
    EMO_ORDER_7,
    build_label_map_from_order,
)
from src.models.chado_trimodal import CHADOTrimodal


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for k in ["state_dict", "model", "model_state_dict", "net"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                ckpt_obj = ckpt_obj[k]
                break
    if not isinstance(ckpt_obj, dict):
        raise RuntimeError("Checkpoint is not a dict and has no usable state_dict-like content.")

    cleaned = {}
    for kk, vv in ckpt_obj.items():
        nk = kk[7:] if kk.startswith("module.") else kk
        cleaned[nk] = vv
    return cleaned


def build_meld_dataset(cfg, split_csv, label_map):
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    # MELD header-safe defaults
    text_col = data_cfg.get("text_col", "text")
    label_col = data_cfg.get("label_col", "emotion")  # IMPORTANT for your CSV
    audio_path_col = data_cfg.get("audio_path_col", "audio_path")
    video_path_col = data_cfg.get("video_path_col", "video_path")
    utt_id_col = data_cfg.get("utt_id_col", "utt_id")

    sig = inspect.signature(MeldDataset.__init__)
    accepted = set(sig.parameters.keys())

    candidates = {
        "csv_path": split_csv,
        "text_col": text_col,
        "label_col": label_col,
        "audio_path_col": audio_path_col,
        "video_path_col": video_path_col,
        "utt_id_col": utt_id_col,
        "num_frames": data_cfg.get("num_frames", 8),
        "frame_size": data_cfg.get("frame_size", 224),
        "sample_rate": data_cfg.get("sample_rate", 16000),
        "max_audio_seconds": data_cfg.get("max_audio_seconds", 6.0),
        "label_map": label_map,
        "use_text": bool(model_cfg.get("use_text", True)),
        "use_audio": bool(model_cfg.get("use_audio", True)),
        "use_video": bool(model_cfg.get("use_video", True)),
        "text_model_name": model_cfg.get("text_model_name", "roberta-base"),
        # tolerate alternate repos’ arg names
        "hf_model_name": model_cfg.get("text_model_name", "roberta-base"),
        "bert_name": model_cfg.get("text_model_name", "roberta-base"),
    }

    kwargs = {k: v for k, v in candidates.items() if k in accepted and v is not None}

    required = [
        p.name for p in sig.parameters.values()
        if p.name != "self"
        and p.default is inspect._empty
        and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    missing = [r for r in required if r not in kwargs]
    if missing:
        raise TypeError(f"MeldDataset missing required args: {missing}\nSignature: {sig}")

    ds = MeldDataset(**kwargs)
    return ds


@torch.no_grad()
def infer_logits(model, device, batch, use_text, use_audio, use_video):
    labels = batch.labels.to(device)
    text = {k: v.to(device) for k, v in batch.text_input.items()} if (use_text and batch.text_input) else None
    audio = batch.audio_wave.to(device) if (use_audio and batch.audio_wave is not None) else None
    video = batch.video_frames.to(device) if (use_video and batch.video_frames is not None) else None

    logits, _, _ = model(text_input=text, audio_wave=audio, video_frames=video, modality_mask=None)
    preds = torch.argmax(logits, dim=-1)
    return preds.detach().cpu().numpy(), labels.detach().cpu().numpy()


def wrap(s, width):
    if s is None:
        return ""
    return "\n".join(textwrap.wrap(str(s), width=width, break_long_words=False, replace_whitespace=False))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_scan_batches", type=int, default=500)
    ap.add_argument("--pick_per_group", type=int, default=2)
    ap.add_argument("--max_text_width", type=int, default=70)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load(open(args.config))

    label_map = build_label_map_from_order(EMO_ORDER_7)
    inv_label = {v: k for k, v in label_map.items()}  # e.g., 0->"neutral"
    # nicer display order
    id2name = {i: EMO_ORDER_7[i] for i in range(len(EMO_ORDER_7))}

    # choose csv
    if args.split == "train":
        csv_path = cfg["data"]["train_csv"]
    elif args.split == "val":
        csv_path = cfg["data"]["val_csv"]
    else:
        csv_path = cfg["data"]["test_csv"]

    # Build dataset in tri-modal mode so collate has everything available
    ds = build_meld_dataset(cfg, csv_path, label_map)

    # We will always collate with tokenizer available (tri) for text rendering and text-only forward.
    use_text_tri = True
    use_audio_tri = True
    use_video_tri = True

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda b: collate_meld(b, ds.tokenizer, use_text_tri, use_audio_tri, use_video_tri),
    )

    # --- Build FOUR models (this is the key fix) ---
    def make_model(use_text, use_audio, use_video):
        m = CHADOTrimodal(
            text_model_name=cfg["model"]["text_model_name"],
            audio_model_name=cfg["model"]["audio_model_name"],
            video_model_name=cfg["model"]["video_model_name"],
            num_classes=cfg["data"]["num_classes"],
            proj_dim=cfg["model"]["proj_dim"],
            dropout=cfg["model"]["dropout"],
            use_text=use_text,
            use_audio=use_audio,
            use_video=use_video,
            use_gated_fusion=cfg["model"]["use_gated_fusion"],
            use_causal=False,
            use_hyperbolic=False,
            use_transport=False,
            use_refinement=False,
        ).to(device)
        return m

    m_tri = make_model(True, True, True)
    m_t = make_model(True, False, False)
    m_a = make_model(False, True, False)
    m_v = make_model(False, False, True)

    print(f"[LOAD] ckpt={args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = _extract_state_dict(ckpt)

    # strict=False is fine; keys not used in modality-off configs will be ignored.
    for name, m in [("tri", m_tri), ("text", m_t), ("audio", m_a), ("video", m_v)]:
        missing, unexpected = m.load_state_dict(sd, strict=False)
        print(f"[LOAD] {name}: missing={len(missing)} unexpected={len(unexpected)}")
        m.eval()

    # ---- scan to collect correct and incorrect samples for the TRI model ----
    correct_samples = []
    wrong_samples = []

    for bi, batch in enumerate(loader):
        if bi >= args.max_scan_batches:
            break

        tri_preds, tri_labels = infer_logits(m_tri, device, batch, True, True, True)

        # we need raw text for qualitative display
        # MeldDataset typically stores text in its dataframe; easiest is use batch.meta if present;
        # fallback: reconstruct from batch.text_str if your collate provides it.
        # We'll handle both safely:
        texts = None
        utt_ids = None
        if hasattr(batch, "texts"):
            texts = batch.texts
        if hasattr(batch, "utt_ids"):
            utt_ids = batch.utt_ids

        # Fallback: try to fetch texts from dataset indices (not ideal but works)
        # If collate does not provide indices, we cannot do perfect mapping; we still display placeholder.
        if texts is None:
            texts = ["(text not provided by collate_meld)"] * len(tri_preds)
        if utt_ids is None:
            utt_ids = ["(utt_id not provided by collate_meld)"] * len(tri_preds)

        for i in range(len(tri_preds)):
            y = int(tri_labels[i])
            p = int(tri_preds[i])
            item = {
                "utt_id": str(utt_ids[i]),
                "text": str(texts[i]),
                "y": y,
                "p": p,
            }
            if p == y:
                correct_samples.append(item)
            else:
                wrong_samples.append(item)

    if len(correct_samples) == 0 or len(wrong_samples) == 0:
        raise RuntimeError(
            f"Could not collect samples: correct={len(correct_samples)} wrong={len(wrong_samples)}. "
            f"Increase --max_scan_batches."
        )

    # choose examples
    random.shuffle(correct_samples)
    random.shuffle(wrong_samples)
    correct_samples = correct_samples[: args.pick_per_group]
    wrong_samples = wrong_samples[: args.pick_per_group]

    # --- compute modality-only predicted labels for chosen samples ---
    # We re-run a small loader with shuffle=False is hard; simplest: just rescan until we hit chosen utt_ids.
    target_ids = set([s["utt_id"] for s in correct_samples + wrong_samples])
    gathered = {}

    for bi, batch in enumerate(loader):
        if len(gathered) == len(target_ids):
            break

        # attempt to get utt_ids from batch
        utt_ids = getattr(batch, "utt_ids", None)
        if utt_ids is None:
            continue

        # which positions match target ids?
        positions = [j for j, uid in enumerate(utt_ids) if str(uid) in target_ids]
        if not positions:
            continue

        # run modality models on full batch once
        p_tri, y_true = infer_logits(m_tri, device, batch, True, True, True)
        p_t, _ = infer_logits(m_t, device, batch, True, False, False)
        p_a, _ = infer_logits(m_a, device, batch, False, True, False)
        p_v, _ = infer_logits(m_v, device, batch, False, False, True)

        texts = getattr(batch, "texts", None)
        if texts is None:
            texts = ["(text not provided by collate_meld)"] * len(utt_ids)

        for j in positions:
            uid = str(utt_ids[j])
            gathered[uid] = {
                "utt_id": uid,
                "text": str(texts[j]),
                "y": int(y_true[j]),
                "tri": int(p_tri[j]),
                "text": int(p_t[j]),
                "audio": int(p_a[j]),
                "video": int(p_v[j]),
            }

    # merge back into ordered lists (keep original selection order)
    def resolve(lst):
        out = []
        for s in lst:
            uid = s["utt_id"]
            if uid not in gathered:
                # fallback: keep at least tri/y/text
                out.append({**s, "tri": s["p"], "text_pred": None, "audio_pred": None, "video_pred": None})
            else:
                out.append(gathered[uid])
        return out

    correct = resolve(correct_samples)
    wrong = resolve(wrong_samples)

    # ---- draw figure ----
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(2, max(args.pick_per_group, 1))

    def render_cell(ax, item, title_prefix):
        y = item["y"]
        tri = item.get("tri", item.get("p", -1))
        t = item.get("text", None)
        pa = item.get("audio", None)
        pv = item.get("video", None)
        pt = item.get("text", None) if "text" in item and isinstance(item["text"], int) else item.get("text", None)

        # predicted class names
        y_name = id2name.get(y, str(y))
        tri_name = id2name.get(tri, str(tri))
        t_name = id2name.get(item.get("text", tri), str(item.get("text", tri))) if isinstance(item.get("text", None), int) else None
        a_name = id2name.get(pa, str(pa)) if pa is not None else "N/A"
        v_name = id2name.get(pv, str(pv)) if pv is not None else "N/A"
        txt_name = id2name.get(item.get("text", -1), str(item.get("text", -1))) if isinstance(item.get("text", None), int) else None

        lines = []
        lines.append(f"GT: {y_name}")
        lines.append(f"Tri: {tri_name}")
        lines.append(f"Text: {id2name.get(item['text'], item['text']) if isinstance(item.get('text', None), int) else id2name.get(item.get('text', tri), tri)}" if isinstance(item.get("text", None), int) else f"Text: {id2name.get(item.get('text', tri), tri)}")
        # correct modality label printing:
        lines.append(f"Audio: {a_name}")
        lines.append(f"Video: {v_name}")
        lines.append("")
        lines.append(wrap(t, args.max_text_width))

        ax.axis("off")
        ax.set_title(f"{title_prefix}\nutt_id={item.get('utt_id','')}", fontsize=10)
        ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")

    # Row 0: Correct
    for c in range(args.pick_per_group):
        ax = fig.add_subplot(gs[0, c])
        render_cell(ax, correct[c], "Correct")

    # Row 1: Incorrect
    for c in range(args.pick_per_group):
        ax = fig.add_subplot(gs[1, c])
        render_cell(ax, wrong[c], "Incorrect")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=300)
    plt.close()
    print(f"[OK] Saved → {args.out}")


if __name__ == "__main__":
    main()
