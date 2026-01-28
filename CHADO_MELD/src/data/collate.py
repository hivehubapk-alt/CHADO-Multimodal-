# src/data/collate.py
import torch
from types import SimpleNamespace
from typing import List

def collate_meld(
    batch: List[dict],
    tokenizer,
    use_text: bool,
    use_audio: bool,
    use_video: bool,
):
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

    out = {
        "labels": labels,
        "text_input": None,
        "audio_wave": None,
        "video_frames": None,
    }

    if use_text:
        texts = [b["text"] for b in batch]
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        out["text_input"] = enc

    if use_audio:
        # Placeholder: actual waveform loading handled in model / loader
        out["audio_wave"] = None

    if use_video:
        # Placeholder: actual frame loading handled in model / loader
        out["video_frames"] = None

    return SimpleNamespace(**out)
