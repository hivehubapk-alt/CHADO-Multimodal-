import torch

def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "wav": torch.stack([b["wav"] for b in batch]),
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
    }
