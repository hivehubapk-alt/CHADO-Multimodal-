# src/data/label_maps.py
from typing import Dict, List

# Standard MELD 7-class order (paper-consistent)
EMO_ORDER_7: List[str] = [
    "neutral",
    "joy",
    "surprise",
    "anger",
    "sadness",
    "disgust",
    "fear",
]

def build_label_map_from_order(order: List[str]) -> Dict[str, int]:
    """
    Stable string → int mapping.
    MUST be deterministic across runs.
    """
    return {label: idx for idx, label in enumerate(order)}
