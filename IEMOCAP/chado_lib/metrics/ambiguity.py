import numpy as np

def predictive_entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(probs, eps, 1.0)
    return -(p * np.log(p)).sum(axis=1)

def bucket_by_entropy(ent: np.ndarray, low_thr: float, high_thr: float):
    low = ent < low_thr
    mid = (ent >= low_thr) & (ent <= high_thr)
    high = ent > high_thr
    return low, mid, high
