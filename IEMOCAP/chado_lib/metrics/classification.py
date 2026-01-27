import numpy as np
import torch

@torch.no_grad()
def accuracy(pred: torch.Tensor, gold: torch.Tensor) -> float:
    return (pred == gold).float().mean().item()

@torch.no_grad()
def macro_f1(pred: torch.Tensor, gold: torch.Tensor, num_classes: int) -> float:
    f1s = []
    for c in range(num_classes):
        tp = ((pred == c) & (gold == c)).sum().item()
        fp = ((pred == c) & (gold != c)).sum().item()
        fn = ((pred != c) & (gold == c)).sum().item()
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        f1s.append(f1)
    return float(sum(f1s) / len(f1s))

def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def per_class_prf(cm: np.ndarray, class_names):
    rows = []
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        support = int(tp + fn)
        prec = float(tp / max(1, tp + fp))
        rec  = float(tp / max(1, tp + fn))
        f1   = float((2 * prec * rec) / max(1e-12, prec + rec))
        rows.append(dict(class_id=i, class_name=name, support=support,
                         precision=prec, recall=rec, f1=f1))
    return rows
