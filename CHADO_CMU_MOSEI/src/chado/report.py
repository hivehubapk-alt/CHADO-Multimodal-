import torch

def _safe_div(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return a / (b + eps)

@torch.no_grad()
def multilabel_prf_report(y_true: torch.Tensor, y_pred: torch.Tensor, class_names=None):
    """
    y_true, y_pred: int tensors [N,C] with values in {0,1}
    Returns dict with per-class and averages.
    """
    y_true = y_true.int()
    y_pred = y_pred.int()

    tp = (y_true & y_pred).sum(dim=0).float()
    fp = ((1 - y_true) & y_pred).sum(dim=0).float()
    fn = (y_true & (1 - y_pred)).sum(dim=0).float()
    tn = ((1 - y_true) & (1 - y_pred)).sum(dim=0).float()

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    support = y_true.sum(dim=0).float()  # positives per class
    total_support = support.sum().clamp(min=1.0)

    macro_p = precision.mean()
    macro_r = recall.mean()
    macro_f1 = f1.mean()

    weighted_p = (precision * support).sum() / total_support
    weighted_r = (recall * support).sum() / total_support
    weighted_f1 = (f1 * support).sum() / total_support

    # micro
    tp_m = tp.sum()
    fp_m = fp.sum()
    fn_m = fn.sum()
    micro_p = _safe_div(tp_m, tp_m + fp_m)
    micro_r = _safe_div(tp_m, tp_m + fn_m)
    micro_f1 = _safe_div(2 * micro_p * micro_r, micro_p + micro_r)

    rows = []
    C = y_true.shape[1]
    if class_names is None:
        class_names = [str(i) for i in range(C)]
    for i in range(C):
        rows.append({
            "name": class_names[i],
            "precision": float(precision[i].item()),
            "recall": float(recall[i].item()),
            "f1": float(f1[i].item()),
            "support": int(support[i].item())
        })

    summary = {
        "macro_avg": dict(precision=float(macro_p.item()), recall=float(macro_r.item()), f1=float(macro_f1.item())),
        "weighted_avg": dict(precision=float(weighted_p.item()), recall=float(weighted_r.item()), f1=float(weighted_f1.item())),
        "micro_avg": dict(precision=float(micro_p.item()), recall=float(micro_r.item()), f1=float(micro_f1.item())),
    }

    confusion = {
        "tp": tp.int(), "fp": fp.int(), "fn": fn.int(), "tn": tn.int()
    }
    return rows, summary, confusion

def format_report(rows, summary):
    lines = []
    lines.append("\nPer-class metrics (multilabel):")
    lines.append(f"{'class':10s} {'prec':>7s} {'rec':>7s} {'f1':>7s} {'support':>8s}")
    for r in rows:
        lines.append(f"{r['name']:10s} {r['precision']*100:7.2f} {r['recall']*100:7.2f} {r['f1']*100:7.2f} {r['support']:8d}")

    for k in ["micro_avg", "macro_avg", "weighted_avg"]:
        v = summary[k]
        lines.append(f"{k:10s} {v['precision']*100:7.2f} {v['recall']*100:7.2f} {v['f1']*100:7.2f}")

    return "\n".join(lines)
