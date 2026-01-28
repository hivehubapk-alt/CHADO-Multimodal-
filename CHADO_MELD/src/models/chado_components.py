# src/models/chado_components.py
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .components.mad import mad_regularizer
from .components.transport import ot_alignment_loss
from .components.hyperbolic import hyperbolic_reg_loss, hyp_distance
from .components.causal import causal_invariance_loss
from .components.refinement import entropy_minimization_loss

@dataclass
class CHADOConfig:
    use_causal: bool = True
    use_hyperbolic: bool = True
    use_transport: bool = True
    use_refinement: bool = True

    w_mad: float = 1.0        # λ1
    w_ot: float = 1.0         # λ2
    w_hyp: float = 1.0        # λ3
    w_causal: float = 1.0
    w_refine: float = 1.0

    mad_mode: str = "entropy"     # entropy | margin
    ot_eps: float = 0.05
    ot_iters: int = 30
    ot_metric: str = "cosine"

    hyp_c: float = 1.0            # curvature
    causal_mode: str = "kl"
    causal_temp: float = 1.0

def compute_chado_losses(
    cfg: CHADOConfig,
    logits_full: torch.Tensor,                    # [B,C]
    embeddings: Dict[str, Optional[torch.Tensor]],# keys: "t","a","v","fused"
    logits_intervened: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Returns dict of named losses. Caller decides how to sum.
    """
    losses: Dict[str, torch.Tensor] = {}

    # MAD ambiguity regularizer (always defined; can be weighted to 0)
    losses["mad"] = mad_regularizer(logits_full, mode=cfg.mad_mode)

    # OT alignment (requires >=2 modalities present)
    if cfg.use_transport:
        zt, za, zv = embeddings.get("t"), embeddings.get("a"), embeddings.get("v")
        ot = 0.0
        n = 0
        if (zt is not None) and (za is not None):
            ot = ot + ot_alignment_loss(zt, za, eps=cfg.ot_eps, iters=cfg.ot_iters, metric=cfg.ot_metric); n += 1
        if (zt is not None) and (zv is not None):
            ot = ot + ot_alignment_loss(zt, zv, eps=cfg.ot_eps, iters=cfg.ot_iters, metric=cfg.ot_metric); n += 1
        if (za is not None) and (zv is not None):
            ot = ot + ot_alignment_loss(za, zv, eps=cfg.ot_eps, iters=cfg.ot_iters, metric=cfg.ot_metric); n += 1
        losses["ot"] = ot / max(1, n)
    else:
        losses["ot"] = torch.zeros((), device=logits_full.device)

    # Hyperbolic regularization (keep embeddings away from boundary)
    if cfg.use_hyperbolic:
        zf = embeddings.get("fused")
        if zf is None:
            zf = next((z for z in [embeddings.get("t"), embeddings.get("a"), embeddings.get("v")] if z is not None), None)
        losses["hyp"] = hyperbolic_reg_loss(zf, c=cfg.hyp_c) if zf is not None else torch.zeros((), device=logits_full.device)
    else:
        losses["hyp"] = torch.zeros((), device=logits_full.device)

    # Causal invariance under interventions
    if cfg.use_causal and (logits_intervened is not None):
        losses["causal"] = causal_invariance_loss(
            logits_full, logits_intervened, mode=cfg.causal_mode, temperature=cfg.causal_temp
        )
    else:
        losses["causal"] = torch.zeros((), device=logits_full.device)

    # Refinement: entropy minimization
    if cfg.use_refinement:
        losses["refine"] = entropy_minimization_loss(logits_full)
    else:
        losses["refine"] = torch.zeros((), device=logits_full.device)

    return losses

def sum_chado_loss(cfg: CHADOConfig, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Weighted sum for training.
    """
    return (
        cfg.w_mad * losses["mad"]
        + cfg.w_ot * losses["ot"]
        + cfg.w_hyp * losses["hyp"]
        + cfg.w_causal * losses["causal"]
        + cfg.w_refine * losses["refine"]
    )
