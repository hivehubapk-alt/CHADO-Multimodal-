# src/models/chado_trimodal.py
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn

from src.models.baseline_trimodal import TriModalBaseline
from src.models.chado_components import CHADOConfig, compute_chado_losses, sum_chado_loss
from src.models.components.causal import modality_intervention_mask


@dataclass
class CHADOOutputs:
    logits: torch.Tensor
    losses: Dict[str, torch.Tensor]
    total_chado_loss: torch.Tensor
    embeddings: Dict[str, Optional[torch.Tensor]]
    logits_intervened: Optional[torch.Tensor] = None


class CHADOTrimodal(nn.Module):
    """
    CHADO wrapper around TriModalBaseline.
    This module does NOT do optimizer steps; it returns losses + logits.
    """
    def __init__(
        self,
        text_model_name: str,
        audio_model_name: str,
        video_model_name: str,
        num_classes: int,
        proj_dim: int = 256,
        dropout: float = 0.2,
        use_text: bool = True,
        use_audio: bool = True,
        use_video: bool = True,
        use_gated_fusion: bool = True,
        # CHADO toggles
        use_causal: bool = True,
        use_hyperbolic: bool = True,
        use_transport: bool = True,
        use_refinement: bool = True,
        # weights / params
        w_mad: float = 1.0,
        w_ot: float = 1.0,
        w_hyp: float = 1.0,
        w_causal: float = 1.0,
        w_refine: float = 1.0,
        hyp_c: float = 1.0,
        ot_eps: float = 0.05,
        ot_iters: int = 30,
        mad_mode: str = "entropy",
        causal_drop_prob: float = 0.33,
        causal_mode: str = "kl",
        causal_temp: float = 1.0,
        local_files_only: bool = False,
    ):
        super().__init__()

        self.base = TriModalBaseline(
            text_model_name=text_model_name,
            audio_model_name=audio_model_name,
            video_model_name=video_model_name,
            num_classes=num_classes,
            proj_dim=proj_dim,
            dropout=dropout,
            use_text=use_text,
            use_audio=use_audio,
            use_video=use_video,
            use_gated_fusion=use_gated_fusion,
            local_files_only=local_files_only,
        )

        self.cfg = CHADOConfig(
            use_causal=use_causal,
            use_hyperbolic=use_hyperbolic,
            use_transport=use_transport,
            use_refinement=use_refinement,
            w_mad=w_mad,
            w_ot=w_ot,
            w_hyp=w_hyp,
            w_causal=w_causal,
            w_refine=w_refine,
            mad_mode=mad_mode,
            ot_eps=ot_eps,
            ot_iters=ot_iters,
            hyp_c=hyp_c,
            causal_mode=causal_mode,
            causal_temp=causal_temp,
        )

        self.causal_drop_prob = float(causal_drop_prob)

    def forward(
        self,
        text_input: Optional[Dict[str, torch.Tensor]] = None,
        audio_wave: Optional[torch.Tensor] = None,
        video_frames: Optional[torch.Tensor] = None,
        modality_mask: Optional[torch.Tensor] = None,
        # training flag: whether to compute CHADO losses
        compute_losses: bool = True,
    ) -> CHADOOutputs:

        out = self.base(
            text_input=text_input,
            audio_wave=audio_wave,
            video_frames=video_frames,
            modality_mask=modality_mask,
        )

        embeddings = {
            "t": out.z_t,
            "a": out.z_a,
            "v": out.z_v,
            "fused": out.z_fused,
        }

        logits_intervened = None

        if compute_losses and self.cfg.use_causal:
            # sample intervention masks that drop modalities
            # (T,A,V) per-sample
            B = out.logits.size(0)
            device = out.logits.device
            m_int = modality_intervention_mask(B, drop_prob=self.causal_drop_prob, device=device)
            # combine with any external mask
            if modality_mask is not None:
                m_int = m_int * modality_mask.to(device=device, dtype=torch.float32)

            out_int = self.base(
                text_input=text_input,
                audio_wave=audio_wave,
                video_frames=video_frames,
                modality_mask=m_int,
            )
            logits_intervened = out_int.logits

        if not compute_losses:
            losses = {"mad": out.logits.new_zeros(()),
                      "ot": out.logits.new_zeros(()),
                      "hyp": out.logits.new_zeros(()),
                      "causal": out.logits.new_zeros(()),
                      "refine": out.logits.new_zeros(())}
            total = out.logits.new_zeros(())
            return CHADOOutputs(
                logits=out.logits,
                losses=losses,
                total_chado_loss=total,
                embeddings=embeddings,
                logits_intervened=logits_intervened,
            )

        losses = compute_chado_losses(
            cfg=self.cfg,
            logits_full=out.logits,
            embeddings=embeddings,
            logits_intervened=logits_intervened,
        )
        total = sum_chado_loss(self.cfg, losses)

        return CHADOOutputs(
            logits=out.logits,
            losses=losses,
            total_chado_loss=total,
            embeddings=embeddings,
            logits_intervened=logits_intervened,
        )





# import torch
# import torch.nn as nn

# from src.models.baseline_trimodal import TriModalBaseline
# from src.chado.components import CausalHead, HyperbolicHead, TransportHead, RefinementHead, MADComputer

# class CHADOTrimodal(nn.Module):
#     """
#     Safe CHADO wrapper:
#     - calls TriModalBaseline to get logits
#     - applies optional component residuals to logits
#     - components are near-zero init => baseline preserved at start
#     """
#     def __init__(
#         self,
#         text_model_name: str,
#         audio_model_name: str,
#         video_model_name: str,
#         num_classes: int,
#         proj_dim: int,
#         dropout: float,
#         use_text: bool,
#         use_audio: bool,
#         use_video: bool,
#         use_gated_fusion: bool,
#         # CHADO toggles
#         use_causal: bool = True,
#         use_hyperbolic: bool = True,
#         use_transport: bool = True,
#         use_refinement: bool = True,
#         # weights
#         w_causal: float = 1.0,
#         w_hyperbolic: float = 1.0,
#         w_transport: float = 1.0,
#         w_refine: float = 1.0,
#     ):
#         super().__init__()
#         self.base = TriModalBaseline(
#             text_model_name=text_model_name,
#             audio_model_name=audio_model_name,
#             video_model_name=video_model_name,
#             num_classes=num_classes,
#             proj_dim=proj_dim,
#             dropout=dropout,
#             use_text=use_text,
#             use_audio=use_audio,
#             use_video=use_video,
#             use_gated_fusion=use_gated_fusion,
#         )

#         self.num_classes = num_classes

#         self.use_causal = use_causal
#         self.use_hyperbolic = use_hyperbolic
#         self.use_transport = use_transport
#         self.use_refinement = use_refinement

#         self.w_causal = w_causal
#         self.w_hyperbolic = w_hyperbolic
#         self.w_transport = w_transport
#         self.w_refine = w_refine

#         self.causal = CausalHead(num_classes) if use_causal else None
#         self.hyperbolic = HyperbolicHead(num_classes) if use_hyperbolic else None
#         self.transport = TransportHead(num_classes) if use_transport else None
#         self.refine = RefinementHead(num_classes) if use_refinement else None

#         self.mad = MADComputer()

#     def forward(self, text_input=None, audio_wave=None, video_frames=None, modality_mask=None):
#         logits, aux1, aux2 = self.base(
#             text_input=text_input,
#             audio_wave=audio_wave,
#             video_frames=video_frames,
#             modality_mask=modality_mask,
#         )

#         # Residual composition (safe)
#         delta = 0.0
#         if self.causal is not None:
#             delta = delta + self.w_causal * self.causal(logits)
#         if self.hyperbolic is not None:
#             delta = delta + self.w_hyperbolic * self.hyperbolic(logits)
#         if self.transport is not None:
#             delta = delta + self.w_transport * self.transport(logits)
#         if self.refine is not None:
#             delta = delta + self.w_refine * self.refine(logits)

#         out_logits = logits + delta

#         # MAD score for analysis (not used in loss by default)
#         mad_scores = self.mad.compute_from_logits(out_logits).detach()

#         return out_logits, mad_scores, (aux1, aux2)
