# src/models/baseline_trimodal.py
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel


@dataclass
class BaselineOutputs:
    logits: torch.Tensor                         # [B, C]
    z_fused: torch.Tensor                        # [B, D]
    z_t: Optional[torch.Tensor] = None           # [B, D]
    z_a: Optional[torch.Tensor] = None           # [B, D]
    z_v: Optional[torch.Tensor] = None           # [B, D]


class TriModalBaseline(nn.Module):
    """
    Clean tri-modal baseline:
      - text: HF encoder (e.g., roberta-base)
      - audio: HF encoder (e.g., wav2vec2-base)
      - video: HF encoder (e.g., vit-base-patch16-224-in21k)

    Returns pooled modality embeddings -> fused -> logits.

    IMPORTANT:
      - Safe when any modality is disabled (no encoder called).
      - Supports modality_mask [B,3] (T,A,V) to drop modalities per-sample.
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
        local_files_only: bool = False,
    ):
        super().__init__()

        self.use_text = bool(use_text)
        self.use_audio = bool(use_audio)
        self.use_video = bool(use_video)
        self.use_gated_fusion = bool(use_gated_fusion)

        # ---- encoders (only instantiated if enabled)
        self.text_encoder = AutoModel.from_pretrained(text_model_name, local_files_only=local_files_only) if self.use_text else None
        self.audio_encoder = AutoModel.from_pretrained(audio_model_name, local_files_only=local_files_only) if self.use_audio else None
        self.video_encoder = AutoModel.from_pretrained(video_model_name, local_files_only=local_files_only) if self.use_video else None

        # infer hidden sizes
        t_dim = self.text_encoder.config.hidden_size if self.text_encoder is not None else 0
        a_dim = self.audio_encoder.config.hidden_size if self.audio_encoder is not None else 0
        v_dim = self.video_encoder.config.hidden_size if self.video_encoder is not None else 0

        self.proj_t = nn.Linear(t_dim, proj_dim) if self.use_text else None
        self.proj_a = nn.Linear(a_dim, proj_dim) if self.use_audio else None
        self.proj_v = nn.Linear(v_dim, proj_dim) if self.use_video else None

        self.drop = nn.Dropout(dropout)

        # gated fusion weights per modality
        if self.use_gated_fusion:
            self.gate = nn.Sequential(
                nn.Linear(proj_dim * 3, proj_dim),
                nn.GELU(),
                nn.Linear(proj_dim, 3),
                nn.Sigmoid(),
            )
        else:
            self.gate = None

        self.head = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_classes),
        )

    def _pool_text(self, text_out) -> torch.Tensor:
        # prefer pooler_output if available else CLS token
        if hasattr(text_out, "pooler_output") and text_out.pooler_output is not None:
            return text_out.pooler_output
        return text_out.last_hidden_state[:, 0]  # CLS

    def _pool_audio(self, audio_out) -> torch.Tensor:
        # mean pool time dimension
        x = audio_out.last_hidden_state  # [B,T,H]
        return x.mean(dim=1)

    def _pool_video(self, video_out) -> torch.Tensor:
        # ViT: CLS token at position 0
        x = video_out.last_hidden_state  # [B,N,H]
        return x[:, 0]

    def forward(
        self,
        text_input: Optional[Dict[str, torch.Tensor]] = None,
        audio_wave: Optional[torch.Tensor] = None,        # [B, L]
        video_frames: Optional[torch.Tensor] = None,      # [B, T, 3, H, W]
        modality_mask: Optional[torch.Tensor] = None,     # [B,3] (T,A,V) 1=keep 0=drop
    ) -> BaselineOutputs:

        B = None
        device = None
        if text_input is not None:
            any_tensor = next(iter(text_input.values()))
            B = any_tensor.size(0); device = any_tensor.device
        elif audio_wave is not None:
            B = audio_wave.size(0); device = audio_wave.device
        elif video_frames is not None:
            B = video_frames.size(0); device = video_frames.device
        else:
            raise ValueError("At least one modality input must be provided.")

        if modality_mask is None:
            modality_mask = torch.ones(B, 3, device=device, dtype=torch.float32)
        else:
            modality_mask = modality_mask.to(device=device, dtype=torch.float32)

        # ---- Text
        zt = None
        if self.use_text and (text_input is not None):
            tkeep = modality_mask[:, 0].view(B, 1)
            out_t = self.text_encoder(**text_input)
            pt = self._pool_text(out_t)
            zt = self.proj_t(pt)
            zt = self.drop(F.normalize(zt, dim=-1))
            zt = zt * tkeep  # drop per-sample

        # ---- Audio
        za = None
        if self.use_audio and (audio_wave is not None):
            akeep = modality_mask[:, 1].view(B, 1)
            out_a = self.audio_encoder(input_values=audio_wave)
            pa = self._pool_audio(out_a)
            za = self.proj_a(pa)
            za = self.drop(F.normalize(za, dim=-1))
            za = za * akeep

        # ---- Video
        zv = None
        if self.use_video and (video_frames is not None):
            vkeep = modality_mask[:, 2].view(B, 1)
            # collapse frames by mean after encoding each frame token set
            # If your collate already gives a single image [B,3,H,W], this still works:
            if video_frames.dim() == 5:
                B, T, C, H, W = video_frames.shape
                vf = video_frames.view(B * T, C, H, W)
                out_v = self.video_encoder(pixel_values=vf)
                pv = self._pool_video(out_v).view(B, T, -1).mean(dim=1)
            else:
                out_v = self.video_encoder(pixel_values=video_frames)
                pv = self._pool_video(out_v)
            zv = self.proj_v(pv)
            zv = self.drop(F.normalize(zv, dim=-1))
            zv = zv * vkeep

        # ---- Fuse
        # If a modality is disabled or missing, treat as zeros.
        zeros = torch.zeros(B, self.head[0].in_features, device=device)

        # NOTE: self.head[0].in_features == proj_dim because head starts with Linear(proj_dim,...)
        # but we don't store proj_dim; infer it by using any existing z
        proj_dim = None
        for z in (zt, za, zv):
            if z is not None:
                proj_dim = z.size(-1)
                break
        if proj_dim is None:
            raise RuntimeError("No modality embedding produced. Check inputs/use_* flags.")
        zeros = torch.zeros(B, proj_dim, device=device)

        zt0 = zt if zt is not None else zeros
        za0 = za if za is not None else zeros
        zv0 = zv if zv is not None else zeros

        if self.gate is not None:
            g = self.gate(torch.cat([zt0, za0, zv0], dim=-1))  # [B,3]
            zf = (g[:, 0:1] * zt0 + g[:, 1:2] * za0 + g[:, 2:3] * zv0)
        else:
            # average over available modalities by mask
            m = modality_mask
            denom = m.sum(dim=1, keepdim=True).clamp(min=1.0)
            zf = (zt0 + za0 + zv0) / denom

        zf = self.drop(zf)
        logits = self.head(zf)

        return BaselineOutputs(logits=logits, z_fused=zf, z_t=zt, z_a=za, z_v=zv)






# from dataclasses import dataclass
# from typing import Optional, Dict

# import torch
# import torch.nn as nn
# from transformers import AutoModel


# @dataclass
# class BaselineOutputs:
#     logits: torch.Tensor
#     fused: torch.Tensor
#     gate: Optional[torch.Tensor]  # [B, M]


# class GatedFusion(nn.Module):
#     def __init__(self, dim: int, num_modalities: int):
#         super().__init__()
#         self.num_modalities = num_modalities
#         self.gate = nn.Linear(dim * num_modalities, num_modalities)

#     def forward(self, embs: torch.Tensor):
#         # embs: [B, M, D]
#         B, M, D = embs.shape
#         x = embs.reshape(B, M * D)
#         w = self.gate(x)               # [B,M]
#         g = torch.softmax(w, dim=-1)   # [B,M]
#         fused = (embs * g.unsqueeze(-1)).sum(dim=1)  # [B,D]
#         return fused, g


# class TriModalBaseline(nn.Module):
#     def __init__(
#         self,
#         text_model_name: str,
#         audio_model_name: str,
#         video_model_name: str,
#         num_classes: int,
#         proj_dim: int = 256,
#         dropout: float = 0.2,
#         use_text: bool = True,
#         use_audio: bool = True,
#         use_video: bool = True,
#         use_gated_fusion: bool = True,
#     ):
#         super().__init__()
#         self.use_text = use_text
#         self.use_audio = use_audio
#         self.use_video = use_video

#         self.modalities = []

#         if use_text:
#             self.modalities.append("text")
#             self.text_encoder = AutoModel.from_pretrained(text_model_name)
#             t_dim = self.text_encoder.config.hidden_size
#             self.text_proj = nn.Sequential(nn.Linear(t_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout))
#         else:
#             self.text_encoder = None
#             self.text_proj = None

#         if use_audio:
#             self.modalities.append("audio")
#             self.audio_encoder = AutoModel.from_pretrained(audio_model_name)
#             a_dim = self.audio_encoder.config.hidden_size
#             self.audio_proj = nn.Sequential(nn.Linear(a_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout))
#         else:
#             self.audio_encoder = None
#             self.audio_proj = None

#         if use_video:
#             self.modalities.append("video")
#             self.video_encoder = AutoModel.from_pretrained(video_model_name)
#             v_dim = self.video_encoder.config.hidden_size
#             self.video_proj = nn.Sequential(nn.Linear(v_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout))
#         else:
#             self.video_encoder = None
#             self.video_proj = None

#         self.num_modalities = len(self.modalities)
#         assert self.num_modalities >= 1, "At least one modality must be enabled."

#         self.fusion = GatedFusion(proj_dim, self.num_modalities) if (use_gated_fusion and self.num_modalities > 1) else None
#         self.classifier = nn.Linear(proj_dim, num_classes)

#     def freeze_encoders(self, freeze_text: bool, freeze_audio: bool, freeze_video: bool):
#         def set_req(m, req: bool):
#             if m is None:
#                 return
#             for p in m.parameters():
#                 p.requires_grad = req

#         if self.text_encoder is not None:
#             set_req(self.text_encoder, not freeze_text)
#         if self.audio_encoder is not None:
#             set_req(self.audio_encoder, not freeze_audio)
#         if self.video_encoder is not None:
#             set_req(self.video_encoder, not freeze_video)

#     def forward(
#         self,
#         text_input: Optional[Dict[str, torch.Tensor]] = None,
#         audio_wave: Optional[torch.Tensor] = None,
#         video_frames: Optional[torch.Tensor] = None,
#         modality_mask: Optional[torch.Tensor] = None,  # [B,M]
#     ):
#         embs = []

#         for m in self.modalities:
#             if m == "text":
#                 out = self.text_encoder(**text_input)
#                 pooled = out.last_hidden_state[:, 0]  # CLS
#                 embs.append(self.text_proj(pooled))

#             elif m == "audio":
#                 out = self.audio_encoder(input_values=audio_wave)
#                 pooled = out.last_hidden_state.mean(dim=1)
#                 embs.append(self.audio_proj(pooled))

#             elif m == "video":
#                 # video_frames: [B,T,3,H,W] -> ViT expects [B,3,H,W]
#                 B, T, C, H, W = video_frames.shape
#                 frames = video_frames.reshape(B * T, C, H, W)
#                 out = self.video_encoder(pixel_values=frames)
#                 pooled = out.last_hidden_state[:, 0]        # [B*T, D]
#                 pooled = pooled.reshape(B, T, -1).mean(dim=1)
#                 embs.append(self.video_proj(pooled))

#         embs = torch.stack(embs, dim=1)  # [B,M,D]

#         if modality_mask is not None:
#             embs = embs * modality_mask.unsqueeze(-1)

#         gate = None
#         if self.fusion is not None:
#             fused, gate = self.fusion(embs)
#         else:
#             fused = embs.mean(dim=1)

#         logits = self.classifier(fused)
#         return logits, fused, gate
