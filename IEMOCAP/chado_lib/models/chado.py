import torch
import torch.nn as nn
from transformers import AutoModel

from .components import Disentangler

class CHADO(nn.Module):
    def __init__(self, text_model, audio_model, vision_model,
                 num_classes=4,
                 use_text=True, use_audio=True, use_video=True):
        super().__init__()
        self.use_text = use_text
        self.use_audio = use_audio
        self.use_video = use_video

        self.text_encoder = AutoModel.from_pretrained(text_model, use_safetensors=False, low_cpu_mem_usage=True)
        from transformers import AutoModel as AutoAudioModel
        self.audio_encoder = AutoAudioModel.from_pretrained(audio_model, use_safetensors=False, low_cpu_mem_usage=True)
        from transformers import AutoModel as AutoVisionModel
        self.vision_encoder = AutoVisionModel.from_pretrained(vision_model, use_safetensors=False, low_cpu_mem_usage=True)

        for m in [self.text_encoder, self.audio_encoder, self.vision_encoder]:
            if hasattr(m, "gradient_checkpointing_disable"):
                m.gradient_checkpointing_disable()
            if hasattr(m, "config") and hasattr(m.config, "gradient_checkpointing"):
                m.config.gradient_checkpointing = False

        self.text_dim = self.text_encoder.config.hidden_size
        self.audio_dim = self.audio_encoder.config.hidden_size
        self.vis_dim = self.vision_encoder.config.hidden_size

        self.fusion_dim = self.text_dim + self.audio_dim + self.vis_dim
        self.fuser = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.head = nn.Linear(512, num_classes)
        self.disent = Disentangler(in_dim=512, k_factors=4, d_factor=128)

    def forward_feats(self, input_ids, attention_mask, wav, pixel_values):
        B = input_ids.size(0)
        device = input_ids.device

        if self.use_text:
            t_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            t_cls = t_out.last_hidden_state[:, 0]
        else:
            t_cls = torch.zeros((B, self.text_dim), device=device, dtype=torch.float32)

        if self.use_audio:
            a_out = self.audio_encoder(input_values=wav)
            a_emb = a_out.last_hidden_state.mean(dim=1)
        else:
            a_emb = torch.zeros((B, self.audio_dim), device=device, dtype=torch.float32)

        if self.use_video:
            B2, T, C, H, W = pixel_values.shape
            pv = pixel_values.view(B2 * T, C, H, W)
            v_out = self.vision_encoder(pixel_values=pv)
            v_cls = v_out.last_hidden_state[:, 0].view(B2, T, -1).mean(dim=1)
        else:
            v_cls = torch.zeros((B, self.vis_dim), device=device, dtype=torch.float32)

        z = torch.cat([t_cls, a_emb, v_cls], dim=-1)
        h = self.fuser(z)
        return h

    def forward(self, input_ids, attention_mask, wav, pixel_values):
        h = self.forward_feats(input_ids, attention_mask, wav, pixel_values)
        logits = self.head(h)
        return logits, h
