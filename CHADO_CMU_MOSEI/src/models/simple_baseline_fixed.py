import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMultimodalBaselineFixed(nn.Module):
    """
    Fixed baseline with normalization and stability improvements
    """
    
    def __init__(self, 
                 audio_dim=74,
                 visual_dim=35,
                 text_dim=300,
                 hidden_dim=128,
                 num_classes=6):
        super().__init__()
        
        # Layer normalization for input stability
        self.audio_norm = nn.LayerNorm(audio_dim)
        self.visual_norm = nn.LayerNorm(visual_dim)
        self.text_norm = nn.LayerNorm(text_dim)
        
        # Feature projections with BatchNorm
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Fusion with attention
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, audio, visual, text):
        """
        Args:
            audio: [batch, time, 74]
            visual: [batch, time, 35]
            text: [batch, time, 300]
        """
        batch_size = audio.size(0)
        
        # Normalize inputs (handle NaN)
        audio = torch.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        visual = torch.nan_to_num(visual, nan=0.0, posinf=0.0, neginf=0.0)
        text = torch.nan_to_num(text, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Layer norm on each timestep
        audio = self.audio_norm(audio)
        visual = self.visual_norm(visual)
        text = self.text_norm(text)
        
        # Mean pooling over time
        audio_pooled = audio.mean(dim=1)    # [batch, 74]
        visual_pooled = visual.mean(dim=1)  # [batch, 35]
        text_pooled = text.mean(dim=1)      # [batch, 300]
        
        # Project to common dimension
        audio_feat = self.audio_proj(audio_pooled)    # [batch, 128]
        visual_feat = self.visual_proj(visual_pooled) # [batch, 128]
        text_feat = self.text_proj(text_pooled)       # [batch, 128]
        
        # Stack for attention
        features = torch.stack([audio_feat, visual_feat, text_feat], dim=1)  # [batch, 3, 128]
        
        # Self-attention
        attn_out, _ = self.attention(features, features, features)  # [batch, 3, 128]
        
        # Flatten
        fused = attn_out.reshape(batch_size, -1)  # [batch, 384]
        
        # Classify
        logits = self.classifier(fused)  # [batch, 6]
        
        return logits
