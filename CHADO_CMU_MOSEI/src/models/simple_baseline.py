import torch
import torch.nn as nn

class SimpleMultimodalBaseline(nn.Module):
    """
    Simple baseline for CMU-MOSEI
    Concatenate features -> FC layers -> Classifier
    """
    
    def __init__(self, 
                 audio_dim=74,
                 visual_dim=35,
                 text_dim=300,
                 hidden_dim=128,
                 num_classes=6):
        super().__init__()
        
        # Feature projections
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Fusion and classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, audio, visual, text):
        """
        Args:
            audio: [batch, time, 74]
            visual: [batch, time, 35]
            text: [batch, time, 300]
        Returns:
            logits: [batch, num_classes]
        """
        # Mean pooling over time
        audio_pooled = audio.mean(dim=1)    # [batch, 74]
        visual_pooled = visual.mean(dim=1)  # [batch, 35]
        text_pooled = text.mean(dim=1)      # [batch, 300]
        
        # Project to common dimension
        audio_feat = self.audio_proj(audio_pooled)    # [batch, 128]
        visual_feat = self.visual_proj(visual_pooled) # [batch, 128]
        text_feat = self.text_proj(text_pooled)       # [batch, 128]
        
        # Concatenate
        fused = torch.cat([audio_feat, visual_feat, text_feat], dim=-1)  # [batch, 384]
        
        # Classify
        logits = self.classifier(fused)  # [batch, 6]
        
        return logits
