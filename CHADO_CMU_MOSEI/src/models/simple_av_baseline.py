import torch
import torch.nn as nn

class SimpleAVBaseline(nn.Module):
    """
    Super simple baseline - just concatenate + MLP
    """
    
    def __init__(self, audio_dim=74, visual_dim=35, num_classes=6):
        super().__init__()
        
        # Standardization (learned)
        self.audio_bn = nn.BatchNorm1d(audio_dim)
        self.visual_bn = nn.BatchNorm1d(visual_dim)
        
        # Simple MLP
        self.network = nn.Sequential(
            nn.Linear(audio_dim + visual_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, num_classes)
        )
        
    def forward(self, audio, visual, text=None):
        """
        Args:
            audio: [batch, time, 74]
            visual: [batch, time, 35]
        """
        # Clean NaN
        audio = torch.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        visual = torch.nan_to_num(visual, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Mean pool over time
        audio = audio.mean(dim=1)    # [batch, 74]
        visual = visual.mean(dim=1)  # [batch, 35]
        
        # Standardize
        audio = self.audio_bn(audio)
        visual = self.visual_bn(visual)
        
        # Concatenate
        x = torch.cat([audio, visual], dim=1)  # [batch, 109]
        
        # Forward
        logits = self.network(x)
        
        return logits
