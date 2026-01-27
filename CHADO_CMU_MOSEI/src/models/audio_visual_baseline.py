import torch
import torch.nn as nn

class AudioVisualBaseline(nn.Module):
    """
    Baseline using ONLY audio and visual (text is placeholder zeros)
    """
    
    def __init__(self, 
                 audio_dim=74,
                 visual_dim=35,
                 hidden_dim=256,
                 num_classes=6):
        super().__init__()
        
        # Input normalization
        self.audio_norm = nn.LayerNorm(audio_dim)
        self.visual_norm = nn.LayerNorm(visual_dim)
        
        # Temporal encoding with LSTM
        self.audio_lstm = nn.LSTM(audio_dim, hidden_dim//2, num_layers=2, 
                                   batch_first=True, dropout=0.3, bidirectional=True)
        self.visual_lstm = nn.LSTM(visual_dim, hidden_dim//2, num_layers=2,
                                    batch_first=True, dropout=0.3, bidirectional=True)
        
        # Attention pooling
        self.audio_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//4),
            nn.Tanh(),
            nn.Linear(hidden_dim//4, 1)
        )
        
        self.visual_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//4),
            nn.Tanh(),
            nn.Linear(hidden_dim//4, 1)
        )
        
        # Cross-modal fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        # Classifier
        self.classifier = nn.Linear(hidden_dim//2, num_classes)
        
    def attention_pooling(self, features, attention_layer):
        """
        Attention-based pooling over time dimension
        Args:
            features: [batch, time, hidden_dim]
        Returns:
            pooled: [batch, hidden_dim]
        """
        # Compute attention weights
        attn_weights = attention_layer(features)  # [batch, time, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Weighted sum
        pooled = torch.sum(features * attn_weights, dim=1)  # [batch, hidden_dim]
        
        return pooled
    
    def forward(self, audio, visual, text=None):
        """
        Args:
            audio: [batch, time, 74]
            visual: [batch, time, 35]
            text: ignored (placeholder)
        """
        # Handle NaN
        audio = torch.nan_to_num(audio, nan=0.0)
        visual = torch.nan_to_num(visual, nan=0.0)
        
        # Normalize
        audio = self.audio_norm(audio)
        visual = self.visual_norm(visual)
        
        # LSTM encoding
        audio_feat, _ = self.audio_lstm(audio)     # [batch, time, hidden_dim]
        visual_feat, _ = self.visual_lstm(visual)  # [batch, time, hidden_dim]
        
        # Attention pooling
        audio_pooled = self.attention_pooling(audio_feat, self.audio_attention)
        visual_pooled = self.attention_pooling(visual_feat, self.visual_attention)
        
        # Fusion
        fused = torch.cat([audio_pooled, visual_pooled], dim=-1)  # [batch, hidden_dim*2]
        fused = self.fusion(fused)  # [batch, hidden_dim//2]
        
        # Classification
        logits = self.classifier(fused)  # [batch, 6]
        
        return logits
