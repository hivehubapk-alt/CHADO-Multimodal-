import torch
import torch.nn as nn

class SelfCorrectionHead(nn.Module):
    """
    Residual correction:
      logits1 = base classifier(z)
      delta   = f([z, sigmoid(logits1)])
      logits2 = logits1 + delta
    """
    def __init__(self, d_model: int = 256, num_classes: int = 6):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model + num_classes, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, z: torch.Tensor, logits1: torch.Tensor) -> torch.Tensor:
        p1 = torch.sigmoid(logits1)
        x = torch.cat([z, p1], dim=1)
        delta = self.mlp(x)
        return logits1 + delta
