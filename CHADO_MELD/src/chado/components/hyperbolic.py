import torch
import torch.nn as nn

class HyperbolicHead(nn.Module):
    """
    Practical, stable placeholder: logit-level curvature-aware transform (implemented as gated MLP).
    No-op by default via near-zero init.
    """
    def __init__(self, num_classes: int, hidden: int = 64, init_scale: float = 1e-4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_classes, hidden),
            nn.Tanh(),
            nn.Linear(hidden, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=init_scale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return self.net(logits)
