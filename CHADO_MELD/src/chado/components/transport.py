import torch
import torch.nn as nn

class TransportHead(nn.Module):
    """
    OT/Transport as a simple learnable logit reweighting (stable proxy).
    No-op by default via near-zero init.
    """
    def __init__(self, num_classes: int, init_scale: float = 1e-4):
        super().__init__()
        self.W = nn.Linear(num_classes, num_classes, bias=False)
        nn.init.normal_(self.W.weight, std=init_scale)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return self.W(logits)
