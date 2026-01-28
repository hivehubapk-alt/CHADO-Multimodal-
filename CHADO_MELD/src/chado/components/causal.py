import torch
import torch.nn as nn

class CausalHead(nn.Module):
    """
    No-op by default: produces delta logits initialized to ~0.
    """
    def __init__(self, num_classes: int, hidden: int = 0, init_scale: float = 1e-4):
        super().__init__()
        if hidden and hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(num_classes, hidden),
                nn.GELU(),
                nn.Linear(hidden, num_classes),
            )
        else:
            self.net = nn.Linear(num_classes, num_classes, bias=True)

        # Initialize near-zero so baseline is preserved at start
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=init_scale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return self.net(logits)
