import torch
import torch.nn as nn

class CausalDisentangler(nn.Module):
    """
    Disentangles fused embedding into:
      z_shared: task-relevant shared factor
      z_spec  : modality-specific residual factor
    Regularize z_shared ⟂ z_spec (orthogonality).
    """
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
        )
        self.spec = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, z: torch.Tensor):
        z = self.norm(z)
        z_shared = self.shared(z)
        z_spec = self.spec(z)
        return z_shared, z_spec

    @staticmethod
    def orth_loss(z_shared: torch.Tensor, z_spec: torch.Tensor) -> torch.Tensor:
        """
        Encourage orthogonality between shared and specific components.
        """
        # normalize
        a = torch.nn.functional.normalize(z_shared, dim=1)
        b = torch.nn.functional.normalize(z_spec, dim=1)
        # mean squared cosine similarity
        cos = (a * b).sum(dim=1)
        return (cos ** 2).mean()
