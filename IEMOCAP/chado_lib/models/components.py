import numpy as np
import torch
import torch.nn as nn

def safe_l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6):
    return x / (x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps))

class Disentangler(nn.Module):
    def __init__(self, in_dim: int, k_factors: int = 4, d_factor: int = 128):
        super().__init__()
        self.k = k_factors
        self.d = d_factor
        self.proj = nn.Linear(in_dim, k_factors * d_factor)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.proj(z)
        return h.view(z.size(0), self.k, self.d)

def disentanglement_loss(factors: torch.Tensor) -> torch.Tensor:
    B, K, d = factors.shape
    x = safe_l2_normalize(factors.float(), dim=-1)
    G = torch.matmul(x, x.transpose(1, 2))
    I = torch.eye(K, device=factors.device, dtype=G.dtype).unsqueeze(0)
    return ((G - I) ** 2).mean()

def causal_dropout(x: torch.Tensor, p: float) -> torch.Tensor:
    if (not x.requires_grad) or p <= 0:
        return x
    keep = (torch.rand_like(x[..., :1]) > p).float()
    return x * keep

def poincare_project(x: torch.Tensor, c: float, eps: float = 1e-5) -> torch.Tensor:
    x = x.float()
    r = (1.0 / np.sqrt(max(c, 1e-8))) - eps
    norm = x.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-9)
    scale = torch.clamp(r / norm, max=1.0)
    return x * scale

def hyperbolic_reg_loss(factors: torch.Tensor, c: float) -> torch.Tensor:
    x = poincare_project(factors, c=c)
    return (x.norm(p=2, dim=-1) ** 2).mean()

def sinkhorn_ot_cost(x: torch.Tensor, y: torch.Tensor, eps: float = 0.05, iters: int = 30) -> torch.Tensor:
    x = x.float()
    y = y.float()
    B, K, d = x.shape
    C = torch.cdist(x, y, p=2).clamp_min(1e-6)

    a = torch.full((B, K), 1.0 / K, device=x.device, dtype=x.dtype)
    b = torch.full((B, K), 1.0 / K, device=x.device, dtype=x.dtype)

    Klog = (-C / eps).clamp(min=-50.0, max=50.0)
    u = torch.zeros_like(a)
    v = torch.zeros_like(b)

    for _ in range(iters):
        u = torch.log(a + 1e-9) - torch.logsumexp(Klog + v.unsqueeze(1), dim=2)
        v = torch.log(b + 1e-9) - torch.logsumexp(Klog + u.unsqueeze(2), dim=1)

    logP = Klog + u.unsqueeze(2) + v.unsqueeze(1)
    P = torch.exp(logP).clamp_min(1e-12)
    cost = (P * C).sum(dim=(1, 2)).mean()
    return cost

class MADState:
    def __init__(self):
        self.ema_teacher_logits = None  # [C]

def mad_loss(student_logits: torch.Tensor, teacher_logits_vec: torch.Tensor, temp: float) -> torch.Tensor:
    B, C = student_logits.shape
    teacher_logits = teacher_logits_vec.view(1, C).expand(B, C)
    s = (student_logits.float() / max(temp, 1e-6))
    t = (teacher_logits.float() / max(temp, 1e-6))
    p_t = torch.softmax(t, dim=-1).clamp_min(1e-8)
    log_p_s = torch.log_softmax(s, dim=-1)
    return -(p_t * log_p_s).sum(dim=-1).mean()
