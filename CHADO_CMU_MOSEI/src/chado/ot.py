import torch

def sinkhorn_distance(a: torch.Tensor, b: torch.Tensor, eps: float = 0.05, iters: int = 30) -> torch.Tensor:
    """
    Differentiable Sinkhorn distance between two empirical distributions.
    a, b: [B, D] embeddings
    returns: scalar OT distance
    """
    # cost matrix: squared euclidean
    C = torch.cdist(a, b, p=2) ** 2  # [B,B]

    # uniform marginals
    B = a.size(0)
    mu = torch.full((B,), 1.0 / B, device=a.device, dtype=a.dtype)
    nu = torch.full((B,), 1.0 / B, device=a.device, dtype=a.dtype)

    # kernel
    K = torch.exp(-C / eps)

    u = torch.ones_like(mu)
    v = torch.ones_like(nu)

    # Sinkhorn iterations
    for _ in range(iters):
        u = mu / (K @ v + 1e-9)
        v = nu / (K.t() @ u + 1e-9)

    # transport plan
    P = torch.diag(u) @ K @ torch.diag(v)
    dist = torch.sum(P * C)
    return dist


def ot_counterfactual_consistency(z: torch.Tensor, sigma: float = 0.05, eps: float = 0.05, iters: int = 30) -> torch.Tensor:
    """
    CHADO OT consistency:
    - create counterfactual embedding z_cf by small perturbation
    - penalize distribution shift via Sinkhorn distance
    """
    noise = sigma * torch.randn_like(z)
    z_cf = z + noise
    return sinkhorn_distance(z, z_cf, eps=eps, iters=iters)
