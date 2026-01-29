import torch


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    # x, y: [B, L, D] or [N, D]
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    hsic = (x @ y.T).pow(2).sum()
    hsic_xx = (x @ x.T).pow(2).sum().clamp_min(1e-12)
    hsic_yy = (y @ y.T).pow(2).sum().clamp_min(1e-12)
    return (hsic / (hsic_xx.sqrt() * hsic_yy.sqrt())).item()
