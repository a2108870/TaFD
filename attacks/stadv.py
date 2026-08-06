# stadv_attack_std.py
# -*- coding: utf-8 -*-
"""
stAdv-like differentiable spatial transformation attack (PGD-style).
"""

import torch
import torch.nn.functional as F
from typing import Optional


def _make_base_grid(B: int, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Generate a normalized coordinate grid in [-1,1], shape [B, H, W, 2], last dim (x, y).
    """
    ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    base = torch.stack([grid_x, grid_y], dim=-1)
    return base.unsqueeze(0).expand(B, H, W, 2).contiguous()


def stadv_attack(
    img: torch.Tensor,
    y: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
    eps: float = 0.045,
    steps: int = 10,
    alpha: Optional[float] = None,
    mode: str = "linf",
    padding_mode: str = "border",
    grid_align_corners: bool = True,
    norm_mean=None,
    norm_std=None,
) -> torch.Tensor:
    """
    stAdv (spatial transform) style attack.
    Note: Model handles normalization internally, so input should be in [0,1] pixel domain.
    norm_mean and norm_std parameters are kept for API compatibility but not used.

    Args:
        img: [B,3,H,W] input images (pixel domain, expected in [0,1])
        y:   [B] labels
        model: classifier (handles normalization internally)
        device: torch.device for computation
        eps: maximum per-coordinate perturbation in normalized grid coordinates [-1,1]
        steps: PGD steps
        alpha: step size. If None, defaults to eps*1.5/steps
        mode: "linf" or "l2"
        padding_mode: grid_sample padding_mode ("border" or "zeros")
        grid_align_corners: align_corners for grid_sample

    Returns:
        x_adv: adversarial images [B,3,H,W] clamped to [0,1]
    """
    model.eval()

    x_nat = img.detach().to(device)
    labels = y.detach().to(device)
    B, C, H, W = x_nat.shape
    dtype = x_nat.dtype

    base_grid = _make_base_grid(B, H, W, device, dtype)
    delta = torch.zeros_like(base_grid, requires_grad=True)

    if alpha is None:
        step_alpha = float(eps * 1.5 / max(1, steps))
    else:
        step_alpha = float(alpha)

    for _ in range(int(steps)):
        grid = (base_grid + delta).clamp(-1.0, 1.0)

        x_warp = F.grid_sample(
            x_nat, grid, mode='bilinear',
            padding_mode=padding_mode, align_corners=grid_align_corners
        )

        # Forward pass (model handles normalization internally)
        logits_tuple = model(x_warp)
        logits = logits_tuple[0] if isinstance(logits_tuple, (tuple, list)) else logits_tuple
        loss = F.cross_entropy(logits, labels)

        if delta.grad is not None:
            delta.grad.detach_()
            delta.grad.zero_()
        loss.backward()

        with torch.no_grad():
            g = delta.grad
            if g is None:
                break

            if mode.lower() == "l2":
                g_flat = g.view(B, -1)
                g_norm = torch.norm(g_flat, dim=1, keepdim=True).clamp_min(1e-12)
                g_unit = (g_flat / g_norm).view_as(g)
                delta.add_(step_alpha * g_unit)
                d_flat = delta.view(B, -1)
                d_norm = torch.norm(d_flat, dim=1, keepdim=True).clamp_min(1e-12)
                factor = torch.minimum(torch.ones_like(d_norm), (eps / d_norm))
                delta.copy_((d_flat * factor).view_as(delta))
            else:
                delta.add_(step_alpha * g.sign())
                delta.clamp_(-eps, eps)

        delta.detach_()
        delta.requires_grad_(True)

    final_grid = (base_grid + delta.detach()).clamp(-1.0, 1.0)
    x_adv = F.grid_sample(
        x_nat, final_grid, mode='bilinear',
        padding_mode=padding_mode, align_corners=grid_align_corners
    )
    return x_adv.clamp(0.0, 1.0).detach()
