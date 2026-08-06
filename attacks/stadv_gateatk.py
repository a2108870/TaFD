from typing import Optional

import torch
import torch.nn.functional as F

from attacks.gateatk_utils import balanced_gate_weight, gate_margin_loss_maximize, unpack_model_outputs


def _make_base_grid(batch_size: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    base = torch.stack([grid_x, grid_y], dim=-1)
    return base.unsqueeze(0).expand(batch_size, height, width, 2).contiguous()


def stadv_attack(
    img: torch.Tensor,
    y: torch.Tensor,
    true_gate: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
    eps: float = 0.045,
    steps: int = 10,
    alpha: Optional[float] = None,
    mode: str = 'linf',
    padding_mode: str = 'border',
    grid_align_corners: bool = True,
    gate_loss_scale: float = 1.0,
    norm_mean=None,
    norm_std=None,
) -> torch.Tensor:
    model.eval()

    x_nat = img.detach().to(device)
    labels = y.detach().to(device)
    true_gate = true_gate.detach().to(device)
    batch_size, _, height, width = x_nat.shape
    dtype = x_nat.dtype

    base_grid = _make_base_grid(batch_size, height, width, device, dtype)
    delta = torch.zeros_like(base_grid, requires_grad=True)

    step_alpha = float(eps * 1.5 / max(1, steps)) if alpha is None else float(alpha)

    for _ in range(int(steps)):
        grid = (base_grid + delta).clamp(-1.0, 1.0)
        x_warp = F.grid_sample(x_nat, grid, mode='bilinear', padding_mode=padding_mode, align_corners=grid_align_corners)

        outputs = model(x_warp)
        logits, gate_logits = unpack_model_outputs(outputs)
        loss_cls = F.cross_entropy(logits, labels)

        if gate_logits is not None:
            loss_gate = gate_margin_loss_maximize(gate_logits, true_gate, reduction='mean')
            weight = balanced_gate_weight(loss_cls, loss_gate, gate_loss_scale)
            loss = loss_cls + weight * loss_gate
        else:
            loss = loss_cls

        if delta.grad is not None:
            delta.grad.detach_()
            delta.grad.zero_()
        loss.backward()

        with torch.no_grad():
            grad = delta.grad
            if grad is None:
                break

            if mode.lower() == 'l2':
                grad_flat = grad.view(batch_size, -1)
                grad_norm = torch.norm(grad_flat, dim=1, keepdim=True).clamp_min(1e-12)
                grad_unit = (grad_flat / grad_norm).view_as(grad)
                delta.add_(step_alpha * grad_unit)
                delta_flat = delta.view(batch_size, -1)
                delta_norm = torch.norm(delta_flat, dim=1, keepdim=True).clamp_min(1e-12)
                factor = torch.minimum(torch.ones_like(delta_norm), (eps / delta_norm))
                delta.copy_((delta_flat * factor).view_as(delta))
            else:
                delta.add_(step_alpha * grad.sign())
                delta.clamp_(-eps, eps)

        delta.detach_()
        delta.requires_grad_(True)

    final_grid = (base_grid + delta.detach()).clamp(-1.0, 1.0)
    x_adv = F.grid_sample(x_nat, final_grid, mode='bilinear', padding_mode=padding_mode, align_corners=grid_align_corners)
    return x_adv.clamp(0.0, 1.0).detach()
