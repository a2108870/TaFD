# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict




@torch.no_grad()
def build_pca_bases(trainloader,
                    n_classes: int,
                    rank: int = 128,
                    max_per_class: int = 600,
                    device: torch.device = torch.device('cpu')) -> Dict[int, torch.Tensor]:
    """
    Build PCA bases from training data for each class.
    """
    per_class = [[] for _ in range(n_classes)]
    total_needed = [max_per_class] * n_classes

    for imgs, labels in trainloader:
        for c in labels.unique().tolist():
            c = int(c)
            if total_needed[c] <= 0:
                continue
        for x, y in zip(imgs, labels):
            c = int(y.item())
            if total_needed[c] > 0:
                per_class[c].append(x.reshape(-1).float().cpu())
                total_needed[c] -= 1
        if all(k <= 0 for k in total_needed):
            break

    bases = {}
    for c in range(n_classes):
        X_list = per_class[c]
        if len(X_list) == 0:
            D = imgs[0].numel()
            q = min(rank, D)
            bases[c] = torch.eye(D, q)
            continue
        X = torch.stack(X_list, dim=0)
        X = X - X.mean(dim=0, keepdim=True)
        Nc, D = X.shape
        q = min(rank, Nc, D)
        U, S, V = torch.pca_lowrank(X, q=q, center=False)
        bases[c] = V[:, :q].contiguous().float()

    return bases


def save_pca_bases(path: str, bases: Dict[int, torch.Tensor]) -> None:
    cpu_bases = {k: v.cpu() for k, v in bases.items()}
    torch.save(cpu_bases, path)


def load_pca_bases(path: str, device: torch.device) -> Dict[int, torch.Tensor]:
    bases = torch.load(path, map_location='cpu')
    return {int(k): v.contiguous().float() for k, v in bases.items()}


@torch.no_grad()
def _stack_class_bases(bases_dict, labels, device):
    A_list = [bases_dict[int(lbl.item())].to(device) for lbl in labels]
    qs = [A.shape[1] for A in A_list]
    q = min(qs)
    A_list = [A[:, :q] for A in A_list]
    A = torch.stack(A_list, dim=0)
    return A, q


def subspace_atk(input, y, model, device, bases_dict,
                 steps: int = 10,
                 epsilon: float = 2.0,
                 step_size: float = None,
                 ncls: int = 100,
                 proj: str = 'l2',
                 norm_mean=None, norm_std=None):
    """
    Subspace on-manifold attack using PCA bases.
    Note: Model handles normalization internally, so input should be in [0,1] pixel domain.
    norm_mean and norm_std parameters are kept for API compatibility but not used.
    """
    model.eval()
    x_nat = input.to(device)
    labels = y.to(device)
    B, C, H, W = x_nat.shape
    D = C * H * W

    A, q = _stack_class_bases(bases_dict, labels, device)
    z = torch.zeros(B, q, device=device, requires_grad=True)

    if step_size is None:
        step_size = epsilon * 1.5 / max(1, steps)

    for _ in range(steps):
        x_adv = (x_nat.view(B, -1) + torch.bmm(A, z.unsqueeze(-1)).squeeze(-1)).view_as(x_nat)
        x_adv = x_adv.clamp(0.0, 1.0)

        # Forward pass (model handles normalization internally)
        logits = model(x_adv)[0]
        loss = F.cross_entropy(logits, labels)
        loss.backward()

        with torch.no_grad():
            g = z.grad
            if proj == 'l2':
                g = g / (g.norm(dim=1, keepdim=True) + 1e-8)
                z.add_(step_size * g)
                zn = z.norm(dim=1, keepdim=True).clamp(min=1e-8)
                z.mul_(torch.minimum(torch.ones_like(zn), torch.full_like(zn, epsilon) / zn))
            else:
                z.add_(step_size * g.sign())
                z.clamp_(-epsilon, epsilon)
        z.grad.zero_()

    x_adv = (x_nat.view(B, -1) + torch.bmm(A, z.unsqueeze(-1)).squeeze(-1)).view_as(x_nat)
    return x_adv.clamp(0.0, 1.0).detach()
