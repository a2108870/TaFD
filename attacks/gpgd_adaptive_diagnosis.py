import torch
import torch.nn.functional as F

from attacks.adaptive_diagnosis_utils import balanced_diagnosis_weight, diagnosis_margin_loss_maximize, unpack_model_outputs


@torch.no_grad()
def _stack_class_gpgd_bases(bases_dict, labels, device):
    basis_list = [bases_dict[int(lbl.item())].to(device) for lbl in labels]
    ranks = [basis.shape[1] for basis in basis_list]
    q = min(ranks)
    basis_list = [basis[:, :q] for basis in basis_list]
    stacked = torch.stack(basis_list, dim=0)
    return stacked, q


def gpgd_attack(input, y, target_threat_domain_indices, model, device, bases_dict,
                 steps: int = 10,
                 epsilon: float = 2.0,
                 step_size: float = None,
                 num_classes: int = 100,
                 proj: str = 'l2',
                 diagnosis_loss_scale: float = 1.0,
                 norm_mean=None, norm_std=None):
    model.eval()
    x_nat = input.to(device)
    labels = y.to(device)
    target_threat_domain_indices = target_threat_domain_indices.to(device)
    batch_size = x_nat.shape[0]

    basis, rank = _stack_class_gpgd_bases(bases_dict, labels, device)
    z = torch.zeros(batch_size, rank, device=device, requires_grad=True)

    if step_size is None:
        step_size = epsilon * 1.5 / max(1, steps)

    for _ in range(steps):
        x_adv = (x_nat.view(batch_size, -1) + torch.bmm(basis, z.unsqueeze(-1)).squeeze(-1)).view_as(x_nat)
        x_adv = x_adv.clamp(0.0, 1.0)

        outputs = model(x_adv)
        logits, diagnosis_logits = unpack_model_outputs(outputs)
        loss_cls = F.cross_entropy(logits, labels)

        if diagnosis_logits is not None:
            diagnosis_loss = diagnosis_margin_loss_maximize(diagnosis_logits, target_threat_domain_indices, reduction='mean')
            weight = balanced_diagnosis_weight(loss_cls, diagnosis_loss, diagnosis_loss_scale)
            loss = loss_cls + weight * diagnosis_loss
        else:
            loss = loss_cls

        loss.backward()

        with torch.no_grad():
            grad = z.grad
            if proj == 'l2':
                grad = grad / (grad.norm(dim=1, keepdim=True) + 1e-8)
                z.add_(step_size * grad)
                z_norm = z.norm(dim=1, keepdim=True).clamp(min=1e-8)
                z.mul_(torch.minimum(torch.ones_like(z_norm), torch.full_like(z_norm, epsilon) / z_norm))
            else:
                z.add_(step_size * grad.sign())
                z.clamp_(-epsilon, epsilon)
        z.grad.zero_()

    x_adv = (x_nat.view(batch_size, -1) + torch.bmm(basis, z.unsqueeze(-1)).squeeze(-1)).view_as(x_nat)
    return x_adv.clamp(0.0, 1.0).detach()
