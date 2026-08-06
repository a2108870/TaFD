import torch

from attacks.gateatk_utils import balanced_gate_weight, gate_margin_loss_minimize, unpack_model_outputs


def CF(img, param, steps):
    param = param[:, :, None, None]
    color_curve_sum = torch.sum(param, 4) + 1e-30
    total_image = img * 0
    for i in range(steps):
        total_image += torch.clamp(img - 1.0 * i / steps, 0, 1.0 / steps) * param[:, :, :, :, i]
    total_image *= steps / color_curve_sum
    return total_image


def ACE(input, y, true_gate, model, device, lr=1, max_iterations=10, steps=64, bound=16, ncls=10,
        gate_loss_scale=1.0, norm_mean=None, norm_std=None):
    model.eval()
    batch_size = input.shape[0]

    X_ori = input.to(device)
    labels = y.to(device)
    true_gate = true_gate.to(device)
    labels_onehot = torch.zeros(batch_size, ncls, device=device)
    labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
    labels_infhot = torch.zeros_like(labels_onehot).scatter_(1, labels.unsqueeze(1), float('inf'))

    Paras = torch.full((batch_size, 3, steps), 1 / steps, device=device, requires_grad=True)
    best_adversary = X_ori.clone()
    step_size = 1.0 / steps

    for _ in range(max_iterations):
        X_adv = CF(X_ori, Paras, steps)
        outputs = model(X_adv)
        logits, gate_logits = unpack_model_outputs(outputs)

        real = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        other = (logits - labels_infhot).max(1)[0]
        loss_cls = torch.clamp(real - other, min=0).sum()

        if gate_logits is not None:
            loss_gate = gate_margin_loss_minimize(gate_logits, true_gate, reduction='sum')
            weight = balanced_gate_weight(loss_cls, loss_gate, gate_loss_scale)
            loss = loss_cls + weight * loss_gate
        else:
            loss = loss_cls

        loss.backward()

        with torch.no_grad():
            grad_norms = torch.norm(Paras.grad.view(batch_size, -1), dim=1, keepdim=True).add_(1e-8)
            grad_scaled = Paras.grad.div_(grad_norms.view(-1, 1, 1))
            Paras.sub_(lr * grad_scaled)
            Paras.clamp_(min=step_size, max=step_size * bound)

        Paras.grad.zero_()

        with torch.no_grad():
            predicted_classes = logits.argmax(1)
            is_adv = predicted_classes != labels
            if is_adv.any():
                best_adversary[is_adv] = X_adv[is_adv]

    return best_adversary
