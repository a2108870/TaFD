import torch


def CF(img, param, steps):  # color filter

    param = param[:, :, None, None]
    color_curve_sum = torch.sum(param, 4) + 1e-30
    total_image = img * 0
    for i in range(steps):
        total_image += torch.clamp(img - 1.0 * i / steps, 0, 1.0 / steps) * param[:, :, :, :, i]
    total_image *= steps / color_curve_sum
    return total_image


def ACE(input, y, model, device, lr=1, max_iterations=10, steps=64, bound=16, ncls=10,
        norm_mean=None, norm_std=None):
    """
    ACE attack function.
    Note: Model handles normalization internally, so input should be in [0,1] pixel domain.
    norm_mean and norm_std parameters are kept for API compatibility but not used.
    """
    model.eval()
    batch_size = input.shape[0]

    X_ori = input.to(device)
    labels = y.to(device)
    labels_onehot = torch.zeros(batch_size, ncls, device=device)
    labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
    labels_infhot = torch.zeros_like(labels_onehot).scatter_(1, labels.unsqueeze(1), float('inf'))

    Paras = torch.full((batch_size, 3, steps), 1 / steps, device=device, requires_grad=True)
    best_adversary = X_ori.clone()

    norm_const = 1.0 / (steps * bound)
    step_size = 1.0 / steps

    for iteration in range(max_iterations):
        X_adv = CF(X_ori, Paras, steps)

        logits = model(X_adv)[0]

        real = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        other = (logits - labels_infhot).max(1)[0]
        loss = torch.clamp(real - other, min=0).sum()

        loss.backward()

        with torch.no_grad():
            grad_norms = torch.norm(Paras.grad.view(batch_size, -1), dim=1, keepdim=True).add_(1e-8)
            grad_scaled = Paras.grad.div_(grad_norms.view(-1, 1, 1))
            Paras.sub_(lr * grad_scaled)

            Paras.clamp_(min=step_size, max=step_size * bound)

        Paras.grad.zero_()

        with torch.no_grad():
            predicted_classes = logits.argmax(1)
            is_adv = (predicted_classes != labels)

            if is_adv.any():
                best_adversary[is_adv] = X_adv[is_adv]

    return best_adversary
