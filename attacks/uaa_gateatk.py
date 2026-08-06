import math

import kornia
import torch

from attacks.gateatk_utils import balanced_gate_weight, gate_margin_loss_minimize, unpack_model_outputs


def srgb_to_linear(x):
    mask_below = (x <= 0.04045).to(x)
    mask_above = (x > 0.04045).to(x)
    x_clamped = torch.clamp(x, min=1e-16)
    return x_clamped / 12.92 * mask_below + torch.pow((x_clamped + 0.055) / 1.055, 2.4) * mask_above


def linear_to_srgb(x):
    mask_below = (x <= 0.0031308).to(x)
    mask_above = (x > 0.0031308).to(x)
    x_clamped = torch.clamp(x, min=1e-16)
    return x_clamped * 12.92 * mask_below + (1.055 * torch.pow(x_clamped, 1 / 2.4) - 0.055) * mask_above


def GammaFilter(image, param):
    image = torch.clamp(image, min=1e-9)
    param = param.view(-1, 1, 1, 1)
    image_t = torch.pow(image, param)
    return image_t


def SaturationFilter(image, param):
    image_t = kornia.enhance.adjust_saturation(image, param)
    return image_t


def WBFilter(image, param):
    N = param.size(0)
    device = param.device
    r_scale = torch.ones(N, device=device)
    g_scale = torch.ones(N, device=device)
    b_scale = torch.exp(-param)
    color_scaling = torch.stack([r_scale, g_scale, b_scale], dim=1)
    normalization = 1.0 / (1e-5 + 0.27 * color_scaling[:, 0] + 0.67 * color_scaling[:, 1] + 0.06 * color_scaling[:, 2])
    color_scaling = color_scaling * normalization.unsqueeze(1)
    color_scaling = color_scaling.view(N, 3, 1, 1)
    image_t = image * color_scaling
    return image_t


def HueFilter(image, param):
    image_t = kornia.enhance.adjust_hue(image, param)
    return image_t


def ContrastFilter(image, param):
    torch.pi = math.pi
    contrast = param
    contrast_image = -torch.cos(torch.pi * image) * 0.5 + 0.5
    contrast = contrast.view(-1, 1, 1, 1)
    t_image = torch.lerp(image, contrast_image, contrast)
    return t_image


def CF(img, param, steps):
    batch_size, channels = param.shape[0], param.shape[1]
    param_5d = param.reshape(batch_size, channels, 1, 1, -1)
    if param_5d.shape[-1] < steps:
        raise ValueError(f"Parameter dimension too small: {param_5d.shape[-1]} < {steps}")
    color_curve_sum = torch.sum(param_5d, dim=4) + 1e-30
    step_offsets = torch.arange(steps, device=img.device, dtype=img.dtype).view(1, 1, 1, 1, -1) / steps
    img_expanded = img.unsqueeze(-1)
    clip_values = torch.clamp(img_expanded - step_offsets, 0, 1.0 / steps)
    total_image = (clip_values * param_5d).sum(dim=-1)
    total_image *= steps / color_curve_sum
    return total_image


def sharpness(img, param):
    return kornia.enhance.sharpness(img, param)


def processing_image(image_batch_ori, param, hist_param, hist_bin, device):
    image_batch = CF(image_batch_ori, hist_param, hist_bin)
    image_batch = GammaFilter(image_batch, param[:, 1])
    image_batch = SaturationFilter(image_batch, param[:, 2])
    image_batch = WBFilter(image_batch, param[:, 3])
    image_batch = ContrastFilter(image_batch, param[:, 4])
    image_batch = HueFilter(image_batch, param[:, 5])
    image_batch = sharpness(image_batch, param[:, 6])
    return image_batch


def UAA_atk(input, y, true_gate, model, device, lr=0.1, max_iterations=10, steps=64, bound=16, ncls=10,
            gate_loss_scale=1.0, norm_mean=None, norm_std=None):
    batch_size = input.shape[0]
    X = input.to(device)
    labels = y.to(device)
    true_gate = true_gate.to(device)
    labels_onehot = torch.zeros(labels.size(0), ncls, device=device)
    labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
    labels_infhot = torch.zeros_like(labels_onehot).scatter_(1, labels.unsqueeze(1), float('inf'))
    hist_bin = steps
    best_adversary = X.clone()
    eta = torch.tensor([0, 1, 1, 0, 0, 0, 1 + 10e-16]).unsqueeze(0).repeat(batch_size, 1).to(device).float()
    hist_param = torch.ones(batch_size, 3, hist_bin).to(device) * 1 / hist_bin
    X_pgd_linear = srgb_to_linear(X).detach()
    hist_param.requires_grad = True
    eta.requires_grad = True

    optimizer_eta = torch.optim.Adam([{'params': eta, 'lr': lr, 'betas': (0.9, 0.999)}])
    optimizer_hist_param = torch.optim.Adam([{'params': hist_param, 'lr': lr, 'betas': (0.9, 0.999)}])

    for _ in range(max_iterations):
        X_linear_use = X_pgd_linear.clone().detach()
        X_linear_use = processing_image(X_linear_use, eta, hist_param, hist_bin, device)
        X_srgb_eta = linear_to_srgb(X_linear_use)
        outputs = model(X_srgb_eta)
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

        if eta.grad is not None:
            eta_norm = torch.norm(eta.grad, dim=1, keepdim=True)
            eta.grad = eta.grad / (eta_norm + 1e-9)

        if hist_param.grad is not None:
            hist_param_norm = torch.norm(hist_param.grad, dim=[1, 2], keepdim=True)
            hist_param.grad = hist_param.grad / (hist_param_norm + 1e-9)

        optimizer_eta.step()
        optimizer_hist_param.step()
        optimizer_eta.zero_grad()
        optimizer_hist_param.zero_grad()

        eta.data[:, 2] = torch.clamp(eta.data[:, 2], min=0)
        eta.data[:, 3] = torch.clamp(eta.data[:, 3], min=-1, max=1)
        eta.data[:, 4] = torch.clamp(eta.data[:, 4], min=-1, max=1)
        eta.data[:, 5] = torch.clamp(eta.data[:, 5], min=-1, max=1)
        eta.data[:, 6] = torch.clamp(eta.data[:, 6], min=1)

        hist_param.data = torch.clamp(hist_param.data, min=1 / steps, max=bound / steps)
        if hist_param.shape[1] != 3 or hist_param.shape[2] != hist_bin:
            hist_param.data = hist_param.data.reshape(batch_size, 3, hist_bin)

        with torch.no_grad():
            predicted_classes = logits.argmax(1)
            is_adv = predicted_classes != labels
            if is_adv.any():
                best_adversary[is_adv] = X_srgb_eta[is_adv]

    return best_adversary
