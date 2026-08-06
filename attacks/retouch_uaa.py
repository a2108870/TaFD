import torch
import kornia
import math


def srgb_to_linear(x):
    """sRGB gamma decompression (inverse gamma): sRGB -> linear RGB. Vectorized over batch."""
    mask_below = (x <= 0.04045).to(x)
    mask_above = (x > 0.04045).to(x)
    x_clamped = torch.clamp(x, min=1e-16)
    return x_clamped / 12.92 * mask_below + torch.pow((x_clamped + 0.055) / 1.055, 2.4) * mask_above


def linear_to_srgb(x):
    """sRGB gamma compression: linear RGB -> sRGB. Vectorized over batch."""
    mask_below = (x <= 0.0031308).to(x)
    mask_above = (x > 0.0031308).to(x)
    x_clamped = torch.clamp(x, min=1e-16)
    return x_clamped * 12.92 * mask_below + (1.055 * torch.pow(x_clamped, 1 / 2.4) - 0.055) * mask_above


RETOUCH_FILTER_NAMES = (
    'exposure', 'gamma', 'saturation', 'white_balance',
    'contrast', 'black_and_white', 'tone', 'color',
)


def exposure_filter(image, param):
    image_t = image * (2 ** param)
    return image_t


def gamma_filter(image, param):
    image = torch.clamp(image, min=1e-9)
    param = param.view(-1, 1, 1, 1)
    image_t = torch.pow(image, param)
    return image_t


def saturation_filter(image, param):
    image_t = kornia.enhance.adjust_saturation(image, param)
    return image_t


def white_balance_filter(image, param):
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


def hue_filter(image, param):
    image_t = kornia.enhance.adjust_hue(image, param)
    return image_t


def contrast_filter(image, param):
    torch.pi = math.pi
    contrast = param
    contrast_image = -torch.cos(torch.pi * image) * 0.5 + 0.5
    contrast = contrast.view(-1, 1, 1, 1)
    t_image = torch.lerp(image, contrast_image, contrast)
    return t_image


def apply_color_curve(img, param, steps):
    batch_size, channels = param.shape[0], param.shape[1]
    param_5d = param.reshape(batch_size, channels, 1, 1, -1)

    if param_5d.shape[-1] < steps:
        raise ValueError(f"Parameter dimension too small: {param_5d.shape[-1]} < {steps}")

    color_curve_sum = torch.sum(param_5d, dim=4) + 1e-30

    # Vectorized: compute all steps at once
    step_offsets = torch.arange(steps, device=img.device, dtype=img.dtype).view(1, 1, 1, 1, -1) / steps
    img_expanded = img.unsqueeze(-1)  # [B, C, H, W, 1]
    clip_values = torch.clamp(img_expanded - step_offsets, 0, 1.0 / steps)  # [B, C, H, W, steps]
    total_image = (clip_values * param_5d).sum(dim=-1)  # [B, C, H, W]

    total_image *= steps / color_curve_sum
    return total_image


def sharpness_filter(img, param):
    img_t = kornia.enhance.sharpness(img, param)
    return img_t


def apply_retouch_pipeline(image_batch_ori, param, hist_param, hist_bin):
    image_batch = apply_color_curve(image_batch_ori, hist_param, hist_bin)
    image_batch = gamma_filter(image_batch, param[:, 1])
    image_batch = saturation_filter(image_batch, param[:, 2])
    image_batch = white_balance_filter(image_batch, param[:, 3])
    image_batch = contrast_filter(image_batch, param[:, 4])
    image_batch = hue_filter(image_batch, param[:, 5])
    image_batch = sharpness_filter(image_batch, param[:, 6])
    return image_batch


def retouch_uaa_attack(input, y, model, device, lr=0.01, max_iterations=10, steps=64, bound=16, num_classes=10,
            norm_mean=None, norm_std=None):
    """
    Generate RetouchUAA examples.
    Note: Model handles normalization internally, so input should be in [0,1] pixel domain.
    norm_mean and norm_std parameters are kept for API compatibility but not used.
    """
    batch_size = input.shape[0]
    X, labels = input.to(device), y.to(device)
    labels_onehot = torch.zeros(labels.size(0), num_classes, device=device)
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

    for i in range(max_iterations):
        X_linear_use = X_pgd_linear.clone().detach()
        X_linear_use = apply_retouch_pipeline(X_linear_use, eta, hist_param, hist_bin)
        X_srgb_eta = linear_to_srgb(X_linear_use)
        # Forward pass (model handles normalization internally)
        output = model(X_srgb_eta)[0]

        real = output.gather(1, labels.unsqueeze(1)).squeeze(1)
        other = (output - labels_infhot).max(1)[0]
        loss = torch.clamp(real - other, min=0).sum()

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
            predicted_classes = output.argmax(1)
            is_adv = (predicted_classes != labels)

            if is_adv.any():
                best_adversary[is_adv] = X_srgb_eta[is_adv]

    return best_adversary
