import torch
import numpy as np
import torch.nn as nn
import kornia as K
import math
import matplotlib.pyplot as plt


def CF_HSV(img, param_h, param_s, param_v, steps):  # color filter for HSV
    HSV_img = K.color.rgb_to_hsv(img)

    # Process Hue
    Hue = HSV_img[:, 0:1, :, :] / (2 * math.pi)  # Normalize H to 0..1
    param_h = param_h[:, :, None, None]
    color_curve_sum_h = torch.sum(param_h, 4) + 1e-30
    total_Hue = Hue * 0

    # Process Saturation
    Saturation = HSV_img[:, 1:2, :, :]  # S already in 0..1
    param_s = param_s[:, :, None, None]
    color_curve_sum_s = torch.sum(param_s, 4) + 1e-30
    total_Saturation = Saturation * 0

    # Process Value
    Value = HSV_img[:, 2:3, :, :]  # V already in 0..1
    param_v = param_v[:, :, None, None]
    color_curve_sum_v = torch.sum(param_v, 4) + 1e-30
    total_Value = Value * 0

    # Combined processing loop for all channels
    for i in range(steps):
        total_Hue += torch.clamp(Hue - 1.0 * i / steps, 0, 1.0 / steps) * param_h[:, :, :, :, i]
        total_Saturation += torch.clamp(Saturation - 1.0 * i / steps, 0, 1.0 / steps) * param_s[:, :, :, :, i]
        total_Value += torch.clamp(Value - 1.0 * i / steps, 0, 1.0 / steps) * param_v[:, :, :, :, i]

    # Apply normalization and update HSV channels
    HSV_img[:, 0:1, :, :] = (total_Hue * steps / color_curve_sum_h) * (2 * math.pi)  # Denormalize H
    HSV_img[:, 1:2, :, :] = total_Saturation * steps / color_curve_sum_s
    HSV_img[:, 2:3, :, :] = total_Value * steps / color_curve_sum_v

    # Convert back to RGB and clamp
    img = K.color.hsv_to_rgb(HSV_img)
    img = torch.clamp(img, 0, 1.0)
    return img


def hue_atk(input, y, model, device, lr=1, max_iterations=10, steps=64, bound=16, ncls=10,
            norm_mean=None, norm_std=None):
    """
    Hue attack function.
    Note: Model handles normalization internally, so input should be in [0,1] pixel domain.
    norm_mean and norm_std parameters are kept for API compatibility but not used.
    """
    batch_size = input.shape[0]

    X_ori, labels = input.to(device), y.to(device)

    labels_onehot = torch.zeros(labels.size(0), ncls, device=device)
    labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
    labels_infhot = torch.zeros_like(labels_onehot).scatter_(1, labels.unsqueeze(1), float('inf'))

    # Initialize parameters for H, S, V channels
    Paras_h = torch.full((batch_size, 1, steps), 1 / steps, device=device, requires_grad=True)
    Paras_s = torch.full((batch_size, 1, steps), 1 / steps, device=device, requires_grad=True)
    Paras_v = torch.full((batch_size, 1, steps), 1 / steps, device=device, requires_grad=True)

    best_adversary = X_ori.clone()

    for iteration in range(max_iterations):
        # Apply the HSV color filter
        X_adv = CF_HSV(X_ori, Paras_h, Paras_s, Paras_v, steps)

        # Forward pass (model handles normalization internally)
        logits = model(X_adv)[0]

        real = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        other = (logits - labels_infhot).max(1)[0]
        loss = torch.clamp(real - other, min=0).sum()

        loss.backward()

        # Update parameters for all channels in a single loop
        for Paras in [Paras_h, Paras_s, Paras_v]:
            grad = Paras.grad.clone()
            Paras.data = Paras.data - lr * (grad.permute(1, 2, 0) / (
                    torch.norm(grad.view(batch_size, -1), dim=1) + 1e-8)).permute(2, 0, 1)
            Paras.grad.zero_()
            Paras.data = torch.clamp(Paras.data, min=1 / steps, max=1 / steps * bound)

        with torch.no_grad():
            predicted_classes = logits.argmax(1)
            is_adv = (predicted_classes != labels)

            if is_adv.any():
                best_adversary[is_adv] = X_adv[is_adv]

    return best_adversary


def visualize_hsv_curves(params_h, params_s, params_v, steps, sample_idx=0, iteration=None):
    """
    Visualize the HSV transformation curves for a specific sample
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    x_values = np.linspace(0, 1, steps)

    curve_h = params_h[sample_idx, 0, :].detach().cpu().numpy()
    cumulative_h = np.cumsum(curve_h)

    curve_s = params_s[sample_idx, 0, :].detach().cpu().numpy()
    cumulative_s = np.cumsum(curve_s)

    curve_v = params_v[sample_idx, 0, :].detach().cpu().numpy()
    cumulative_v = np.cumsum(curve_v)

    axs[0].plot(x_values, curve_h, 'b-', linewidth=2, label='Hue weights')
    axs[0].plot(x_values, cumulative_h, 'r-', linewidth=2, label='Cumulative')
    axs[0].set_xlabel('Hue value (normalized)', fontsize=12)
    axs[0].set_ylabel('Transform intensity', fontsize=12)
    axs[0].set_title('Hue Transform Curve', fontsize=14)
    axs[0].legend(fontsize=12)
    axs[0].grid(True)

    axs[1].plot(x_values, curve_s, 'b-', linewidth=2, label='Saturation weights')
    axs[1].plot(x_values, cumulative_s, 'r-', linewidth=2, label='Cumulative')
    axs[1].set_xlabel('Saturation value', fontsize=12)
    axs[1].set_ylabel('Transform intensity', fontsize=12)
    axs[1].set_title('Saturation Transform Curve', fontsize=14)
    axs[1].legend(fontsize=12)
    axs[1].grid(True)

    axs[2].plot(x_values, curve_v, 'b-', linewidth=2, label='Value weights')
    axs[2].plot(x_values, cumulative_v, 'r-', linewidth=2, label='Cumulative')
    axs[2].set_xlabel('Value', fontsize=12)
    axs[2].set_ylabel('Transform intensity', fontsize=12)
    axs[2].set_title('Value Transform Curve', fontsize=14)
    axs[2].legend(fontsize=12)
    axs[2].grid(True)

    main_title = 'HSV Transform Curves'
    if iteration is not None:
        main_title += f' (Iteration {iteration})'
    fig.suptitle(main_title, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
