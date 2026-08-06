import torch
import kornia as K


def apply_lab_lightness_curve(img, param, steps):
    # Convert to LAB color space
    lab_img = K.color.rgb_to_lab(img)
    lab_img[:, 0:1, :, :] = lab_img[:, 0:1, :, :] / 100

    # Reshape param for broadcasting
    param = param[:, :, None, None]  # [B, C, 1, 1, steps]

    # Calculate color curve sum
    color_curve_sum = torch.sum(param, 4) + 1e-30  # [B, C, 1, 1]

    # Create steps tensor
    step_values = torch.linspace(0, 1, steps, device=img.device)  # [steps]

    # Expand dimensions for broadcasting
    lab_expanded = lab_img[:, 0:1, :, :, None]  # [B, 1, H, W, 1]
    step_values = step_values.view(1, 1, 1, 1, -1)  # [1, 1, 1, 1, steps]

    # Vectorized computation
    differences = lab_expanded - step_values  # [B, 1, H, W, steps]
    clamped_differences = torch.clamp(differences, 0, 1.0 / steps)  # [B, 1, H, W, steps]
    weighted_values = clamped_differences * param  # [B, C, H, W, steps]

    # Sum along steps dimension
    total_image = lab_img.clone()
    total_image[:, 0:1, :, :] = (torch.sum(weighted_values, dim=4) * (steps / color_curve_sum)) * 100

    # Convert back to RGB
    total_image = K.color.lab_to_rgb(total_image)
    total_image = torch.clamp(total_image, 0, 1.0)

    return total_image


def ala_attack(input, y, model, device, lr=1, max_iterations=10, steps=64, bound=16, num_classes=10,
              norm_mean=None, norm_std=None):
    """
    Generate ALA examples.
    Note: Model handles normalization internally, so input should be in [0,1] pixel domain.
    norm_mean and norm_std parameters are kept for API compatibility but not used.
    """
    batch_size = input.shape[0]

    X_ori, labels = input.to(device), y.to(device)

    labels_onehot = torch.zeros(labels.size(0), num_classes, device=device)
    labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
    labels_infhot = torch.zeros_like(labels_onehot).scatter_(1, labels.unsqueeze(1), float('inf'))

    m = -0.2  # min value
    n = 0.8   # max value
    Paras = torch.rand(batch_size, 1, steps, device=device) * (n - m) + m
    Paras.requires_grad = True
    best_adversary = X_ori.clone()

    for iteration in range(max_iterations):
        X_adv = apply_lab_lightness_curve(X_ori, Paras, steps)

        # Forward pass (model handles normalization internally)
        logits = model(X_adv)[0]
        real = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        other = (logits - labels_infhot).max(1)[0]
        loss = torch.clamp(real - other, min=0).sum()

        loss.backward()
        grad_a = Paras.grad.clone()

        Paras.data = Paras.data - lr * (grad_a.permute(1, 2, 0) / (
                torch.norm(grad_a.view(batch_size, -1), dim=1) + 1e-8)).permute(2, 0, 1)
        Paras.grad.zero_()
        Paras.data = torch.clamp(Paras.data, min=1 / steps, max=1 / steps * bound)

        with torch.no_grad():
            predicted_classes = logits.argmax(1)
            is_adv = (predicted_classes != labels)

            if is_adv.any():
                best_adversary[is_adv] = X_adv[is_adv]

    return best_adversary
