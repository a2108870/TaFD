import kornia as K
import torch

from attacks.adaptive_diagnosis_utils import balanced_diagnosis_weight, diagnosis_margin_loss_minimize, unpack_model_outputs


def apply_lab_lightness_curve(img, param, steps):
    lab_img = K.color.rgb_to_lab(img)
    lab_img[:, 0:1, :, :] = lab_img[:, 0:1, :, :] / 100

    param = param[:, :, None, None]
    color_curve_sum = torch.sum(param, 4) + 1e-30
    step_values = torch.linspace(0, 1, steps, device=img.device)
    lab_expanded = lab_img[:, 0:1, :, :, None]
    step_values = step_values.view(1, 1, 1, 1, -1)
    differences = lab_expanded - step_values
    clamped_differences = torch.clamp(differences, 0, 1.0 / steps)
    weighted_values = clamped_differences * param

    total_image = lab_img.clone()
    total_image[:, 0:1, :, :] = (torch.sum(weighted_values, dim=4) * (steps / color_curve_sum)) * 100
    total_image = K.color.lab_to_rgb(total_image)
    total_image = torch.clamp(total_image, 0, 1.0)
    return total_image


def ala_attack(input, y, target_threat_domain_indices, model, device, lr=1, max_iterations=10, steps=64, bound=16, num_classes=10,
              diagnosis_loss_scale=1.0, norm_mean=None, norm_std=None):
    batch_size = input.shape[0]

    X_ori = input.to(device)
    labels = y.to(device)
    target_threat_domain_indices = target_threat_domain_indices.to(device)
    labels_onehot = torch.zeros(labels.size(0), num_classes, device=device)
    labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
    labels_infhot = torch.zeros_like(labels_onehot).scatter_(1, labels.unsqueeze(1), float('inf'))

    Paras = torch.rand(batch_size, 1, steps, device=device) * 1.0 - 0.2
    Paras.requires_grad = True
    best_adversary = X_ori.clone()

    for _ in range(max_iterations):
        X_adv = apply_lab_lightness_curve(X_ori, Paras, steps)
        outputs = model(X_adv)
        logits, diagnosis_logits = unpack_model_outputs(outputs)
        real = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        other = (logits - labels_infhot).max(1)[0]
        loss_cls = torch.clamp(real - other, min=0).sum()

        if diagnosis_logits is not None:
            diagnosis_loss = diagnosis_margin_loss_minimize(diagnosis_logits, target_threat_domain_indices, reduction='sum')
            weight = balanced_diagnosis_weight(loss_cls, diagnosis_loss, diagnosis_loss_scale)
            loss = loss_cls + weight * diagnosis_loss
        else:
            loss = loss_cls

        loss.backward()
        grad_a = Paras.grad.clone()

        Paras.data = Paras.data - lr * (grad_a.permute(1, 2, 0) / (
                torch.norm(grad_a.view(batch_size, -1), dim=1) + 1e-8)).permute(2, 0, 1)
        Paras.grad.zero_()
        Paras.data = torch.clamp(Paras.data, min=1 / steps, max=1 / steps * bound)

        with torch.no_grad():
            predicted_classes = logits.argmax(1)
            is_adv = predicted_classes != labels
            if is_adv.any():
                best_adversary[is_adv] = X_adv[is_adv]

    return best_adversary
