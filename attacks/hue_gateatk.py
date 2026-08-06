import math

import kornia as K
import torch

from attacks.gateatk_utils import balanced_gate_weight, gate_margin_loss_minimize, unpack_model_outputs


def CF_HSV(img, param_h, param_s, param_v, steps):
    HSV_img = K.color.rgb_to_hsv(img)

    Hue = HSV_img[:, 0:1, :, :] / (2 * math.pi)
    param_h = param_h[:, :, None, None]
    color_curve_sum_h = torch.sum(param_h, 4) + 1e-30
    total_Hue = Hue * 0

    Saturation = HSV_img[:, 1:2, :, :]
    param_s = param_s[:, :, None, None]
    color_curve_sum_s = torch.sum(param_s, 4) + 1e-30
    total_Saturation = Saturation * 0

    Value = HSV_img[:, 2:3, :, :]
    param_v = param_v[:, :, None, None]
    color_curve_sum_v = torch.sum(param_v, 4) + 1e-30
    total_Value = Value * 0

    for i in range(steps):
        total_Hue += torch.clamp(Hue - 1.0 * i / steps, 0, 1.0 / steps) * param_h[:, :, :, :, i]
        total_Saturation += torch.clamp(Saturation - 1.0 * i / steps, 0, 1.0 / steps) * param_s[:, :, :, :, i]
        total_Value += torch.clamp(Value - 1.0 * i / steps, 0, 1.0 / steps) * param_v[:, :, :, :, i]

    HSV_img[:, 0:1, :, :] = (total_Hue * steps / color_curve_sum_h) * (2 * math.pi)
    HSV_img[:, 1:2, :, :] = total_Saturation * steps / color_curve_sum_s
    HSV_img[:, 2:3, :, :] = total_Value * steps / color_curve_sum_v

    img = K.color.hsv_to_rgb(HSV_img)
    img = torch.clamp(img, 0, 1.0)
    return img


def hue_atk(input, y, true_gate, model, device, lr=1, max_iterations=10, steps=64, bound=16, ncls=10,
            gate_loss_scale=1.0, norm_mean=None, norm_std=None):
    batch_size = input.shape[0]

    X_ori = input.to(device)
    labels = y.to(device)
    true_gate = true_gate.to(device)
    labels_onehot = torch.zeros(labels.size(0), ncls, device=device)
    labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
    labels_infhot = torch.zeros_like(labels_onehot).scatter_(1, labels.unsqueeze(1), float('inf'))

    Paras_h = torch.full((batch_size, 1, steps), 1 / steps, device=device, requires_grad=True)
    Paras_s = torch.full((batch_size, 1, steps), 1 / steps, device=device, requires_grad=True)
    Paras_v = torch.full((batch_size, 1, steps), 1 / steps, device=device, requires_grad=True)
    best_adversary = X_ori.clone()

    for _ in range(max_iterations):
        X_adv = CF_HSV(X_ori, Paras_h, Paras_s, Paras_v, steps)
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

        for Paras in [Paras_h, Paras_s, Paras_v]:
            grad = Paras.grad.clone()
            Paras.data = Paras.data - lr * (grad.permute(1, 2, 0) / (
                    torch.norm(grad.view(batch_size, -1), dim=1) + 1e-8)).permute(2, 0, 1)
            Paras.grad.zero_()
            Paras.data = torch.clamp(Paras.data, min=1 / steps, max=1 / steps * bound)

        with torch.no_grad():
            predicted_classes = logits.argmax(1)
            is_adv = predicted_classes != labels
            if is_adv.any():
                best_adversary[is_adv] = X_adv[is_adv]

    return best_adversary
