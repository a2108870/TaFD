import torch


def unpack_model_outputs(outputs):
    if isinstance(outputs, tuple) and len(outputs) >= 3:
        return outputs[0], outputs[2]
    if isinstance(outputs, tuple):
        return outputs[0], None
    return outputs, None


def gate_margin_loss_maximize(gate_logits, true_gate, reduction='mean'):
    true_gate = true_gate.to(gate_logits.device).long()
    one_hot = torch.zeros_like(gate_logits)
    one_hot.scatter_(1, true_gate.view(-1, 1), 1)
    real_logit = (gate_logits * one_hot).sum(1)
    other_logits = (gate_logits - 1e9 * one_hot).max(1)[0]
    loss = other_logits - real_logit
    if reduction == 'none':
        return loss
    if reduction == 'sum':
        return loss.sum()
    return loss.mean()


def gate_margin_loss_minimize(gate_logits, true_gate, reduction='mean'):
    true_gate = true_gate.to(gate_logits.device).long()
    one_hot = torch.zeros_like(gate_logits)
    one_hot.scatter_(1, true_gate.view(-1, 1), 1)
    real_logit = (gate_logits * one_hot).sum(1)
    other_logits = (gate_logits - 1e9 * one_hot).max(1)[0]
    loss = real_logit - other_logits
    if reduction == 'none':
        return loss
    if reduction == 'sum':
        return loss.sum()
    return loss.mean()


def balanced_gate_weight(loss_cls, loss_gate, gate_loss_scale=1.0):
    cls_abs = loss_cls.detach().abs()
    gate_abs = loss_gate.detach().abs()
    if cls_abs.ndim > 0:
        cls_abs = cls_abs.mean()
    if gate_abs.ndim > 0:
        gate_abs = gate_abs.mean()
    weight = cls_abs / (gate_abs + 1e-8)
    weight = torch.clamp(weight, 0.1, 10.0)
    return weight * gate_loss_scale
