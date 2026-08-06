import torch


def unpack_model_outputs(outputs):
    if isinstance(outputs, tuple) and len(outputs) >= 3:
        return outputs[0], outputs[2]
    if isinstance(outputs, tuple):
        return outputs[0], None
    return outputs, None


def diagnosis_margin_loss_maximize(diagnosis_logits, target_threat_domain_indices, reduction='mean'):
    target_threat_domain_indices = target_threat_domain_indices.to(diagnosis_logits.device).long()
    one_hot = torch.zeros_like(diagnosis_logits)
    one_hot.scatter_(1, target_threat_domain_indices.view(-1, 1), 1)
    real_logit = (diagnosis_logits * one_hot).sum(1)
    other_logits = (diagnosis_logits - 1e9 * one_hot).max(1)[0]
    loss = other_logits - real_logit
    if reduction == 'none':
        return loss
    if reduction == 'sum':
        return loss.sum()
    return loss.mean()


def diagnosis_margin_loss_minimize(diagnosis_logits, target_threat_domain_indices, reduction='mean'):
    target_threat_domain_indices = target_threat_domain_indices.to(diagnosis_logits.device).long()
    one_hot = torch.zeros_like(diagnosis_logits)
    one_hot.scatter_(1, target_threat_domain_indices.view(-1, 1), 1)
    real_logit = (diagnosis_logits * one_hot).sum(1)
    other_logits = (diagnosis_logits - 1e9 * one_hot).max(1)[0]
    loss = real_logit - other_logits
    if reduction == 'none':
        return loss
    if reduction == 'sum':
        return loss.sum()
    return loss.mean()


def balanced_diagnosis_weight(classification_loss, diagnosis_loss, diagnosis_loss_scale=1.0):
    classification_magnitude = classification_loss.detach().abs()
    diagnosis_magnitude = diagnosis_loss.detach().abs()
    if classification_magnitude.ndim > 0:
        classification_magnitude = classification_magnitude.mean()
    if diagnosis_magnitude.ndim > 0:
        diagnosis_magnitude = diagnosis_magnitude.mean()
    weight = classification_magnitude / (diagnosis_magnitude + 1e-8)
    weight = torch.clamp(weight, 0.1, 10.0)
    return weight * diagnosis_loss_scale
