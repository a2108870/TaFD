import torch
from torch import optim

from recoloradv.mister_ed import adversarial_attacks as aa
from recoloradv.mister_ed import adversarial_perturbations as ap
from recoloradv.mister_ed import adversarial_training as advtrain
from recoloradv.mister_ed import loss_functions as lf

from recoloradv import color_spaces as cs
from recoloradv import color_transformers as ct
from recoloradv import perturbations as pt

from attacks.adaptive_diagnosis_utils import balanced_diagnosis_weight, diagnosis_margin_loss_minimize, unpack_model_outputs


class IdentityNormalizer:
    def __call__(self, x):
        return x

    def forward(self, x):
        return x

    def differentiable_call(self):
        return self

    def nondifferentiable_call(self):
        return self


class JointClassificationDiagnosisCWMarginLoss(lf.PartialLoss):
    def __init__(self, classifier, normalizer=None, kappa=0.0, target_threat_domain_indices=None, diagnosis_loss_scale=1.0):
        super().__init__()
        self.classifier = classifier
        self.normalizer = normalizer
        self.kappa = kappa
        self.target_threat_domain_indices = target_threat_domain_indices
        self.diagnosis_loss_scale = diagnosis_loss_scale
        self.nets.append(self.classifier)

    def forward(self, examples, labels, *args, **kwargs):
        classifier_in = self.normalizer.forward(examples) if self.normalizer is not None else examples
        outputs = self.classifier.forward(classifier_in)
        classifier_out, diagnosis_logits = unpack_model_outputs(outputs)

        target_logits = torch.gather(classifier_out, 1, labels.view(-1, 1))
        max_2_logits, argmax_2_logits = torch.topk(classifier_out, 2, dim=1)
        top_max, second_max = max_2_logits.chunk(2, dim=1)
        top_argmax, _ = argmax_2_logits.chunk(2, dim=1)
        targets_eq_max = top_argmax.squeeze(1).eq(labels).float().view(-1, 1)
        targets_ne_max = top_argmax.squeeze(1).ne(labels).float().view(-1, 1)
        max_other = targets_eq_max * second_max + targets_ne_max * top_max
        loss_cls = torch.clamp(target_logits - max_other, min=-1 * self.kappa).squeeze(1)

        if diagnosis_logits is not None and self.target_threat_domain_indices is not None:
            diagnosis_loss = diagnosis_margin_loss_minimize(diagnosis_logits, self.target_threat_domain_indices, reduction='none')
            if self.kappa != float('inf'):
                diagnosis_loss = torch.clamp(diagnosis_loss, min=-1 * self.kappa)
            weight = balanced_diagnosis_weight(loss_cls, diagnosis_loss, self.diagnosis_loss_scale)
            return loss_cls + weight * diagnosis_loss

        return loss_cls


def recoloradv_adaptive_diagnosis_attack(inputs, labels, target_threat_domain_indices, model, device,
                                         max_iterations=10, lr=0.01, num_classes=10,
                                         diagnosis_loss_scale=1.0, norm_mean=None, norm_std=None):
    inputs = inputs.to(device)
    labels = labels.to(device)
    target_threat_domain_indices = target_threat_domain_indices.to(device)
    normalizer = IdentityNormalizer()

    threats = []
    norm_weights = []

    threats.append(ap.ThreatModel(
        pt.ReColorAdv,
        ap.PerturbationParameters(
            lp_style='inf',
            lp_bound=[0.06, 0.06, 0.06],
            xform_params={
                'resolution_x': 16,
                'resolution_y': 32,
                'resolution_z': 32,
            },
            xform_class=ct.FullSpatial,
            use_smooth_loss=True,
            cspace=cs.CIELUVColorSpace(),
        ),
    ))
    norm_weights.append(1.0)

    threats.append(ap.ThreatModel(
        ap.DeltaAddition,
        ap.PerturbationParameters(
            lp_style='inf',
            lp_bound=8.0 / 255,
        ),
    ))
    norm_weights.append(0.0)

    sequence_threat = ap.ThreatModel(
        ap.SequentialPerturbation,
        threats,
        ap.PerturbationParameters(norm_weights=norm_weights),
    )

    adv_loss = JointClassificationDiagnosisCWMarginLoss(
        model,
        normalizer,
        kappa=float('inf'),
        target_threat_domain_indices=target_threat_domain_indices,
        diagnosis_loss_scale=diagnosis_loss_scale,
    )
    st_loss = lf.PerturbationNormLoss(lp=2)
    loss_fxn = lf.RegularizedLoss(
        {'adv': adv_loss, 'pert': st_loss},
        {'adv': 1.0, 'pert': 0.05},
        negate=True,
    )

    pgd_attack = aa.PGD(model, normalizer, sequence_threat, loss_fxn)
    attack = advtrain.AdversarialAttackParameters(
        pgd_attack,
        1.0,
        attack_specific_params={'attack_kwargs': {
            'num_iterations': max_iterations,
            'optimizer': optim.Adam,
            'optimizer_kwargs': {'lr': lr},
            'signed': False,
            'verbose': False,
        }},
    )

    adv_inputs = attack.attack(inputs, labels)[0]
    return adv_inputs.clamp(0.0, 1.0).detach()
