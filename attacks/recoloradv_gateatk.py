import torch
from torch import optim

from recoloradv.mister_ed import adversarial_attacks as aa
from recoloradv.mister_ed import adversarial_perturbations as ap
from recoloradv.mister_ed import adversarial_training as advtrain
from recoloradv.mister_ed import loss_functions as lf

from recoloradv import color_spaces as cs
from recoloradv import color_transformers as ct
from recoloradv import perturbations as pt

from attacks.gateatk_utils import balanced_gate_weight, gate_margin_loss_minimize, unpack_model_outputs


class IdentityNormalizer:
    def __call__(self, x):
        return x

    def forward(self, x):
        return x

    def differentiable_call(self):
        return self

    def nondifferentiable_call(self):
        return self


class JointCWLossWithGate(lf.PartialLoss):
    def __init__(self, classifier, normalizer=None, kappa=0.0, true_gate=None, gate_loss_scale=1.0):
        super(JointCWLossWithGate, self).__init__()
        self.classifier = classifier
        self.normalizer = normalizer
        self.kappa = kappa
        self.true_gate = true_gate
        self.gate_loss_scale = gate_loss_scale
        self.nets.append(self.classifier)

    def forward(self, examples, labels, *args, **kwargs):
        classifier_in = self.normalizer.forward(examples) if self.normalizer is not None else examples
        outputs = self.classifier.forward(classifier_in)
        classifier_out, gate_logits = unpack_model_outputs(outputs)

        target_logits = torch.gather(classifier_out, 1, labels.view(-1, 1))
        max_2_logits, argmax_2_logits = torch.topk(classifier_out, 2, dim=1)
        top_max, second_max = max_2_logits.chunk(2, dim=1)
        top_argmax, _ = argmax_2_logits.chunk(2, dim=1)
        targets_eq_max = top_argmax.squeeze(1).eq(labels).float().view(-1, 1)
        targets_ne_max = top_argmax.squeeze(1).ne(labels).float().view(-1, 1)
        max_other = targets_eq_max * second_max + targets_ne_max * top_max
        loss_cls = torch.clamp(target_logits - max_other, min=-1 * self.kappa).squeeze(1)

        if gate_logits is not None and self.true_gate is not None:
            loss_gate = gate_margin_loss_minimize(gate_logits, self.true_gate, reduction='none')
            if self.kappa != float('inf'):
                loss_gate = torch.clamp(loss_gate, min=-1 * self.kappa)
            weight = balanced_gate_weight(loss_cls, loss_gate, self.gate_loss_scale)
            return loss_cls + weight * loss_gate

        return loss_cls


def ReColorAdv(inputs, labels, true_gate, model, device, max_iterations=10, lr=0.01, ncls=10,
               gate_loss_scale=1.0, norm_mean=None, norm_std=None):
    inputs = inputs.to(device)
    labels = labels.to(device)
    true_gate = true_gate.to(device)
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

    adv_loss = JointCWLossWithGate(
        model,
        normalizer,
        kappa=float('inf'),
        true_gate=true_gate,
        gate_loss_scale=gate_loss_scale,
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
