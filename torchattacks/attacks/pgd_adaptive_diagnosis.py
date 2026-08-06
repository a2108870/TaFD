# -*- coding: utf-8 -*-
"""Projected-gradient attack that jointly targets classification and threat-domain diagnosis."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..attack import Attack


class AdaptiveDiagnosisPGD(Attack):
    r"""PGD over the classification and threat-domain diagnosis objectives."""

    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, random_start=True, diagnosis_loss_scale=1.0):
        super().__init__("AdaptiveDiagnosisPGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.diagnosis_loss_scale = diagnosis_loss_scale
        self.supported_mode = ['default']

    def forward(self, images, labels, target_threat_domain_indices):
        r"""Generate adversarial examples for a target threat-domain assignment."""
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        target_threat_domain_indices = target_threat_domain_indices.clone().detach().to(self.device)

        adv_images = images.clone().detach()

        if self.random_start:
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True



            if self._normalization_applied:
                norm_images = self.normalize(adv_images)
            else:
                norm_images = adv_images

            outputs = self.model(norm_images)

            if isinstance(outputs, tuple) and len(outputs) >= 3:
                cls_logits, _, diagnosis_logits, _ = outputs
            else:
                cls_logits = outputs[0] if isinstance(outputs, tuple) else outputs
                diagnosis_logits = None


            loss_cls = F.cross_entropy(cls_logits, labels)





            if diagnosis_logits is not None:


                one_hot = torch.zeros_like(diagnosis_logits)
                one_hot.scatter_(1, target_threat_domain_indices.view(-1, 1), 1)

                real_logit = (diagnosis_logits * one_hot).sum(1)
                other_logits = (diagnosis_logits - 1e9 * one_hot).max(1)[0]



                diagnosis_loss = other_logits - real_logit
                diagnosis_loss = diagnosis_loss.mean()


                with torch.no_grad():

                    w_base = loss_cls.detach().abs()
                    diagnosis_magnitude = diagnosis_loss.detach().abs() + 1e-8
                    weight = w_base / diagnosis_magnitude
                    weight = torch.clamp(weight, 0.1, 10.0)

                # Apply user-defined scaling
                weight = weight * self.diagnosis_loss_scale

                cost = loss_cls + weight * diagnosis_loss
            else:
                cost = loss_cls


            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
