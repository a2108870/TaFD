# -*- coding: utf-8 -*-
"""
PGD GateAtk - 同时攻击分类器和域分类器（门控）的PGD攻击

特点：
1. 动态权重平衡：自动平衡分类损失和门控损失
2. 无目标门控攻击：最大化正确门控的交叉熵
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..attack import Attack


class PGD_GateAtk(Attack):
    r"""
    PGD GateAtk: 同时攻击分类器和门控器的PGD攻击

    Arguments:
        model (nn.Module): 要攻击的模型
        eps (float): 最大扰动 (Default: 8/255)
        alpha (float): 步长 (Default: 2/255)
        steps (int): 迭代步数 (Default: 10)
        random_start (bool): 随机初始化扰动 (Default: True)

    Examples::
        >>> attack = PGD_GateAtk(model, eps=8/255, alpha=2/255, steps=10)
        >>> adv_images = attack(images, labels, true_gate)
    """

    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, random_start=True, gate_loss_scale=1.0):
        super().__init__("PGD_GateAtk", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.gate_loss_scale = gate_loss_scale
        self.supported_mode = ['default']

    def forward(self, images, labels, true_gate):
        r"""
        执行 GateAtk 攻击

        Args:
            images: 输入图像 [B, C, H, W]，范围 [0, 1]
            labels: 分类标签 [B]
            true_gate: 真实门控标签 [B]

        Returns:
            adv_images: 对抗样本 [B, C, H, W]
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        true_gate = true_gate.clone().detach().to(self.device)

        adv_images = images.clone().detach()

        if self.random_start:
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True

            # 前向传播 - 直接调用 model 获取完整输出（包括 gate_logits）
            # 注意：不能使用 self.get_logits()，因为它只返回 cls_logits
            if self._normalization_applied:
                norm_images = self.normalize(adv_images)
            else:
                norm_images = adv_images

            outputs = self.model(norm_images)

            if isinstance(outputs, tuple) and len(outputs) >= 3:
                cls_logits, _, gate_logits, _ = outputs
            else:
                cls_logits = outputs[0] if isinstance(outputs, tuple) else outputs
                gate_logits = None

            # 分类损失
            loss_cls = F.cross_entropy(cls_logits, labels)

            # 门控损失（增强版：Logit Margin Loss）
            # 我们希望 max(Others) > True (即分类错误)
            # 定义 loss = max(Others) - True
            # PGD 会最大化这个 loss
            if gate_logits is not None:
                # 获取除了真实标签以外的最大 logit
                # 创建 mask 屏蔽真实标签
                one_hot = torch.zeros_like(gate_logits)
                one_hot.scatter_(1, true_gate.view(-1, 1), 1)

                real_logit = (gate_logits * one_hot).sum(1)
                other_logits = (gate_logits - 1e9 * one_hot).max(1)[0]

                # 我们希望 other_logits > real_logit
                # 所以最大化 (other_logits - real_logit)
                loss_gate = other_logits - real_logit
                loss_gate = loss_gate.mean()

                # 动态权重平衡
                with torch.no_grad():
                    # 避免除零
                    w_base = loss_cls.detach().abs()
                    w_gate = loss_gate.detach().abs() + 1e-8
                    weight = w_base / w_gate
                    weight = torch.clamp(weight, 0.1, 10.0)

                # Apply user-defined scaling
                weight = weight * self.gate_loss_scale

                cost = loss_cls + weight * loss_gate
            else:
                cost = loss_cls

            # 梯度更新
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
