import time

import numpy as np
import torch
import torch.nn.functional as F

from attacks.gateatk_utils import balanced_gate_weight, gate_margin_loss_maximize, unpack_model_outputs
from ..attack import Attack


class APGD_GateAtk(Attack):
    def __init__(self, model, norm='Linf', eps=8 / 255, steps=10, n_restarts=1, seed=0,
                 loss='ce', eot_iter=1, rho=.75, verbose=False, gate_loss_scale=1.0):
        super().__init__('APGD_GateAtk', model)
        self.eps = eps
        self.steps = steps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.loss = loss
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.verbose = verbose
        self.gate_loss_scale = gate_loss_scale
        self.supported_mode = ['default']

    def forward(self, images, labels, true_gate):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        true_gate = true_gate.clone().detach().to(self.device)
        _, adv_images = self.perturb(images, labels, true_gate, best_loss=True)
        return adv_images

    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = np.zeros(x.shape[1])
        for counter5 in range(k):
            t += x[j - counter5] > x[j - counter5 - 1]
        return t <= k * k3 * np.ones(t.shape)

    def check_shape(self, x):
        return x if len(x.shape) > 0 else np.expand_dims(x, 0)

    def dlr_loss(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        return -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

    def _forward_joint(self, inputs, labels, true_gate):
        if self._normalization_applied:
            inputs = self.normalize(inputs)
        outputs = self.model(inputs)
        cls_logits, gate_logits = unpack_model_outputs(outputs)

        if self.loss == 'ce':
            loss_cls = F.cross_entropy(cls_logits, labels, reduction='none')
        elif self.loss == 'dlr':
            loss_cls = self.dlr_loss(cls_logits, labels)
        else:
            raise ValueError('unknown loss')

        if gate_logits is not None:
            loss_gate = gate_margin_loss_maximize(gate_logits, true_gate, reduction='none')
            weight = balanced_gate_weight(loss_cls, loss_gate, self.gate_loss_scale)
            loss_indiv = loss_cls + weight * loss_gate
        else:
            loss_indiv = loss_cls

        return cls_logits, loss_indiv

    def attack_single_run(self, x_in, y_in, true_gate_in):
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)
        true_gate = true_gate_in.clone() if len(true_gate_in.shape) == 1 else true_gate_in.clone().unsqueeze(0)

        self.steps_2, self.steps_min, self.size_decr = max(int(0.22 * self.steps), 1), max(int(0.06 * self.steps), 1), max(int(0.03 * self.steps), 1)
        if self.verbose:
            print('parameters: ', self.steps, self.steps_2, self.steps_min, self.size_decr)

        if self.norm == 'Linf':
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * t / (t.reshape([t.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1]))
        elif self.norm == 'L2':
            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * t / ((t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
        else:
            raise ValueError(f'Unsupported norm: {self.norm}')

        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.steps, x.shape[0]])
        loss_best_steps = torch.zeros([self.steps + 1, x.shape[0]])
        acc_steps = torch.zeros_like(loss_best_steps)

        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                cls_logits, loss_indiv = self._forward_joint(x_adv, y, true_gate)
                loss = loss_indiv.sum()
            grad += torch.autograd.grad(loss, [x_adv])[0].detach()

        grad /= float(self.eot_iter)
        grad_best = grad.clone()

        acc = cls_logits.detach().max(1)[1] == y
        acc_steps[0] = acc + 0
        loss_best = loss_indiv.detach().clone()

        step_size = self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * torch.tensor([2.0], device=self.device).detach().reshape([1, 1, 1, 1])
        x_adv_old = x_adv.clone()
        k = self.steps_2 + 0
        u = np.arange(x.shape[0])
        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)

        for i in range(self.steps):
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()
                a = 0.75 if i > 0 else 1.0

                if self.norm == 'Linf':
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - self.eps), x + self.eps), 0.0, 1.0)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a), x - self.eps), x + self.eps), 0.0, 1.0)
                else:
                    x_adv_1 = x_adv + step_size * grad / ((grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(self.eps * torch.ones(x.shape).to(self.device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()), 0.0, 1.0)
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(self.eps * torch.ones(x.shape).to(self.device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

                x_adv = x_adv_1 + 0.

            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    cls_logits, loss_indiv = self._forward_joint(x_adv, y, true_gate)
                    loss = loss_indiv.sum()
                grad += torch.autograd.grad(loss, [x_adv])[0].detach()

            grad /= float(self.eot_iter)

            pred = cls_logits.detach().max(1)[1] == y
            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc + 0
            x_best_adv[(pred == 0).nonzero().squeeze()] = x_adv[(pred == 0).nonzero().squeeze()] + 0.
            if self.verbose:
                print('iteration: {} - Best loss: {:.6f}'.format(i, loss_best.sum()))

            with torch.no_grad():
                y1 = loss_indiv.detach().clone()
                loss_steps[i] = y1.cpu() + 0
                ind = (y1 > loss_best).nonzero().squeeze()
                x_best[ind] = x_adv[ind].clone()
                grad_best[ind] = grad[ind].clone()
                loss_best[ind] = y1[ind] + 0
                loss_best_steps[i + 1] = loss_best + 0

                counter3 += 1

                if counter3 == k:
                    fl_oscillation = self.check_oscillation(loss_steps.detach().cpu().numpy(), i, k, loss_best.detach().cpu().numpy(), k3=self.thr_decr)
                    fl_reduce_no_impr = (~reduced_last_check) * (loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy())
                    fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                    reduced_last_check = np.copy(fl_oscillation)
                    loss_best_last_check = loss_best.clone()

                    if np.sum(fl_oscillation) > 0:
                        step_size[u[fl_oscillation]] /= 2.0
                        fl_oscillation = np.where(fl_oscillation)
                        x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                        grad[fl_oscillation] = grad_best[fl_oscillation].clone()

                    counter3 = 0
                    k = np.maximum(k - self.size_decr, self.steps_min)

        return x_best, acc, loss_best, x_best_adv

    def perturb(self, x_in, y_in, true_gate_in, best_loss=False, cheap=True):
        assert self.norm in ['Linf', 'L2']
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)
        true_gate = true_gate_in.clone() if len(true_gate_in.shape) == 1 else true_gate_in.clone().unsqueeze(0)

        adv = x.clone()
        acc = self.get_logits(x).max(1)[1] == y
        if self.verbose:
            print('-------------------------- running {}-attack with epsilon {:.4f} --------------------------'.format(self.norm, self.eps))
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))
        startt = time.time()

        if not best_loss:
            torch.random.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.random.manual_seed(self.seed)

            if not cheap:
                raise ValueError('not implemented yet')

            for counter in range(self.n_restarts):
                ind_to_fool = acc.nonzero().squeeze()
                if len(ind_to_fool.shape) == 0:
                    ind_to_fool = ind_to_fool.unsqueeze(0)
                if ind_to_fool.numel() != 0:
                    x_to_fool = x[ind_to_fool].clone()
                    y_to_fool = y[ind_to_fool].clone()
                    tg_to_fool = true_gate[ind_to_fool].clone()
                    best_curr, acc_curr, loss_curr, adv_curr = self.attack_single_run(x_to_fool, y_to_fool, tg_to_fool)
                    ind_curr = (acc_curr == 0).nonzero().squeeze()
                    acc[ind_to_fool[ind_curr]] = 0
                    adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                    if self.verbose:
                        print('restart {} - robust accuracy: {:.2%} - cum. time: {:.1f} s'.format(counter, acc.float().mean(), time.time() - startt))

            return acc, adv

        adv_best = x.detach().clone()
        loss_best = torch.ones([x.shape[0]]).to(self.device) * (-float('inf'))
        for counter in range(self.n_restarts):
            best_curr, _, loss_curr, _ = self.attack_single_run(x, y, true_gate)
            ind_curr = (loss_curr > loss_best).nonzero().squeeze()
            adv_best[ind_curr] = best_curr[ind_curr] + 0.
            loss_best[ind_curr] = loss_curr[ind_curr] + 0.
            if self.verbose:
                print('restart {} - loss: {:.5f}'.format(counter, loss_best.sum()))

        return loss_best, adv_best
