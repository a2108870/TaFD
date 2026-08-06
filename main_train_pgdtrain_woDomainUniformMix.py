#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_woorth_firstblock_FDConv10_mix_randT.py
--------------------------------------------
鍩轰簬 main_woorth_firstblock_FDConv10_mix.py 鐨勬敼杩涚増鏈€?

鏍稿績鏀瑰姩锛?
  - 验证时：使用当前评估配置
  - SPSA 浠呭湪 epoch=50 鍜?75 鏃惰瘎浼?

浣跨敤鏂规硶锛?
  python main_woorth_firstblock_FDConv10_mix_randT.py --dataset CIFAR100
"""

import os

import random
import argparse
import numpy as np
from tqdm import tqdm

# 鈹€鈹€鈹€ PyTorch & TorchVision 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
# 淇锛氬湪瀵煎叆 pyplot 涔嬪墠锛屽己鍒朵娇鐢?'Agg' 鍚庣锛堥伩鍏嶅杩涚▼涓?tkinter 鍐茬獊锛?
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 鈹€鈹€鈹€ 绗笁鏂?/ 鑷畾涔夊簱 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
from utils.utils import *
from utils.datasets_utils import GetDataLoader
from torchattacks import APGD, PGD, PGDL2, SPSA

# V10 鏀诲嚮
from attacks.ace import ACE
from attacks.recoloradv import ReColorAdv
from attacks.light import light_atk
from attacks.hue import hue_atk
from attacks.uaa import UAA_atk

# V20 鏀诲嚮
from attacks.subspace import subspace_atk, build_pca_bases, save_pca_bases, load_pca_bases
from attacks.stadv import stadv_attack

# 鏁村悎妯″瀷
from models.encoder_woDomainUniformMix import create_encoder

# -------------------------------------------------------------------------
# 鍏ㄥ眬璁惧锛堢敱 --gpu 鍙傛暟鍦?main 鍏ュ彛澶勮鐩栵級
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# -------------------------------------------------------------------------

# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
#  鏀诲嚮閰嶇疆娉ㄥ唽琛?
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
ATTACK_CONFIGS = {
    'v10': {
        'train_attacks': ['APGD_Linf', 'APGD_L2', 'ACE', 'Hue', 'ReColorAdv', 'Light', 'UAA'],
        'test_attacks': ['Clean', 'APGD_Linf', 'APGD_L2', 'ACE', 'ReColorAdv', 'Hue', 'Light', 'UAA', 'SPSA'],
        'num_sources': 7,
        'domain_names': ['APGD_Linf', 'APGD_L2', 'ACE', 'Hue', 'ReColorAdv', 'Light', 'UAA'],
    },
    'v20': {
        'train_attacks': ['APGD_Linf', 'APGD_L2', 'ACE', 'SUB', 'STADV'],
        'test_attacks': ['Clean', 'APGD_Linf', 'APGD_L2', 'ACE', 'SUB', 'STADV', 'SPSA'],
        'num_sources': 5,
        'domain_names': ['APGD_Linf', 'APGD_L2', 'ACE', 'SUB', 'STADV'],
    }
}

# 鍙敤鏀诲嚮娉ㄥ唽琛紙鍚堝苟 V10 鍜?V20锛?
ALL_ATTACKS = ['Clean', 'APGD_Linf', 'APGD_L2', 'SPSA', 'ACE', 'ReColorAdv', 'Hue', 'Light', 'UAA', 'SUB', 'STADV']
DOMAIN_ATTACKS = ['APGD_Linf', 'APGD_L2', 'ACE', 'ReColorAdv', 'Hue', 'Light', 'UAA', 'SUB', 'STADV']
DOMAIN_SUPERVISED_ATTACKS = list(DOMAIN_ATTACKS)

# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
SPSA_EVAL_EPOCHS = {}

ATTACK_SHORT_NAMES = {
    'APGD_Linf': 'AP_L',
    'APGD_L2': 'AP_2',
    'ACE': 'ACE',
    'Hue': 'Hue',
    'ReColorAdv': 'ReC',
    'Light': 'Lig',
    'UAA': 'UAA',
    'SUB': 'SUB',
    'STADV': 'STA',
    'SPSA': 'SPSA',
    'Clean': 'Clean',
}


def _cuda_sync_if_needed():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _update_confusion_matrix(cm: torch.Tensor, true_domain: torch.Tensor, pred_domain: torch.Tensor):
    if true_domain.numel() == 0:
        return
    num_domains = cm.size(0)
    true_cpu = true_domain.detach().view(-1).to(torch.long).cpu()
    pred_cpu = pred_domain.detach().view(-1).to(torch.long).cpu()
    flat = true_cpu * num_domains + pred_cpu
    counts = torch.bincount(flat, minlength=num_domains * num_domains).view(num_domains, num_domains)
    cm += counts


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
#  宸ュ叿鍑芥暟锛氬涔犵巼銆佽閲忓櫒銆佸噯纭巼銆佸浘鍍忓綊涓€鍖?
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
def set_group_lrs(optimizer: optim.Optimizer, lr_main: float, lr_domain: float):
    """Set learning rates for named optimizer parameter groups."""
    for pg in optimizer.param_groups:
        if pg.get("name") == "domain":
            pg["lr"] = lr_domain
            pg["initial_lr"] = lr_domain
        else:
            pg["lr"] = lr_main
            pg["initial_lr"] = lr_main


def adjust_learning_rate(optimizer, epoch, lr_main_init=0.001, lr_domain_init=0.003):
    """Decay main and domain learning rates using the same schedule."""
    if epoch < 50:
        factor = 1.0
    elif epoch < 70:
        factor = 0.1
    else:
        factor = 0.01
    for pg in optimizer.param_groups:
        if pg.get("name") == "domain":
            pg["lr"] = lr_domain_init * factor
        else:
            pg["lr"] = lr_main_init * factor


def get_current_lrs(optimizer):
    """Return current main and domain learning rates."""
    lrs = {}
    for i, pg in enumerate(optimizer.param_groups):
        name = pg.get("name", f"group{i}")
        lrs[name] = pg["lr"]
    lr_main = lrs.get("main", list(lrs.values())[0])
    lr_domain = lrs.get("domain", lr_main)
    return lr_main, lr_domain


class AverageMeter:
    def __init__(self): self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        # 濡傛灉鏄?Tensor锛屼繚鎸佸湪 GPU 涓婄疮鍔狅紝閬垮厤鍚屾
        if isinstance(val, torch.Tensor):
            val = val.detach()
            # 纭繚鏄爣閲?tensor
            if val.numel() == 1:
                val = val.squeeze()
            if not isinstance(self.sum, torch.Tensor):
                self.sum = torch.tensor(0.0, device=val.device)
            self.sum = self.sum + val * n
        else:
            self.sum = self.sum + val * n
        self.count += n

    @property
    def avg(self):
        if self.count == 0: return 0
        ret = self.sum / self.count
        # 浠呭湪闇€瑕佹樉绀烘椂鎵嶅悓姝ュ洖 CPU
        if isinstance(ret, torch.Tensor):
            return ret.item()
        return ret


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# === 鍔ㄦ€佺被骞宠　鏉冮噸锛堜繚鐣欏疄鐜帮紝鎸夐渶鍚敤锛?==
def compute_balanced_ce_weight(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float().to(labels.device)
    counts = torch.clamp(counts, min=1.0)
    inv = 1.0 / counts
    weights = inv / inv.mean()  # 骞冲潎鏉冮噸=1
    return weights


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
#  缁樺浘锛氬噯纭巼鏇茬嚎
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
def plot_accuracy_curves(accuracies_dict, epoch, save_dir, title_prefix="acc"):
    if not accuracies_dict: return
    valid = any(len(v) for v in accuracies_dict.values())
    if not valid: return

    plt.figure(figsize=(14, 9))
    markers = ['o', 's', '^', 'D', 'v', '>', '<', '*', 'p', 'X', 'd', '|']
    colors = (plt.cm.tab20(np.linspace(0, 1, 20)).tolist() +
              plt.cm.tab20b(np.linspace(0, 1, 20)).tolist() +
              plt.cm.tab20c(np.linspace(0, 1, 20)).tolist())

    for i, (name, vals) in enumerate(accuracies_dict.items()):
        if not vals: continue
        epochs = list(range(len(vals)))
        plt.plot(epochs, vals, marker=markers[i % len(markers)],
                 linewidth=2, markersize=5, color=colors[i % len(colors)],
                 label=name)
        for x, y in zip(epochs, vals):
            plt.annotate(f'{y:.1f}', (x, y),
                         xytext=(0, 0.5 + (i % 3) * 0.8),
                         textcoords="offset points", ha='center', fontsize=7,
                         bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    plt.title(f'{title_prefix} vs. epoch', fontsize=14)
    plt.xlabel('epoch');
    plt.ylabel('acc (%)');
    plt.grid(ls='--', alpha=.6)
    plt.legend(loc='center left', bbox_to_anchor=(1, .5))
    os.makedirs(os.path.join(save_dir, 'accuracy_curves'), exist_ok=True)
    fname = f'{title_prefix}_curve_epoch_{epoch}.png'
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_curves', fname), dpi=100)
    plt.close()


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
#  瀵规姉鏍锋湰鐢熸垚锛堝惈锛歍iny 鏃?lr 脳0.1锛涗笖 ncls 闅忔暟鎹泦浼犲叆锛?
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
def generate_single_attack(attack_func, img, tgt, model, device,
                           attack_name="Unknown", **kwargs):
    try:
        if attack_name in ('APGD_Linf', 'APGD_L2', 'SPSA'):
            return attack_func(img, tgt)
        else:
            return attack_func(img, tgt, model, device, **kwargs)
    except Exception as e:
        print(f"鏀诲嚮澶辫触 {attack_name}: {e}")
        return img.clone()


def _domain_id_from_attack(attack_name: str, domain_names: list):
    if attack_name in domain_names:
        return domain_names.index(attack_name)
    return None


class _APGDAllSamples(APGD):
    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        _, adv_images = self.perturb(images, labels, best_loss=True)
        return adv_images


def _build_apgd_attack(model: nn.Module, norm: str, eps: float, steps: int, attack_cls=_APGDAllSamples):
    return attack_cls(model, norm=norm, eps=eps, steps=steps, n_restarts=1,
                      seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False)


def _build_train_pgd_attack(model: nn.Module, norm: str, eps: float, steps: int):
    if norm == 'Linf':
        return PGD(model, eps=eps, alpha=2 / 255, steps=steps, random_start=True)
    if norm == 'L2':
        return PGDL2(model, eps=eps, alpha=0.1, steps=steps, random_start=True)
    raise ValueError(f'Unsupported PGD norm: {norm}')


def generate_attack_batch(img, tgt, model, device, attack_names,
                          atk_apgd_linf=None, atk_apgd_l2=None, atk_spsa=None,
                          lr_scale=1.0, ncls=100, subspace_bases=None):
    """
    鏍规嵁 attack_names 鐢熸垚瀵规姉鏍锋湰瀛楀吀 {attack_name: adv_x}銆?
    鏀寔 V10 涓?V20 缁熶竴璋冪敤銆?
    """
    attack_inputs = {}

    for name in attack_names:
        if name == 'Clean':
            attack_inputs['Clean'] = img
        elif name == 'APGD_Linf':
            if atk_apgd_linf is None:
                raise ValueError("APGD_Linf 鏀诲嚮瀵硅薄鏈垵濮嬪寲")
            attack_inputs['APGD_Linf'] = generate_single_attack(atk_apgd_linf, img, tgt, model, device, "APGD_Linf")
        elif name == 'APGD_L2':
            if atk_apgd_l2 is None:
                raise ValueError("APGD_L2 鏀诲嚮瀵硅薄鏈垵濮嬪寲")
            attack_inputs['APGD_L2'] = generate_single_attack(atk_apgd_l2, img, tgt, model, device, "APGD_L2")
        elif name == 'SPSA':
            if atk_spsa is None:
                raise ValueError("SPSA 鏀诲嚮瀵硅薄鏈垵濮嬪寲")
            attack_inputs['SPSA'] = generate_single_attack(atk_spsa, img, tgt, model, device, "SPSA")
        elif name == 'ACE':
            attack_inputs['ACE'] = generate_single_attack(
                ACE, img, tgt, model, device, "ACE",
                lr=1 * lr_scale, max_iterations=10, ncls=ncls
            )
        elif name == 'Hue':
            attack_inputs['Hue'] = generate_single_attack(
                hue_atk, img, tgt, model, device, "Hue",
                lr=1 * lr_scale, max_iterations=10, ncls=ncls
            )
        elif name == 'ReColorAdv':
            attack_inputs['ReColorAdv'] = generate_single_attack(
                ReColorAdv, img, tgt, model, device, "ReColorAdv",
                lr=0.01 * lr_scale, max_iterations=10, ncls=ncls
            )
        elif name == 'Light':
            attack_inputs['Light'] = generate_single_attack(
                light_atk, img, tgt, model, device, "Light",
                lr=1 * lr_scale, max_iterations=10, ncls=ncls
            )
        elif name == 'UAA':
            attack_inputs['UAA'] = generate_single_attack(
                UAA_atk, img, tgt, model, device, "UAA",
                lr=0.1 * lr_scale, max_iterations=10, ncls=ncls
            )
        elif name == 'SUB':
            if subspace_bases is None:
                raise ValueError("SUB 鏀诲嚮闇€瑕?PCA bases锛岃鍏堟瀯寤烘垨鍔犺浇")
            attack_inputs['SUB'] = generate_single_attack(
                subspace_atk, img, tgt, model, device, "SUB",
                bases_dict=subspace_bases, steps=10, epsilon=2.0, ncls=ncls, proj='l2'
            )
        elif name == 'STADV':
            attack_inputs['STADV'] = generate_single_attack(
                stadv_attack, img, tgt, model, device, "STADV",
                eps=0.045, steps=10, mode='linf'
            )
        else:
            print(f"[WARN] 鏈敮鎸佺殑鏀诲嚮: {name}锛屽凡璺宠繃")

    return attack_inputs


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
#  杈呭姪锛氬瓧绗︿覆瑙ｆ瀽 + tqdm 鏄犲皠灞曠ず鍑芥暟
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
def _is_tiny_dataset(dataset_name: str) -> bool:
    return (dataset_name is not None) and ('tiny' in str(dataset_name).lower())


def _is_imagenette_dataset(dataset_name: str) -> bool:
    return (dataset_name is not None) and ('imagenette' in str(dataset_name).lower())


def _infer_input_size(dataset_name: str, fallback: int = 32) -> int:
    if dataset_name is None:
        return fallback
    name = str(dataset_name).lower()
    if 'imagenette' in name:
        return 224
    if name == 'tiny':
        return 64
    return 32


def _infer_num_classes(dataset_name: str, fallback: int = 10) -> int:
    """
    [淇敼] 淇閫昏緫锛岀‘淇?10class 浼樺厛鍖归厤
    瑙勫垯锛?
    - 鍚?'200class' 鈫?200
    - 鍚?'100' 鈫?100
    - 鍚?'10class' 鈫?10 (Tiny_32_10class)
    - 鍚?'cifar10' 鈫?10
    - 鍚?'tiny' (涓旀湭鍛戒腑 10class) 鈫?200 (Default Tiny ImageNet)
    - 鍚﹀垯鐢?fallback
    """
    if dataset_name is None:
        return fallback
    name = str(dataset_name).lower()

    if '200class' in name:
        return 200
    if 'imagenette' in name:
        return 10
    if ('cifar100' in name) or ('cifar-100' in name):
        return 100
    # [鍏抽敭淇] 浼樺厛鍖归厤 10class锛岄槻姝㈣ tiny 鎶㈠崰
    if '10class' in name:
        return 10
    if ('cifar10' in name) or ('cifar-10' in name):
        return 10

    # Tiny ImageNet 榛樿鏄?200 绫?
    if 'tiny' in name:
        return 200

    return fallback


def mapping_status_str(status_dict: dict, order=None) -> str:
    if order is None:
        order = ['APGD_Linf', 'APGD_L2', 'ACE', 'Hue', 'ReColorAdv', 'Light', 'UAA']
    parts = []
    for k in order:
        short_k = ATTACK_SHORT_NAMES.get(k, k)
        v = status_dict.get(k, 'Unassigned')
        if isinstance(v, str) and v.startswith('Domain-'):
            parts.append(f"{short_k}:{v.split('-')[-1]}")
        elif isinstance(v, int):
            parts.append(f"{short_k}:{v}")
        else:
            parts.append(f"{short_k}:{v}")
    return " ".join(parts)


def prepare_subspace_bases(args, trainloader):
    """Build or load PCA bases required by the SUB attack."""
    need_sub = ('SUB' in getattr(args, 'train_attacks', [])) or ('SUB' in getattr(args, 'attacks', []))
    if not need_sub:
        args.subspace_bases = None
        return

    if args.subspace_basis_path:
        basis_path = args.subspace_basis_path
    else:
        basis_path = os.path.join(
            args.result_dir,
            f"subspace_pca_bases_{args.dataset}_c{args.n_cls}_r{args.subspace_rank}.pth"
        )
    args.subspace_basis_path = basis_path

    if os.path.isfile(basis_path):
        print(f"[SUB] 鍔犺浇 PCA bases: {basis_path}")
        args.subspace_bases = load_pca_bases(basis_path, device=device)
        return

    print("[SUB] 鏈壘鍒?PCA bases锛屽紑濮嬫瀯寤猴紙鍙兘鑰楁椂锛?..")
    args.subspace_bases = build_pca_bases(
        trainloader,
        n_classes=args.n_cls,
        rank=args.subspace_rank,
        max_per_class=args.subspace_max_per_class,
        device=torch.device('cpu')
    )
    os.makedirs(os.path.dirname(basis_path) or '.', exist_ok=True)
    save_pca_bases(basis_path, args.subspace_bases)
    print(f"[SUB] PCA bases 宸蹭繚瀛? {basis_path}")


def _should_batch_eval_forwards(dataset_name: str) -> bool:
    return _infer_input_size(dataset_name) <= 64


def _domain_route_ablation_enabled(args) -> bool:
    return getattr(args, "ablate_domain_route", "none") != "none"


def _domain_route_ablation_mode(args) -> str:
    return getattr(args, "ablate_domain_route", "none")


def _fixed_domain_route_enabled(args) -> bool:
    return _domain_route_ablation_mode(args).startswith("fixed")


def _forced_domain_assignments(args, batch_size: int, target_device):
    if not _fixed_domain_route_enabled(args):
        return None
    route_id = int(getattr(args, "ablate_domain_route_id", 0))
    return torch.full((batch_size,), route_id, dtype=torch.long, device=target_device)


def _configure_domain_route_ablation(model, args):
    route_id = int(getattr(args, "ablate_domain_route_id", 0)) if _fixed_domain_route_enabled(args) else None
    uniform_mix = _domain_route_ablation_mode(args) == "uniform"
    if hasattr(model, "set_forced_domain_route"):
        model.set_forced_domain_route(route_id)
    else:
        setattr(model, "forced_domain_route_id", route_id)
    if hasattr(model, "set_uniform_domain_mix"):
        model.set_uniform_domain_mix(uniform_mix)
    else:
        setattr(model, "uniform_domain_mix", uniform_mix)


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
#  璁粌锛氱粺涓€鐢?棰勬祴 domain"杩涜璺敱 + 姣?N iter 鍔ㄦ€佽仛绫绘洿鏂?
#  鈽?鏍稿績鏀瑰姩锛欱PDA 娓╁害鍥哄畾涓?1.0
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
def train(epoch, trainloader, criterion, optimizer, n_classes, model,
          initial_lr_main=0.001, initial_lr_domain=0.003, args=None, global_iter_start=0):
    adjust_learning_rate(optimizer, epoch, lr_main_init=initial_lr_main, lr_domain_init=initial_lr_domain)
    lr_main, lr_domain = get_current_lrs(optimizer)
    print(
        f"\nEpoch {epoch:03d} | lr_main={lr_main:.6f} | lr_domain={lr_domain:.6f} | domains={args.domains}")

    losses = {k: AverageMeter() for k in ['cls', 'domain', 'total']}
    _configure_domain_route_ablation(model, args)
    domains = list(args.train_attacks)

    # [淇敼] 璁粌鏃跺垎鍒褰曞垎绫诲噯纭巼鍜屽煙鍑嗙‘鐜?
    train_cls_accs = {d: AverageMeter() for d in domains}
    train_dom_accs = {d: AverageMeter() for d in domains}

    domain_acc_meter = AverageMeter()

    # 鈽?璁板綍鏈?epoch 浣跨敤鐨勬俯搴︾粺璁?

    # 鈽?妯″瀷鍐呴儴宸插鐞嗘爣鍑嗗寲锛宼orchattacks 涓嶉渶瑕佸啀璁剧疆
    atk_apgd_linf = _build_train_pgd_attack(model, norm='Linf', eps=8 / 255, steps=10)
    atk_apgd_l2 = _build_train_pgd_attack(model, norm='L2', eps=0.5, steps=10)

    model.train()
    pbar = tqdm(trainloader, leave=True, desc=f"Train {epoch}")
    global_iter = global_iter_start

    # 鈽?tiny 鏁版嵁闆嗘椂缂╂斁棰滆壊/鍏夌収绫绘敾鍑?lr
    lr_scale = 0.1 if _is_tiny_dataset(getattr(args, "dataset", None)) else 1.0

    for step_idx, (img, tgt) in enumerate(pbar, start=1):
        img = img.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        bsz = img.size(0)
        source_ids_per_attack = torch.arange(len(domains), device=device, dtype=torch.long).repeat_interleave(bsz)

        # 鐢熸垚瀵规姉鏍锋湰锛堜紶鍏ュ姩鎬?n_classes锛?
        model.eval()

        # =========== 銆愭牳蹇冩敼鍔ㄣ€?===========

        # 2. 寮€鍚?BPDA锛岃鏀诲嚮鑳藉绌块€?DomainClassifier
        if hasattr(model, "set_bpda"):
            model.set_bpda(True)
        # ===================================

        attack_inputs = generate_attack_batch(
            img, tgt, model, device,
            attack_names=domains,
            atk_apgd_linf=atk_apgd_linf,
            atk_apgd_l2=atk_apgd_l2,
            lr_scale=lr_scale,
            ncls=n_classes,
            subspace_bases=getattr(args, 'subspace_bases', None)
        )

        # =========== 銆愪慨鏀广€?===========
        # 3. 鏀诲嚮鐢熸垚瀹屾瘯锛屽叧闂?BPDA锛屼繚璇佽缁冩潈閲嶆椂浣跨敤鐨勬槸鐪熷疄鐨?Hard Routing
        if hasattr(model, "set_bpda"):
            model.set_bpda(False)

        imgs_mix = [attack_inputs[d] for d in domains]

        global_iter += 1
        domain_route_ablated = _domain_route_ablation_enabled(args)
        if not domain_route_ablated:
            if epoch < 10:
                if global_iter % args.map_update_every == 0 or global_iter == 1:
                    with torch.no_grad():
                        model.extract_wavelet_features(torch.cat(imgs_mix, dim=0), source_ids_per_attack)

        if (not domain_route_ablated) and (global_iter % args.map_update_every == 0 or global_iter == 1):
            model.update_domain_mappings(epoch, args.end_epoch)

        domain_labels = source_ids_per_attack

        model.train()
        combined = torch.cat(imgs_mix, 0).detach()
        forced_domain_assignments = _forced_domain_assignments(args, combined.size(0), combined.device)
        if domain_route_ablated:
            true_domain_labels = (
                forced_domain_assignments
                if forced_domain_assignments is not None
                else model.get_domain_labels(domain_labels)
            )
        else:
            true_domain_labels = model.get_domain_labels(domain_labels)

        outputs = model(
            combined, None, criterion, bsz, flag=1,
            domain_ids=domain_labels,
            forced_domain_assignments=forced_domain_assignments
        )
        cls_logits, _, domain_logits, _ = outputs

        # domain ??
        if domain_route_ablated:
            domain_loss = domain_logits.new_zeros(())
        else:
            domain_loss = F.cross_entropy(domain_logits, true_domain_labels)
        losses['domain'].update(domain_loss, combined.size(0))

        with torch.no_grad():
            _, pred_domain = domain_logits.max(1)
            domain_acc = (pred_domain == true_domain_labels).float().mean() * 100
            domain_acc_meter.update(domain_acc, combined.size(0))

        repeated_tgt = tgt.repeat(len(domains))
        for i, dname in enumerate(domains):
            dom_out = cls_logits[bsz * i:bsz * (i + 1)]
            dom_logits_slice = domain_logits[bsz * i:bsz * (i + 1)]
            dom_tgt_slice = true_domain_labels[bsz * i:bsz * (i + 1)]

            with torch.no_grad():
                acc_cls = accuracy(dom_out, tgt, (1,))[0]
                train_cls_accs[dname].update(acc_cls, bsz)
                acc_dom = (dom_logits_slice.max(1)[1] == dom_tgt_slice).float().mean() * 100
                train_dom_accs[dname].update(acc_dom, bsz)

        cls_loss = F.cross_entropy(cls_logits, repeated_tgt)
        losses['cls'].update(cls_loss, combined.size(0))

        total_loss = cls_loss + args.domain_loss_weight * domain_loss
        losses['total'].update(total_loss, combined.size(0))

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        # tqdm 灞曠ず: 瀹炴椂鏄剧ず
        if step_idx % args.log_every == 0:
            status = model.get_all_mapping_statuses().get("global_mapping", {})
            pbar.set_postfix({
                "loss": f"{losses['cls'].avg:.3f}",
                "map": mapping_status_str(status, order=args.domain_names),
                **{ATTACK_SHORT_NAMES.get(d, d): (f"{train_cls_accs[d].avg:.1f}/{train_dom_accs[d].avg:.1f}" if d in DOMAIN_SUPERVISED_ATTACKS and train_dom_accs[d].count > 0 else f"{train_cls_accs[d].avg:.1f}/-") for d in domains},
            })

    pbar.close()
    return global_iter


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
#  验证：使用当前评估配置
#  鈽?鏍稿績鏀瑰姩锛歋PSA 浠呭湪鎸囧畾 epoch 璇勪及
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
def validation_pgd(epoch, testloader, criterion, model, n_cls,
                   acc_hist=None, domain_hist=None, args=None):
    """Run validation with the attack list selected in args.attacks."""
    # 鑾峰彇閫夋嫨鐨勬敾鍑?
    _configure_domain_route_ablation(model, args)
    selected_attacks = args.attacks if args and hasattr(args, 'attacks') else ALL_ATTACKS
    if 'SUB' in selected_attacks and getattr(args, 'subspace_bases', None) is None:
        raise RuntimeError("閫夋嫨浜?SUB 璇勪及锛屼絾鏈噯澶?PCA bases")

    print(f"\n[Validation] 閫夋嫨鐨勬敾鍑? {selected_attacks}")

    # 鈽?涓烘瘡涓俯搴﹀垱寤虹嫭绔嬬殑鎸囨爣瀛樺偍

    selected_attacks_runtime = list(selected_attacks)
    if 'SPSA' in selected_attacks_runtime and epoch not in SPSA_EVAL_EPOCHS:
        selected_attacks_runtime.remove('SPSA')
        print(f"[Validation] Skip SPSA except at epochs {sorted(SPSA_EVAL_EPOCHS)}")

    atk_apgd_linf = None
    atk_apgd_l2 = None
    atk_spsa = None
    if 'APGD_Linf' in selected_attacks_runtime:
        atk_apgd_linf = _build_apgd_attack(model, norm='Linf', eps=8 / 255, steps=100)
    if 'APGD_L2' in selected_attacks_runtime:
        atk_apgd_l2 = _build_apgd_attack(model, norm='L2', eps=0.5, steps=100)
    if 'SPSA' in selected_attacks_runtime:
        atk_spsa = SPSA(model, eps=8 / 255, delta=0.01, lr=0.01,
                        nb_iter=20, nb_sample=128, max_batch_size=32)

        print(f"\n{'='*70}")
        print(f"{'='*70}")

        # 涓哄綋鍓嶆俯搴﹀垵濮嬪寲鎸囨爣
        metrics = {k: AverageMeter() for k in selected_attacks_runtime}
        domain_accs = {k: AverageMeter() for k in selected_attacks_runtime if k in DOMAIN_ATTACKS}

        cm = torch.zeros(model.num_threat_domains, model.num_threat_domains, dtype=torch.long)
        model.eval()

        pbar = tqdm(testloader, desc=f"Val Ep{epoch}", leave=True)

        lr_scale = 0.1 if _is_tiny_dataset(getattr(args, "dataset", None)) else 1.0
        batch_eval_forwards = _should_batch_eval_forwards(getattr(args, "dataset", None))

        for step_idx, (img, tgt) in enumerate(pbar, start=1):
            img = img.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            bsz = img.size(0)

            # ========== 鈽?鏀诲嚮鐢熸垚闃舵锛氫娇鐢ㄥ綋鍓嶈瘎浼版俯搴?==========
            if hasattr(model, "set_bpda"):
                model.set_bpda(True)

            # 鈹€鈹€鈹€ 鏍规嵁閫夋嫨鐨勬敾鍑荤敓鎴愬鎶楁牱鏈?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            attack_inputs = generate_attack_batch(
                img, tgt, model, device,
                attack_names=selected_attacks_runtime,
                atk_apgd_linf=atk_apgd_linf,
                atk_apgd_l2=atk_apgd_l2,
                atk_spsa=atk_spsa,
                lr_scale=lr_scale,
                ncls=n_cls,
                subspace_bases=getattr(args, 'subspace_bases', None)
            )

            domain_ids = {}
            if not _domain_route_ablation_enabled(args):
                for name in attack_inputs.keys():
                    if name not in DOMAIN_ATTACKS:
                        continue
                    domain_idx = _domain_id_from_attack(name, args.domain_names)
                    if domain_idx is None:
                        continue
                    domain_ids[name] = torch.full((bsz,), domain_idx, device=device, dtype=torch.long)

            # ========== 鈽?鏀诲嚮鐢熸垚瀹岋細鍏抽棴 BPDA锛堣瘎浼?forward 淇濇寔鍘熼€昏緫锛?==========
            if hasattr(model, "set_bpda"):
                model.set_bpda(False)

            # 鈹€鈹€鈹€ 璇勪及姣忎釜鏀诲嚮 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            with torch.no_grad():
                if batch_eval_forwards and attack_inputs:
                    ordered_names = list(attack_inputs.keys())
                    combined_inputs = torch.cat([attack_inputs[name] for name in ordered_names], dim=0)
                    forced_domain_assignments = _forced_domain_assignments(
                        args, combined_inputs.size(0), combined_inputs.device
                    )
                    out_tuple = model(
                        combined_inputs,
                        domain_ids=None,
                        forced_domain_assignments=forced_domain_assignments
                    )
                    logits_all = out_tuple[0] if isinstance(out_tuple, tuple) else out_tuple
                    domain_logits_all = out_tuple[2] if isinstance(out_tuple, tuple) and len(out_tuple) > 2 else None

                    for idx, name in enumerate(ordered_names):
                        start = idx * bsz
                        end = start + bsz
                        logits = logits_all[start:end]
                        acc1 = accuracy(logits, tgt, (1,))[0]
                        metrics[name].update(acc1.item(), bsz)

                        if name in domain_ids and domain_logits_all is not None:
                            domain_logit = domain_logits_all[start:end]
                            pred_domain = domain_logit.max(1)[1]
                            true_domain = model.get_domain_labels(domain_ids[name])
                            d_acc = (pred_domain == true_domain).float().mean() * 100
                            domain_accs[name].update(d_acc.item(), bsz)

                            _update_confusion_matrix(cm, true_domain, pred_domain)
                else:
                    for name, x_in in attack_inputs.items():
                        forced_domain_assignments = _forced_domain_assignments(args, x_in.size(0), x_in.device)
                        out_tuple = model(
                            x_in,
                            domain_ids=None,
                            forced_domain_assignments=forced_domain_assignments
                        )
                        logits = out_tuple[0] if isinstance(out_tuple, tuple) else out_tuple
                        acc1 = accuracy(logits, tgt, (1,))[0]
                        metrics[name].update(acc1.item(), bsz)

                        # 鍩熷噯纭巼锛堜粎鏈夊搴旂湡鍩熸爣绛炬椂缁熻锛?
                        if name in domain_ids and isinstance(out_tuple, tuple) and len(out_tuple) > 2:
                            domain_logit = out_tuple[2]
                            pred_domain = domain_logit.max(1)[1]
                            true_domain = model.get_domain_labels(domain_ids[name])
                            d_acc = (pred_domain == true_domain).float().mean() * 100
                            domain_accs[name].update(d_acc.item(), bsz)

                            _update_confusion_matrix(cm, true_domain, pred_domain)

            # 鈹€鈹€鈹€ 鏋勫缓 tqdm postfix: Attack=ClsAcc/DomAcc 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            if step_idx % args.log_every == 0:
                postfix_dict = {}
                for k in selected_attacks_runtime:
                    if k in metrics:
                        cls_acc_str = f"{metrics[k].avg:.1f}"
                        if k in domain_accs and domain_accs[k].count > 0:
                            dom_acc_str = f"{domain_accs[k].avg:.1f}"
                            postfix_dict[k] = f"{cls_acc_str}/{dom_acc_str}"
                        else:
                            postfix_dict[k] = cls_acc_str
                pbar.set_postfix(postfix_dict)

        pbar.close()

        # 鈹€鈹€鈹€ 鎵撳嵃褰撳墠娓╁害鐨勯獙璇佹眹鎬?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        print(f"\n{'='*70}")
        print(f"Validation summary T={eval_T} (classification / domain accuracy)")
        print("=" * 70)
        print(f"{'Attack':<15} {'Cls Acc':>12} {'Domain Acc':>12} {'Format':>15}")
        print("-" * 70)
        for k in selected_attacks_runtime:
            if k in metrics:
                cls_acc = metrics[k].avg
                if k in domain_accs and domain_accs[k].count > 0:
                    dom_acc = domain_accs[k].avg
                    fmt_str = f"{cls_acc:.1f}/{dom_acc:.1f}"
                    print(f"{k:<15} {cls_acc:>11.2f}% {dom_acc:>11.2f}% {fmt_str:>15}")
                else:
                    print(f"{k:<15} {cls_acc:>11.2f}% {'N/A':>12} {cls_acc:>15.1f}")
        print("=" * 70)

        if cm.sum() > 0:
            print(f"\nDomain confusion matrix T={eval_T} (rows=true, cols=pred):")
            print(cm.cpu().numpy())

        # 瀛樺偍褰撳墠娓╁害鐨勭粨鏋?
        
        

    # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
    #  鎵撳嵃褰撳墠娓╁害姹囨€?
    # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
    print(f"\n{'#'*80}")
    print(f"  Validation summary (Epoch {epoch})")
    print(f"{'#'*80}")

    print(f"{'Attack':<12} | {'Cls%':>9}")
    print('-' * 24)
    for k in selected_attacks_runtime:
        if k in metrics:
            print(f"{k:<12} | {metrics[k].avg:>9.2f}%")

    print(f"{'#'*80}\n")

    # 更新历史记录
    if acc_hist is not None:
        for k, v in metrics.items():
            acc_hist.setdefault(k, []).append(v.avg)

    if domain_hist is not None:
        for k, v in domain_accs.items():
            if v.count > 0:
                domain_hist.setdefault(k, []).append(v.avg)

    clean_acc = metrics['Clean'].avg if 'Clean' in metrics else 0.0
    apgd_linf_acc = metrics['APGD_Linf'].avg if 'APGD_Linf' in metrics else 0.0

    return clean_acc, apgd_linf_acc, acc_hist, domain_hist


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
#  Checkpoint I/O
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
def save_checkpoint(model, optimizer, epoch, path, acc_hist=None, domain_hist=None):
    os.makedirs(path, exist_ok=True)
    state = {"model": model.state_dict(),
             "optimizer": optimizer.state_dict(),
             "epoch": epoch,
             "accuracy_history": acc_hist,
             "domain_acc_history": domain_hist}
    # torch.save(state, os.path.join(path, f"checkpoint_epoch_{epoch}.pth"))
    torch.save(state, os.path.join(path, "latest_model.pth"))


def load_checkpoint(model, optimizer, ckpt_path, base_main_lr=0.001, base_domain_lr=0.003):
    load_dev = device if torch.cuda.is_available() else torch.device("cpu")
    try:
        ckpt = torch.load(ckpt_path, map_location=load_dev, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=load_dev)
    # 鍏佽涓庡綋鍓嶆瀯閫犵暐鏈変笉涓€鑷达紙渚嬪 domains 鍙樺姩锛?
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        print(f"[WARN] state_dict 涓嶅畬鍏ㄥ尮閰嶏紝missing={len(missing)}, unexpected={len(unexpected)}")
    try:
        optimizer.load_state_dict(ckpt["optimizer"])
    except Exception as e:
        print(f"[WARN] 鍔犺浇浼樺寲鍣ㄧ姸鎬佸け璐ワ紙鍒嗙粍缁撴瀯涓嶅悓锛燂級宸插拷鐣ワ細{e}")
    set_group_lrs(optimizer, base_main_lr, base_domain_lr)
    print(f"=> 鎭㈠鑷?epoch {ckpt['epoch']}锛屽涔犵巼宸查噸璁句负 main={base_main_lr} / domain={base_domain_lr}")
    return ckpt["epoch"] + 1, ckpt.get("accuracy_history"), ckpt.get("domain_acc_history")


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
#  涓诲嚱鏁?
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
def _resume_eval_history_len(acc_hist, args):
    if not acc_hist:
        return 0

    preferred_attacks = ['Clean']
    if args is not None and hasattr(args, 'attacks'):
        preferred_attacks.extend([atk for atk in args.attacks if atk not in ('Clean', 'SPSA')])

    seen = set()
    for attack_name in preferred_attacks:
        if attack_name in seen:
            continue
        seen.add(attack_name)
        if attack_name in acc_hist:
            return len(acc_hist[attack_name])

    non_spsa_lengths = [len(vals) for name, vals in acc_hist.items() if name != 'SPSA']
    if non_spsa_lengths:
        return max(non_spsa_lengths)
    return 0


def _should_run_resume_eval(last_completed_epoch, acc_hist, args):
    if args is None or last_completed_epoch <= 0:
        return False
    if args.eval_freq <= 0 or last_completed_epoch % args.eval_freq != 0:
        return False

    expected_eval_points = last_completed_epoch // args.eval_freq
    recorded_eval_points = _resume_eval_history_len(acc_hist, args)
    if recorded_eval_points < expected_eval_points:
        return True

    curve_path = os.path.join(
        args.result_dir,
        'accuracy_curves',
        f'acc_curve_epoch_{last_completed_epoch}.png'
    )
    return not os.path.isfile(curve_path)


def main(args):
    print("==> Preparing data ...")
    trainloader, testloader = GetDataLoader(args.dataset,
                                            args.batch_size,
                                            args.test_batch_size,
                                            args.dataset_path,
                                            num_workers=args.num_workers,
                                            pin_memory=not args.disable_pin_memory,
                                            persistent_workers=not args.disable_persistent_workers,
                                            prefetch_factor=args.prefetch_factor)

    # SUB 鏀诲嚮鎵€闇€ PCA bases锛堟寜闇€鍑嗗锛?
    prepare_subspace_bases(args, trainloader)

    print("==> Building model ...")
    # 浣跨敤 create_encoder 宸ュ巶鍑芥暟锛屾牴鎹?backbone 鍜?attack_config 鍒涘缓妯″瀷
    # 浼犲叆 dataset 鍙傛暟锛屾ā鍨嬪唴閮ㄤ細鑷姩閫夋嫨姝ｇ‘鐨勬爣鍑嗗寲鍙傛暟
    model_kwargs = dict(
        backbone=args.backbone,
        dataset=args.dataset,
        num_sources=args.num_sources,
        num_domains=args.domains,
        fd_num_experts=args.domains,
        num_classes=args.n_cls,
        source_names=args.domain_names
    )
    if args.backbone == 'mobilevit':
        model_kwargs['size'] = _infer_input_size(args.dataset)
        model_kwargs['mvit_fdconv'] = bool(getattr(args, 'mvit_fdconv', False))

    model = create_encoder(**model_kwargs).to(device)
    model.count_frequency_convolutions()
    _configure_domain_route_ablation(model, args)

    # 鈹€鈹€ 鍒嗙粍浼樺寲鍣細threat_domain_classifier 鐢ㄦ洿澶у涔犵巼 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    domain_params = list(model.threat_domain_classifier.parameters())
    domain_param_ids = set(id(p) for p in domain_params)
    main_params = [p for p in model.parameters() if id(p) not in domain_param_ids]

    optimizer = optim.Adam([
        {"params": main_params, "lr": args.lr, "weight_decay": args.weight_decay, "name": "main"},
        {"params": domain_params, "lr": args.lr_domain, "weight_decay": args.weight_decay, "name": "domain"},
    ])
    set_group_lrs(optimizer, args.lr, args.lr_domain)

    # 鏍规嵁閫夋嫨鐨勬敾鍑诲垵濮嬪寲鍘嗗彶璁板綍
    acc_hist = {k: [] for k in args.attacks}
    domain_hist = {k: [] for k in args.attacks if k in DOMAIN_ATTACKS}

    start_epoch = args.start_epoch
    if args.resume and os.path.isfile(args.resume):
        start_epoch, tmp_acc, tmp_domain = load_checkpoint(
            model, optimizer, args.resume,
            base_main_lr=args.lr,
            base_domain_lr=args.lr_domain
        )
        if tmp_acc:  acc_hist = tmp_acc
        if tmp_domain: domain_hist = tmp_domain

    criterion = nn.CrossEntropyLoss()

    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"n_cls (auto): {args.n_cls}")
    print(f"鍒濆瀛︿範鐜?main={args.lr} | domain={args.lr_domain} | 璋冨害: 0-49脳1, 50-74脳0.1, 鈮?5脳0.01")
    print(f"domain_loss_weight: {args.domain_loss_weight}")
    if _domain_route_ablation_enabled(args):
        if _fixed_domain_route_enabled(args):
            print(f"[Ablation] domain routing: {args.ablate_domain_route}, route_id={args.ablate_domain_route_id}")
        else:
            print(f"[Ablation] domain routing: {args.ablate_domain_route}")
    print(f"map_update_every: {args.map_update_every} iters")
    print(f"domains(鑱氱被/鍩熸暟)={args.domains}")
    print(f"璁粌鏀诲嚮: {args.train_attacks}")
    print(
        f"attack_lr_scale: {'0.1 (tiny)' if _is_tiny_dataset(args.dataset) else '1.0'}  鈫?浣滅敤浜?ACE/Hue/ReColorAdv/Light/UAA")
    print(f"閫夋嫨鐨勮瘎浼版敾鍑? {args.attacks}")
    if getattr(args, 'subspace_bases', None) is not None:
        print(f"SUB PCA bases: {args.subspace_basis_path}")
    if _is_imagenette_dataset(args.dataset):
        print("[Note] Imagenette uses higher-resolution inputs; current attack defaults come from low-res experiments, so please retune before final reporting.")
    print("=" * 60)
    print(f"   - SPSA: evaluated only at epochs {sorted(SPSA_EVAL_EPOCHS)}")
    print("=" * 60)

    last_completed_epoch = start_epoch - 1
    if args.resume and _should_run_resume_eval(last_completed_epoch, acc_hist, args):
        print(f"[Resume] Missing validation curve for epoch {last_completed_epoch}; running validation first.")
        validation_pgd(last_completed_epoch, testloader, criterion,
                       model, args.n_cls,
                       acc_hist, domain_hist, args=args)
        plot_accuracy_curves(acc_hist, last_completed_epoch, args.result_dir, "acc")
        plot_accuracy_curves(domain_hist, last_completed_epoch, args.result_dir, "domain_acc")
        save_checkpoint(model, optimizer, last_completed_epoch,
                        args.result_dir, acc_hist, domain_hist)

    global_iter = 0
    for epoch in range(start_epoch, args.end_epoch):
        global_iter = train(epoch, trainloader, criterion, optimizer,
                            args.n_cls, model,
                            initial_lr_main=args.lr, initial_lr_domain=args.lr_domain,
                            args=args, global_iter_start=global_iter)

        save_checkpoint(model, optimizer, epoch,
                        args.result_dir, acc_hist, domain_hist)

        if epoch % args.eval_freq == 0 and epoch != 0:
            validation_pgd(epoch, testloader, criterion,
                           model, args.n_cls,
                           acc_hist, domain_hist, args=args)

            plot_accuracy_curves(acc_hist, epoch, args.result_dir, "acc")
            plot_accuracy_curves(domain_hist, epoch, args.result_dir, "domain_acc")
            save_checkpoint(model, optimizer, epoch,
                            args.result_dir, acc_hist, domain_hist)

        # 鈽?[ADDED IN COPY] harmful-routing eval (gateatk) at epoch 25/50/75 -> table + plots in result_dir/hr_eval
        if epoch in (25, 50, 75):
            try:
                from hr_eval_gateatk import run_hr_eval
                print(f"[HR-EVAL] launching harmful-routing eval at epoch {epoch} ...", flush=True)
                run_hr_eval(model, testloader, args, device, epoch, args.result_dir)
            except Exception as _hre:
                import traceback
                print(f"[HR-EVAL] failed at epoch {epoch}: {_hre}", flush=True)
                traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Unified-LR Adversarial Training + PGD-Train APGD-Test + Fixed BPDA-T")
    parser.add_argument("--dataset", type=str, default="CIFAR100",
                        help="鏁版嵁闆嗗悕绉帮紙渚嬪 Tiny_32_10class / Tiny_32_200class / CIFAR10 / CIFAR100 绛夛級")
    parser.add_argument("--dataset_path", type=str, default="./datasets/",
                        help="Root directory containing downloaded datasets.")
    parser.add_argument("--lr", type=float, default=0.001, help="Main learning rate")
    parser.add_argument("--lr_domain", type=float, default=0.001, help="domain_classifier 瀛︿範鐜囷紙寤鸿 3~10 鍊嶏級")
    parser.add_argument("--weight_decay", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=16, help="??/?? batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="DataLoader prefetch_factor")
    parser.add_argument("--disable_pin_memory", action="store_true", help="鍏抽棴 DataLoader pin_memory")
    parser.add_argument("--disable_persistent_workers", action="store_true", help="鍏抽棴 DataLoader persistent_workers")
    parser.add_argument("--log_every", type=int, default=10, help="姣忛殧澶氬皯涓?iter 鏇存柊涓€娆?tqdm 鎸囨爣")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--end_epoch", type=int, default=76)
    parser.add_argument("--eval_freq", type=int, default=25)
    parser.add_argument("--resume", type=str, default="",
                        help="Optional checkpoint path for resuming training or evaluation.")
    parser.add_argument("--result_dir", type=str,
                        default="")
    # 淇濈暀 n_cls 鍙傛暟锛屼絾浼氳鑷姩鎺ㄦ柇瑕嗙洊
    parser.add_argument("--n_cls", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)

    # 鍩?鑱氱被鏁帮紙鎺у埗鍩熷垎绫诲櫒銆佽仛绫汇€佷笓瀹舵暟锛?
    parser.add_argument("--domains", type=int, default=1, help="Number of domains/clusters")

    # domain 鐩戠潱鏉冮噸
    parser.add_argument("--domain_loss_weight", type=float, default=1.0,
                        help="Weight for the domain supervision loss")
    # 鍔ㄦ€佽仛绫婚鐜囷紙杩唬姝ワ級
    parser.add_argument("--ablate_domain_route", type=str, default="none",
                        choices=["none", "fixed0", "uniform"],
                        help="Ablation for domain routing. fixed0 forces route 0; uniform uses all routes equally.")
    parser.add_argument("--ablate_domain_route_id", type=int, default=0,
                        help="Route id used by fixed route ablations.")
    parser.add_argument("--map_update_every", type=int, default=50)

    # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
    #  鏀诲嚮閫夋嫨鍙傛暟
    # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
    parser.add_argument("--attacks", type=str, nargs='+',
                        default=None,
                        choices=['Clean', 'APGD_Linf', 'APGD_L2', 'SPSA', 'ACE', 'ReColorAdv', 'Hue', 'Light', 'UAA', 'SUB', 'STADV'],
                        help="Attack list for evaluation; defaults to attack_config when omitted")

    parser.add_argument("--subspace_basis_path", type=str, default="",
                        help="Path to SUB attack PCA bases; defaults to result_dir when omitted")
    parser.add_argument("--subspace_rank", type=int, default=128,
                        help="SUB 鏀诲嚮 PCA rank")
    parser.add_argument("--subspace_max_per_class", type=int, default=600,
                        help="鏋勫缓 SUB PCA bases 鏃舵瘡绫绘渶澶氶噰鏍锋暟")

    # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
    #  鏂板锛氭敾鍑婚厤缃拰妯″瀷鏋舵瀯閫夋嫨
    # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
    parser.add_argument("--attack_config", type=str, default='v10',
                        choices=['v10', 'v20'],
                        help="鏀诲嚮閰嶇疆: v10=APGD_Linf/APGD_L2/ACE/Hue/ReColorAdv/Light/UAA(7绉?; v20=APGD_Linf/APGD_L2/ACE/SUB/STADV(5绉?")
    parser.add_argument("--backbone", type=str, default='resnet',
                        choices=['resnet', 'mobilevit'],
                        help="妯″瀷楠ㄦ灦: resnet 鎴?mobilevit")
    parser.add_argument("--mvit_fdconv", action="store_true",
                        help="mobilevit: 鎶婃瘡涓?MobileViTBlock 鐨勫叆鍙ｅ嵎绉?conv1 鎹㈡垚 FDConv(鎸夊煙璺敱鐨勬弧鍗风Н涓撳), "
                             "attention remains shared; effective when backbone=mobilevit.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index, e.g. 0, 1, 2")

    args = parser.parse_args()

    # 鈽呪槄鈽?璁剧疆 GPU
    if torch.cuda.is_available() and args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    # 鈽呪槄鈽?鏍规嵁鏀诲嚮閰嶇疆鑷姩璁剧疆 num_sources
    attack_cfg = ATTACK_CONFIGS[args.attack_config]
    args.num_sources = attack_cfg['num_sources']
    args.train_attacks = list(attack_cfg['train_attacks'])
    args.test_attacks = list(attack_cfg['test_attacks'])
    args.domain_names = list(attack_cfg['domain_names'])
    args.subspace_bases = None

    if _fixed_domain_route_enabled(args):
        if args.ablate_domain_route_id < 0 or args.ablate_domain_route_id >= args.domains:
            raise ValueError(
                f"ablate_domain_route_id={args.ablate_domain_route_id} is outside [0, {args.domains - 1}]"
            )
    if _domain_route_ablation_enabled(args):
        if args.domain_loss_weight != 0.0:
            print("[Ablation] Forcing domain_loss_weight=0.0 for domain-route ablation.")
            args.domain_loss_weight = 0.0

    if not args.attacks:
        args.attacks = list(args.test_attacks)

    # 鈽呪槄鈽?鑷姩鎺ㄦ柇 n_cls锛圕IFAR-100鈫?00锛汿iny/CIFAR-10鈫?0锛涘惁鍒欎繚鐣欏懡浠よ鍊硷級
    inferred_n = _infer_num_classes(args.dataset, fallback=args.n_cls)
    if inferred_n != args.n_cls:
        print(f"[Info] n_cls changed from CLI value {args.n_cls} to {inferred_n} based on dataset='{args.dataset}'")
        args.n_cls = inferred_n

    # 鈽呪槄鈽?鑷姩鐢熸垚 result_dir锛堣嫢鏈寚瀹氾級鈥斺€?瑙勮寖鍛藉悕: 鏁版嵁闆?backbone/v鏀诲嚮閰嶇疆/k鍩熸暟/bs/lr/ep/seed
    if not args.result_dir:
        ablation_tag = ""
        if _domain_route_ablation_enabled(args):
            if _fixed_domain_route_enabled(args):
                ablation_tag = f"_ablateRoute-{args.ablate_domain_route}r{args.ablate_domain_route_id}"
            else:
                ablation_tag = f"_ablateRoute-{args.ablate_domain_route}"
        args.result_dir = (
            f"./results/tafd_{args.dataset}_{args.backbone}_{args.attack_config}"
            f"_k{args.domains}_bs{args.batch_size}_lr{args.lr}"
            f"_ep{args.end_epoch}_seed{args.seed}{ablation_tag}"
        )

    print(f"[Auto-Config] Dataset: {args.dataset}")
    print(f"[Auto-Config] n_cls: {args.n_cls}")
    print(f"[Auto-Config] 鏀诲嚮閰嶇疆: {args.attack_config} ({attack_cfg['num_sources']}绉嶆敾鍑?")
    print(f"[Auto-Config] 璁粌鏀诲嚮: {args.train_attacks}")
    print(f"[Auto-Config] 妯″瀷楠ㄦ灦: {args.backbone}")
    print(f"[Auto-Config] 閫夋嫨鐨勮瘎浼版敾鍑? {args.attacks}")
    print(f"[Auto-Config] result_dir: {args.result_dir}")

    os.makedirs(args.result_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    main(args)

