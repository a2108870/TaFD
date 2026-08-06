#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_woorth_firstblock_FDConv10_mix_randT.py
--------------------------------------------
基于 main_woorth_firstblock_FDConv10_mix.py 的改进版本。

核心改动：
  - 验证时：使用当前评估配置
  - SPSA 仅在 epoch=50 和 75 时评估

使用方法：
  python main_woorth_firstblock_FDConv10_mix_randT.py --dataset CIFAR100
"""

import os

import random
import argparse
import numpy as np
from tqdm import tqdm

# ─── PyTorch & TorchVision ────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
# 修正：在导入 pyplot 之前，强制使用 'Agg' 后端（避免多进程与 tkinter 冲突）
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── 第三方 / 自定义库 ─────────────────────────────────────────────────────
from utils.utils import *
from utils.datasets_utils import GetDataLoader
from torchattacks import APGD, PGD, PGDL2, SPSA

# V10 攻击
from attacks.ace import ACE
from attacks.recoloradv import ReColorAdv
from attacks.light import light_atk
from attacks.hue import hue_atk
from attacks.uaa import UAA_atk

# V20 攻击
from attacks.subspace import subspace_atk, build_pca_bases, save_pca_bases, load_pca_bases
from attacks.stadv import stadv_attack

# 整合模型（directMask 频域掩码消融版本）
from models.encoder_directMask import create_encoder

# -------------------------------------------------------------------------
# 全局设备（由 --gpu 参数在 main 入口处覆盖）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# -------------------------------------------------------------------------

# ══════════════════════════════════════════════════════════════════════════
#  攻击配置注册表
# ══════════════════════════════════════════════════════════════════════════
ATTACK_CONFIGS = {
    'v10': {
        'train_attacks': ['APGD_Linf', 'APGD_L2', 'ACE', 'Hue', 'ReColorAdv', 'Light', 'UAA'],
        'test_attacks': ['Clean', 'APGD_Linf', 'APGD_L2', 'ACE', 'ReColorAdv', 'Hue', 'Light', 'UAA'],
        'num_sources': 7,
        'domain_names': ['APGD_Linf', 'APGD_L2', 'ACE', 'Hue', 'ReColorAdv', 'Light', 'UAA'],
    },
    'v20': {
        'train_attacks': ['APGD_Linf', 'APGD_L2', 'ACE', 'SUB', 'STADV'],
        'test_attacks': ['Clean', 'APGD_Linf', 'APGD_L2', 'ACE', 'SUB', 'STADV'],
        'num_sources': 5,
        'domain_names': ['APGD_Linf', 'APGD_L2', 'ACE', 'SUB', 'STADV'],
    }
}

# 可用攻击注册表（合并 V10 和 V20）
ALL_ATTACKS = ['Clean', 'APGD_Linf', 'APGD_L2', 'ACE', 'ReColorAdv', 'Hue', 'Light', 'UAA', 'SUB', 'STADV']
DOMAIN_ATTACKS = ['APGD_Linf', 'APGD_L2', 'ACE', 'ReColorAdv', 'Hue', 'Light', 'UAA', 'SUB', 'STADV']
DOMAIN_SUPERVISED_ATTACKS = list(DOMAIN_ATTACKS)

# ══════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════
SPSA_EVAL_EPOCHS = set()

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


# ══════════════════════════════════════════════════════════════════════════
#  工具函数：学习率、计量器、准确率、图像归一化
# ══════════════════════════════════════════════════════════════════════════
def set_group_lrs(optimizer: optim.Optimizer, lr_main: float, lr_domain: float):
    """为不同 param_group 设置学习率（组需含 'name': 'main' / 'domain'）。"""
    for pg in optimizer.param_groups:
        if pg.get("name") == "domain":
            pg["lr"] = lr_domain
            pg["initial_lr"] = lr_domain
        else:
            pg["lr"] = lr_main
            pg["initial_lr"] = lr_main


def adjust_learning_rate(optimizer, epoch, lr_main_init=0.001, lr_domain_init=0.003):
    """两组一起按相同 schedule 衰减，保持比例不变。"""
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
    """返回 {'main': lr, 'domain': lr}；若未命名则按顺序命名。"""
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
        # 如果是 Tensor，保持在 GPU 上累加，避免同步
        if isinstance(val, torch.Tensor):
            val = val.detach()
            # 确保是标量 tensor
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
        # 仅在需要显示时才同步回 CPU
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


# === 动态类平衡权重（保留实现，按需启用）===
def compute_balanced_ce_weight(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float().to(labels.device)
    counts = torch.clamp(counts, min=1.0)
    inv = 1.0 / counts
    weights = inv / inv.mean()  # 平均权重=1
    return weights


# ══════════════════════════════════════════════════════════════════════════
#  绘图：准确率曲线
# ══════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════
#  对抗样本生成（含：Tiny 时 lr ×0.1；且 ncls 随数据集传入）
# ══════════════════════════════════════════════════════════════════════════
def generate_single_attack(attack_func, img, tgt, model, device,
                           attack_name="Unknown", **kwargs):
    try:
        if attack_name in ('APGD_Linf', 'APGD_L2', 'SPSA'):
            return attack_func(img, tgt)
        else:
            return attack_func(img, tgt, model, device, **kwargs)
    except Exception as e:
        print(f"攻击失败 {attack_name}: {e}")
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
    根据 attack_names 生成对抗样本字典 {attack_name: adv_x}。
    支持 V10 与 V20 统一调用。
    """
    attack_inputs = {}

    for name in attack_names:
        if name == 'Clean':
            attack_inputs['Clean'] = img
        elif name == 'APGD_Linf':
            if atk_apgd_linf is None:
                raise ValueError("APGD_Linf 攻击对象未初始化")
            attack_inputs['APGD_Linf'] = generate_single_attack(atk_apgd_linf, img, tgt, model, device, "APGD_Linf")
        elif name == 'APGD_L2':
            if atk_apgd_l2 is None:
                raise ValueError("APGD_L2 攻击对象未初始化")
            attack_inputs['APGD_L2'] = generate_single_attack(atk_apgd_l2, img, tgt, model, device, "APGD_L2")
        elif name == 'SPSA':
            if atk_spsa is None:
                raise ValueError("SPSA 攻击对象未初始化")
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
                raise ValueError("SUB 攻击需要 PCA bases，请先构建或加载")
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
            print(f"[WARN] 未支持的攻击: {name}，已跳过")

    return attack_inputs


# ══════════════════════════════════════════════════════════════════════════
#  辅助：字符串解析 + tqdm 映射展示函数
# ══════════════════════════════════════════════════════════════════════════
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
    [修改] 修正逻辑，确保 10class 优先匹配
    规则：
    - 含 '200class' → 200
    - 含 '100' → 100
    - 含 '10class' → 10 (Tiny_32_10class)
    - 含 'cifar10' → 10
    - 含 'tiny' (且未命中 10class) → 200 (Default Tiny ImageNet)
    - 否则用 fallback
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
    # [关键修复] 优先匹配 10class，防止被 tiny 抢占
    if '10class' in name:
        return 10
    if ('cifar10' in name) or ('cifar-10' in name):
        return 10

    # Tiny ImageNet 默认是 200 类
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
    """按需构建/加载 SUB 攻击所需的 PCA bases。"""
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
        print(f"[SUB] 加载 PCA bases: {basis_path}")
        args.subspace_bases = load_pca_bases(basis_path, device=device)
        return

    print("[SUB] 未找到 PCA bases，开始构建（可能耗时）...")
    args.subspace_bases = build_pca_bases(
        trainloader,
        n_classes=args.n_cls,
        rank=args.subspace_rank,
        max_per_class=args.subspace_max_per_class,
        device=torch.device('cpu')
    )
    os.makedirs(os.path.dirname(basis_path) or '.', exist_ok=True)
    save_pca_bases(basis_path, args.subspace_bases)
    print(f"[SUB] PCA bases 已保存: {basis_path}")


def _should_batch_eval_forwards(dataset_name: str) -> bool:
    return _infer_input_size(dataset_name) <= 64


# ══════════════════════════════════════════════════════════════════════════
#  训练：统一用"预测 domain"进行路由 + 每 N iter 动态聚类更新
# ══════════════════════════════════════════════════════════════════════════
def train(epoch, trainloader, criterion, optimizer, n_classes, model,
          initial_lr_main=0.001, initial_lr_domain=0.003, args=None, global_iter_start=0):
    adjust_learning_rate(optimizer, epoch, lr_main_init=initial_lr_main, lr_domain_init=initial_lr_domain)
    lr_main, lr_domain = get_current_lrs(optimizer)
    print(
        f"\nEpoch {epoch:03d} | lr_main={lr_main:.6f} | lr_domain={lr_domain:.6f} | domains={args.domains}")

    losses = {k: AverageMeter() for k in ['cls', 'domain', 'total']}
    domains = list(args.train_attacks)

    # [修改] 训练时分别记录分类准确率和域准确率
    train_cls_accs = {d: AverageMeter() for d in domains}
    train_dom_accs = {d: AverageMeter() for d in domains}

    domain_acc_meter = AverageMeter()

    # ★ 记录本 epoch 使用的温度统计

    # ★ 模型内部已处理标准化，torchattacks 不需要再设置
    atk_apgd_linf = _build_train_pgd_attack(model, norm='Linf', eps=8 / 255, steps=10)
    atk_apgd_l2 = _build_train_pgd_attack(model, norm='L2', eps=0.5, steps=10)

    model.train()
    pbar = tqdm(trainloader, leave=True, desc=f"Train {epoch}")
    global_iter = global_iter_start

    # ★ tiny 数据集时缩放颜色/光照类攻击 lr
    lr_scale = 0.1 if _is_tiny_dataset(getattr(args, "dataset", None)) else 1.0

    for step_idx, (img, tgt) in enumerate(pbar, start=1):
        img = img.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        bsz = img.size(0)
        source_ids_per_attack = torch.arange(len(domains), device=device, dtype=torch.long).repeat_interleave(bsz)

        # 生成对抗样本（传入动态 n_classes）
        model.eval()

        # =========== 【核心改动】 ===========

        # 2. 开启 BPDA，让攻击能够穿透 DomainClassifier
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

        # =========== 【修改】 ===========
        # 3. 攻击生成完毕，关闭 BPDA，保证训练权重时使用的是真实的 Hard Routing
        if hasattr(model, "set_bpda"):
            model.set_bpda(False)

        imgs_mix = [attack_inputs[d] for d in domains]

        global_iter += 1
        if epoch < 10:
            if global_iter % args.map_update_every == 0 or global_iter == 1:
                with torch.no_grad():
                    model.extract_wavelet_features(torch.cat(imgs_mix, dim=0), source_ids_per_attack)

        if global_iter % args.map_update_every == 0 or global_iter == 1:
            model.update_domain_mappings(epoch, args.end_epoch)

        domain_labels = source_ids_per_attack
        true_domain_labels = model.get_domain_labels(domain_labels)

        model.train()
        combined = torch.cat(imgs_mix, 0).detach()

        outputs = model(combined, None, criterion, bsz, flag=1, domain_ids=domain_labels)
        cls_logits, _, domain_logits, _ = outputs

        # domain ??
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

        # tqdm 展示: 实时显示
        if step_idx % args.log_every == 0:
            status = model.get_all_mapping_statuses().get("global_mapping", {})
            pbar.set_postfix({
                "loss": f"{losses['cls'].avg:.3f}",
                "map": mapping_status_str(status, order=args.domain_names),
                **{ATTACK_SHORT_NAMES.get(d, d): (f"{train_cls_accs[d].avg:.1f}/{train_dom_accs[d].avg:.1f}" if d in DOMAIN_SUPERVISED_ATTACKS and train_dom_accs[d].count > 0 else f"{train_cls_accs[d].avg:.1f}/-") for d in domains},
            })

    pbar.close()
    return global_iter


# ══════════════════════════════════════════════════════════════════════════
#  验证：使用当前评估配置
#  ★ 核心改动：SPSA 仅在指定 epoch 评估
# ══════════════════════════════════════════════════════════════════════════
def validation_pgd(epoch, testloader, criterion, model, n_cls,
                   acc_hist=None, domain_hist=None, args=None):
    """
    验证函数，支持可配置的攻击选择。

    Args:
        args.attacks: 要评估的攻击列表，例如 ['Clean', 'APGD_Linf', 'APGD_L2', 'SPSA', ...]
    """
    # 获取选择的攻击
    selected_attacks = args.attacks if args and hasattr(args, 'attacks') else ALL_ATTACKS
    if 'SUB' in selected_attacks and getattr(args, 'subspace_bases', None) is None:
        raise RuntimeError("选择了 SUB 评估，但未准备 PCA bases")

    print(f"\n[Validation] 选择的攻击: {selected_attacks}")

    # ★ 为每个温度创建独立的指标存储

    selected_attacks_runtime = list(selected_attacks)
    if 'SPSA' in selected_attacks_runtime and epoch not in SPSA_EVAL_EPOCHS:
        selected_attacks_runtime.remove('SPSA')
        print(f"[Validation] 跳过 SPSA（仅在 epoch {sorted(SPSA_EVAL_EPOCHS)} 评估）")

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

        # 为当前温度初始化指标
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

            # ========== ★ 攻击生成阶段：使用当前评估温度 ==========
            if hasattr(model, "set_bpda"):
                model.set_bpda(True)

            # ─── 根据选择的攻击生成对抗样本 ──────────────────────────────
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
            for name in attack_inputs.keys():
                if name not in DOMAIN_ATTACKS:
                    continue
                domain_idx = _domain_id_from_attack(name, args.domain_names)
                if domain_idx is None:
                    continue
                domain_ids[name] = torch.full((bsz,), domain_idx, device=device, dtype=torch.long)

            # ========== ★ 攻击生成完：关闭 BPDA（评估 forward 保持原逻辑） ==========
            if hasattr(model, "set_bpda"):
                model.set_bpda(False)

            # ─── 评估每个攻击 ────────────────────────────────────────────
            with torch.no_grad():
                if batch_eval_forwards and attack_inputs:
                    ordered_names = list(attack_inputs.keys())
                    combined_inputs = torch.cat([attack_inputs[name] for name in ordered_names], dim=0)
                    out_tuple = model(combined_inputs, domain_ids=None)
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
                        out_tuple = model(x_in, domain_ids=None)  # 统一用预测路由
                        logits = out_tuple[0] if isinstance(out_tuple, tuple) else out_tuple
                        acc1 = accuracy(logits, tgt, (1,))[0]
                        metrics[name].update(acc1.item(), bsz)

                        # 域准确率（仅有对应真域标签时统计）
                        if name in domain_ids and isinstance(out_tuple, tuple) and len(out_tuple) > 2:
                            domain_logit = out_tuple[2]
                            pred_domain = domain_logit.max(1)[1]
                            true_domain = model.get_domain_labels(domain_ids[name])
                            d_acc = (pred_domain == true_domain).float().mean() * 100
                            domain_accs[name].update(d_acc.item(), bsz)

                            _update_confusion_matrix(cm, true_domain, pred_domain)

            # ─── 构建 tqdm postfix: Attack=ClsAcc/DomAcc ────────────────────────
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

        # ─── 打印当前温度的验证汇总 ────────────────────────────────────────────
        print(f"\n{'='*70}")
        print(f"验证汇总 T={eval_T} (分类准确率 / 域分类准确率)")
        print("=" * 70)
        print(f"{'攻击':<15} {'分类准确率':>12} {'域准确率':>12} {'格式':>15}")
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
            print(f"\n域混淆矩阵 T={eval_T} (行=真实, 列=预测):")
            print(cm.cpu().numpy())

        # 存储当前温度的结果
        
        

    # ═══════════════════════════════════════════════════════════════════════
    #  打印当前温度汇总
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*80}")
    print(f"  ★★★ 验证汇总 (Epoch {epoch}) ★★★")
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


# ══════════════════════════════════════════════════════════════════════════
#  Checkpoint I/O
# ══════════════════════════════════════════════════════════════════════════
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
    # 允许与当前构造略有不一致（例如 domains 变动）
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        print(f"[WARN] state_dict 不完全匹配，missing={len(missing)}, unexpected={len(unexpected)}")
    try:
        optimizer.load_state_dict(ckpt["optimizer"])
    except Exception as e:
        print(f"[WARN] 加载优化器状态失败（分组结构不同？）已忽略：{e}")
    set_group_lrs(optimizer, base_main_lr, base_domain_lr)
    print(f"=> 恢复自 epoch {ckpt['epoch']}，学习率已重设为 main={base_main_lr} / domain={base_domain_lr}")
    return ckpt["epoch"] + 1, ckpt.get("accuracy_history"), ckpt.get("domain_acc_history")


# ══════════════════════════════════════════════════════════════════════════
#  主函数
# ══════════════════════════════════════════════════════════════════════════
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
    print("==> 准备数据 …")
    trainloader, testloader = GetDataLoader(args.dataset,
                                            args.batch_size,
                                            args.test_batch_size,
                                            args.dataset_path,
                                            num_workers=args.num_workers,
                                            pin_memory=not args.disable_pin_memory,
                                            persistent_workers=not args.disable_persistent_workers,
                                            prefetch_factor=args.prefetch_factor)

    # SUB 攻击所需 PCA bases（按需准备）
    prepare_subspace_bases(args, trainloader)

    print("==> 构建模型 …")
    # 使用 create_encoder 工厂函数，根据 backbone 和 attack_config 创建模型
    # 传入 dataset 参数，模型内部会自动选择正确的标准化参数
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

    model = create_encoder(**model_kwargs).to(device)
    model.count_frequency_convolutions()

    # ── 分组优化器：threat_domain_classifier 用更大学习率 ─────────────────────
    domain_params = list(model.threat_domain_classifier.parameters())
    domain_param_ids = set(id(p) for p in domain_params)
    main_params = [p for p in model.parameters() if id(p) not in domain_param_ids]

    optimizer = optim.Adam([
        {"params": main_params, "lr": args.lr, "weight_decay": args.weight_decay, "name": "main"},
        {"params": domain_params, "lr": args.lr_domain, "weight_decay": args.weight_decay, "name": "domain"},
    ])
    set_group_lrs(optimizer, args.lr, args.lr_domain)

    # 根据选择的攻击初始化历史记录
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
    print(f"初始学习率 main={args.lr} | domain={args.lr_domain} | 调度: 0-49×1, 50-74×0.1, ≥75×0.01")
    print(f"domain_loss_weight: {args.domain_loss_weight}")
    print(f"map_update_every: {args.map_update_every} iters")
    print(f"domains(聚类/域数)={args.domains}")
    print(f"训练攻击: {args.train_attacks}")
    print(
        f"attack_lr_scale: {'0.1 (tiny)' if _is_tiny_dataset(args.dataset) else '1.0'}  ← 作用于 ACE/Hue/ReColorAdv/Light/UAA")
    print(f"选择的评估攻击: {args.attacks}")
    if getattr(args, 'subspace_bases', None) is not None:
        print(f"SUB PCA bases: {args.subspace_basis_path}")
    print("=" * 60)
    print(f"   - SPSA: 仅在 epoch ∈ {sorted(SPSA_EVAL_EPOCHS)} 时评估")
    print("=" * 60)

    last_completed_epoch = start_epoch - 1
    if args.resume and _should_run_resume_eval(last_completed_epoch, acc_hist, args):
        print(f"[Resume] 检测到 epoch {last_completed_epoch} 的评估/曲线缺失，先补做一次验证。")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Unified-LR Adversarial Training + PGD-Train APGD-Test + Fixed BPDA-T (directMask)")
    parser.add_argument("--dataset", type=str, default="Imagenette",
                        help="数据集名称（例如 Tiny_32_10class / Tiny_32_200class / CIFAR10 / CIFAR100 等）")
    parser.add_argument("--dataset_path", type=str, default="./datasets/",
                        help="Root directory containing downloaded datasets.")
    parser.add_argument("--lr", type=float, default=0.001, help="主干学习率")
    parser.add_argument("--lr_domain", type=float, default=0.001, help="domain_classifier 学习率（建议 3~10 倍）")
    parser.add_argument("--weight_decay", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--test_batch_size", type=int, default=16, help="??/?? batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader worker 数")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="DataLoader prefetch_factor")
    parser.add_argument("--disable_pin_memory", action="store_true", help="关闭 DataLoader pin_memory")
    parser.add_argument("--disable_persistent_workers", action="store_true", help="关闭 DataLoader persistent_workers")
    parser.add_argument("--log_every", type=int, default=10, help="每隔多少个 iter 更新一次 tqdm 指标")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--end_epoch", type=int, default=76)
    parser.add_argument("--eval_freq", type=int, default=5)
    parser.add_argument("--resume", type=str,
                        default="")
    parser.add_argument("--result_dir", type=str,
                        default="")
    # 保留 n_cls 参数，但会被自动推断覆盖
    parser.add_argument("--n_cls", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)

    # 域/聚类数（控制域分类器、聚类、专家数）
    parser.add_argument("--domains", type=int, default=6, help="domain/聚类数")

    # domain 监督权重
    parser.add_argument("--domain_loss_weight", type=float, default=1.0,
                        help="总损失中 domain 监督的权重系数")
    # 动态聚类频率（迭代步）
    parser.add_argument("--map_update_every", type=int, default=50)

    # ══════════════════════════════════════════════════════════════════════════
    #  攻击选择参数
    # ══════════════════════════════════════════════════════════════════════════
    parser.add_argument("--attacks", type=str, nargs='+',
                        default=None,
                        choices=['Clean', 'APGD_Linf', 'APGD_L2', 'SPSA', 'ACE', 'ReColorAdv', 'Hue', 'Light', 'UAA', 'SUB', 'STADV'],
                        help="要评估的攻击列表；留空时随 --attack_config 自动设置。")

    parser.add_argument("--subspace_basis_path", type=str, default="",
                        help="SUB 攻击 PCA bases 路径；留空则自动放在 result_dir 下")
    parser.add_argument("--subspace_rank", type=int, default=128,
                        help="SUB 攻击 PCA rank")
    parser.add_argument("--subspace_max_per_class", type=int, default=600,
                        help="构建 SUB PCA bases 时每类最多采样数")

    # ══════════════════════════════════════════════════════════════════════════
    #  新增：攻击配置和模型架构选择
    # ══════════════════════════════════════════════════════════════════════════
    parser.add_argument("--attack_config", type=str, default='v10',
                        choices=['v10', 'v20'],
                        help="攻击配置: v10=APGD_Linf/APGD_L2/ACE/Hue/ReColorAdv/Light/UAA(7种); v20=APGD_Linf/APGD_L2/ACE/SUB/STADV(5种)")
    parser.add_argument("--backbone", type=str, default='resnet',
                        choices=['resnet', 'mobilevit'],
                        help="模型骨架: resnet 或 mobilevit")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU 编号（例如 0, 1, 2 ...）")

    args = parser.parse_args()

    # ★★★ 设置 GPU
    if torch.cuda.is_available() and args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    # ★★★ 根据攻击配置自动设置 num_sources
    attack_cfg = ATTACK_CONFIGS[args.attack_config]
    args.num_sources = attack_cfg['num_sources']
    args.train_attacks = list(attack_cfg['train_attacks'])
    args.test_attacks = list(attack_cfg['test_attacks'])
    args.domain_names = list(attack_cfg['domain_names'])
    args.subspace_bases = None

    if not args.attacks:
        args.attacks = list(args.test_attacks)

    # ★★★ 自动推断 n_cls（CIFAR-100→100；Tiny/CIFAR-10→10；否则保留命令行值）
    inferred_n = _infer_num_classes(args.dataset, fallback=args.n_cls)
    if inferred_n != args.n_cls:
        print(f"[Info] n_cls 从命令行值 {args.n_cls} 自动更新为 {inferred_n}（基于 dataset='{args.dataset}'）")
        args.n_cls = inferred_n

    # ★★★ 自动生成 result_dir（若未指定）
    if not args.result_dir:
        args.result_dir = (
            f"./results/tafd_pgdtrain_apgdtest_directMask_{args.backbone}_{args.dataset}_d{args.domains}"
            f"_{args.attack_config}_lr{args.lr}_dlr{args.lr_domain}"
            f"_dw{args.domain_loss_weight}_bs{args.batch_size}"
            f"_ep{args.end_epoch}_seed{args.seed}"
        )

    print("[Ablation] FC-Conv removes Zernike-basis masks and directly learns full spectral masks.")
    print(f"[Auto-Config] Dataset: {args.dataset}")
    print(f"[Auto-Config] n_cls: {args.n_cls}")
    print(f"[Auto-Config] 攻击配置: {args.attack_config} ({attack_cfg['num_sources']}种攻击)")
    print(f"[Auto-Config] 训练攻击: {args.train_attacks}")
    print(f"[Auto-Config] 模型骨架: {args.backbone}")
    print(f"[Auto-Config] 选择的评估攻击: {args.attacks}")
    print(f"[Auto-Config] result_dir: {args.result_dir}")

    os.makedirs(args.result_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    main(args)
