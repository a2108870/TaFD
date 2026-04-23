#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py
--------
TaFD: Threat-aware Frequency Decoupling for Heterogeneous Adversarial Robustness.

Main training script with periodic APGD evaluation.

Key design choices:
  - Training:    PGD-based adversarial examples with BPDA surrogate for hard routing
  - Evaluation:  APGD (100 steps) on the full test attack set post-training
  - Domain mapping updated every `--map_update_every` iterations via K-means + Hungarian alignment

Usage:
  python train.py --dataset CIFAR100 --backbone resnet --attack_config v10 --gpu 0
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
import matplotlib.pyplot as plt

# ─── Third-party / Custom libraries ─────────────────────────────────────────────────────
from utils.utils import *
from utils.datasets_utils import GetDataLoader
from torchattacks import APGD, PGD, PGDL2

# V10 Attacks
from attacks.ace import ACE
from attacks.recoloradv import ReColorAdv
from attacks.ala import ala_atk
from attacks.hsvadv import hsvadv_atk
from attacks.retouch_uaa import retouch_uaa_atk

# V20 Attacks
from attacks.gpgd import gpgd_atk, build_pca_bases, save_pca_bases, load_pca_bases
from attacks.stadv import stadv_attack

# Models
from models.encoder import create_encoder

# -------------------------------------------------------------------------
# Global device (overridden by --gpu in main)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# -------------------------------------------------------------------------

# ══════════════════════════════════════════════════════════════════════════
#  Attack configuration registry
# ══════════════════════════════════════════════════════════════════════════
ATTACK_CONFIGS = {
    'v10': {
        # train_attacks: standard PGD-10 used during adversarial training
        'train_attacks': ['Linf_PGD', 'L2_PGD', 'ACE', 'HSVAdv', 'ReColorAdv', 'ALA', 'RetouchUAA'],
        # test_attacks: Auto-PGD (APGD-100) used during evaluation — stronger than training PGD
        'test_attacks': ['Clean', 'Linf_APGD', 'L2_APGD', 'ACE', 'ReColorAdv', 'HSVAdv', 'ALA', 'RetouchUAA'],
        'num_sources': 7,
        'domain_names': ['Linf_PGD', 'L2_PGD', 'ACE', 'HSVAdv', 'ReColorAdv', 'ALA', 'RetouchUAA'],
    },
    'v20': {
        'train_attacks': ['Linf_PGD', 'L2_PGD', 'ACE', 'GPGD', 'StAdv'],
        'test_attacks': ['Clean', 'Linf_APGD', 'L2_APGD', 'ACE', 'GPGD', 'StAdv'],
        'num_sources': 5,
        'domain_names': ['Linf_PGD', 'L2_PGD', 'ACE', 'GPGD', 'StAdv'],
    }
}

# Available attacks registry (V10 and V20 merged)
# Training-phase attacks (standard PGD-10)
ALL_TRAIN_ATTACKS = ['Linf_PGD', 'L2_PGD', 'ACE', 'ReColorAdv', 'HSVAdv', 'ALA', 'RetouchUAA', 'GPGD', 'StAdv']
# Evaluation-phase attacks (APGD-100 for Linf/L2, same impl for others)
ALL_ATTACKS = ['Clean', 'Linf_PGD', 'L2_PGD', 'Linf_APGD', 'L2_APGD',
               'ACE', 'ReColorAdv', 'HSVAdv', 'ALA', 'RetouchUAA', 'GPGD', 'StAdv']
DOMAIN_ATTACKS = ['Linf_PGD', 'L2_PGD', 'Linf_APGD', 'L2_APGD',
                  'ACE', 'ReColorAdv', 'HSVAdv', 'ALA', 'RetouchUAA', 'GPGD', 'StAdv']
DOMAIN_SUPERVISED_ATTACKS = list(DOMAIN_ATTACKS)


ATTACK_SHORT_NAMES = {
    # Training PGD (standard PGD-10)
    'Linf_PGD':  'PGD_L',
    'L2_PGD':    'PGD_2',
    # Evaluation APGD (Auto-PGD-100)
    'Linf_APGD': 'AP_L',
    'L2_APGD':   'AP_2',
    'ACE': 'ACE',
    'HSVAdv': 'HSVAdv',
    'ReColorAdv': 'ReC',
    'ALA': 'ALA',
    'RetouchUAA': 'RetouchUAA',
    'GPGD': 'GPGD',
    'StAdv': 'StA',
    'Clean': 'Clean',
}




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
#  Utility functions: learning rate, meters, accuracy, normalization
# ══════════════════════════════════════════════════════════════════════════
def set_group_lrs(optimizer: optim.Optimizer, lr_main: float, lr_domain: float):
    """Set learning rates for different param_groups (requires 'name': 'main' / 'domain')."""
    for pg in optimizer.param_groups:
        if pg.get("name") == "domain":
            pg["lr"] = lr_domain
            pg["initial_lr"] = lr_domain
        else:
            pg["lr"] = lr_main
            pg["initial_lr"] = lr_main


def adjust_learning_rate(optimizer, epoch, lr_main_init=0.001, lr_domain_init=0.003):
    """Decay both groups according to the same schedule to maintain their ratio."""
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
    """Return {'main': lr, 'domain': lr}; sequentially named if unnamed."""
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
        # If Tensor, accumulate on GPU to avoid sync overhead
        if isinstance(val, torch.Tensor):
            val = val.detach()
            # Ensure scalar tensor
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
        # Sync to CPU only when displaying
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



# ══════════════════════════════════════════════════════════════════════════
#  Plotting: Accuracy curves
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
#  Adversarial sample generation
# ══════════════════════════════════════════════════════════════════════════
def generate_single_attack(attack_func, img, tgt, model, device,
                           attack_name="Unknown", **kwargs):
    try:
        # PGD/APGD are torchattacks objects: call as attack_func(img, tgt)
        if attack_name in ('Linf_PGD', 'L2_PGD', 'Linf_APGD', 'L2_APGD'):
            return attack_func(img, tgt)
        else:
            return attack_func(img, tgt, model, device, **kwargs)
    except Exception as e:
        print(f"Attack failed {attack_name}: {e}")
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
                          atk_apgd_linf=None, atk_apgd_l2=None,
                          lr_scale=1.0, ncls=100, subspace_bases=None):
    """
    Generate a dictionary of adversarial examples {attack_name: adv_x}.
    Supports unified calling for both V10 and V20.
    """
    attack_inputs = {}

    for name in attack_names:
        if name == 'Clean':
            attack_inputs['Clean'] = img
        elif name == 'Linf_PGD':
            # Training: standard PGD-10 (atk_pgd_linf)
            if atk_apgd_linf is None:
                raise ValueError("Linf_PGD attack object not initialized")
            attack_inputs['Linf_PGD'] = generate_single_attack(atk_apgd_linf, img, tgt, model, device, "Linf_PGD")
        elif name == 'L2_PGD':
            if atk_apgd_l2 is None:
                raise ValueError("L2_PGD attack object not initialized")
            attack_inputs['L2_PGD'] = generate_single_attack(atk_apgd_l2, img, tgt, model, device, "L2_PGD")
        elif name == 'Linf_APGD':
            # Evaluation: Auto-PGD-100 (atk_apgd_linf)
            if atk_apgd_linf is None:
                raise ValueError("Linf_APGD attack object not initialized")
            attack_inputs['Linf_APGD'] = generate_single_attack(atk_apgd_linf, img, tgt, model, device, "Linf_APGD")
        elif name == 'L2_APGD':
            if atk_apgd_l2 is None:
                raise ValueError("L2_APGD attack object not initialized")
            attack_inputs['L2_APGD'] = generate_single_attack(atk_apgd_l2, img, tgt, model, device, "L2_APGD")

        elif name == 'ACE':
            attack_inputs['ACE'] = generate_single_attack(
                ACE, img, tgt, model, device, "ACE",
                lr=1 * lr_scale, max_iterations=10, ncls=ncls
            )
        elif name == 'HSVAdv':
            attack_inputs['HSVAdv'] = generate_single_attack(
                hsvadv_atk, img, tgt, model, device, "HSVAdv",
                lr=1 * lr_scale, max_iterations=10, ncls=ncls
            )
        elif name == 'ReColorAdv':
            attack_inputs['ReColorAdv'] = generate_single_attack(
                ReColorAdv, img, tgt, model, device, "ReColorAdv",
                lr=0.01 * lr_scale, max_iterations=10, ncls=ncls
            )
        elif name == 'ALA':
            attack_inputs['ALA'] = generate_single_attack(
                ala_atk, img, tgt, model, device, "ALA",
                lr=1 * lr_scale, max_iterations=10, ncls=ncls
            )
        elif name == 'RetouchUAA':
            attack_inputs['RetouchUAA'] = generate_single_attack(
                retouch_uaa_atk, img, tgt, model, device, "RetouchUAA",
                lr=0.1 * lr_scale, max_iterations=10, ncls=ncls
            )
        elif name == 'GPGD':
            if subspace_bases is None:
                raise ValueError("GPGD attack requires PCA bases, please build or load first")
            attack_inputs['GPGD'] = generate_single_attack(
                gpgd_atk, img, tgt, model, device, "GPGD",
                bases_dict=subspace_bases, steps=10, epsilon=2.0, ncls=ncls, proj='l2'
            )
        elif name == 'StAdv':
            attack_inputs['StAdv'] = generate_single_attack(
                stadv_attack, img, tgt, model, device, "StAdv",
                eps=0.045, steps=10, mode='linf'
            )
        else:
            print(f"[WARN] Unsupported attack: {name}, skipped")

    return attack_inputs


# ══════════════════════════════════════════════════════════════════════════
#  Helpers: String parsing & tqdm mapping format
# ══════════════════════════════════════════════════════════════════════════


def _infer_num_classes(dataset_name: str, fallback: int = 10) -> int:
    """Infer number of classes from dataset name."""
    if dataset_name is None:
        return fallback
    name = str(dataset_name).lower()
    if ('cifar100' in name) or ('cifar-100' in name):
        return 100
    if ('cifar10' in name) or ('cifar-10' in name):
        return 10
    return fallback


def mapping_status_str(status_dict: dict, order=None) -> str:
    if order is None:
        order = ['Linf_PGD', 'L2_PGD', 'ACE', 'HSVAdv', 'ReColorAdv', 'ALA', 'RetouchUAA']
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
    """Build/load PCA bases for GPGD attack as needed."""
    need_sub = ('GPGD' in getattr(args, 'train_attacks', [])) or ('GPGD' in getattr(args, 'attacks', []))
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
        print(f"[GPGD] Loaded PCA bases: {basis_path}")
        args.subspace_bases = load_pca_bases(basis_path, device=device)
        return

    print("[GPGD] PCA bases not found, building (may take time)...")
    args.subspace_bases = build_pca_bases(
        trainloader,
        n_classes=args.n_cls,
        rank=args.subspace_rank,
        max_per_class=args.subspace_max_per_class,
        device=torch.device('cpu')
    )
    os.makedirs(os.path.dirname(basis_path) or '.', exist_ok=True)
    save_pca_bases(basis_path, args.subspace_bases)
    print(f"[GPGD] PCA bases saved: {basis_path}")



# ══════════════════════════════════════════════════════════════════════════
#  Training: Predict domain for routing + update clustering every N iters
# ══════════════════════════════════════════════════════════════════════════
def train(epoch, trainloader, criterion, optimizer, n_classes, model,
          initial_lr_main=0.001, initial_lr_domain=0.003, args=None, global_iter_start=0):
    adjust_learning_rate(optimizer, epoch, lr_main_init=initial_lr_main, lr_domain_init=initial_lr_domain)
    lr_main, lr_domain = get_current_lrs(optimizer)
    print(
        f"\nEpoch {epoch:03d} | lr_main={lr_main:.6f} | lr_domain={lr_domain:.6f} | domains={args.domains}")

    losses = {k: AverageMeter() for k in ['cls', 'domain', 'total']}
    domains = list(args.train_attacks)

    # [Modified] Track classification and domain accuracy separately during training
    train_cls_accs = {d: AverageMeter() for d in domains}
    train_dom_accs = {d: AverageMeter() for d in domains}

    domain_acc_meter = AverageMeter()

    # Training: standard PGD-10 (NOT APGD — lighter attack for efficient multi-attack training)
    atk_apgd_linf = _build_train_pgd_attack(model, norm='Linf', eps=8 / 255, steps=10)  # PGD-10, named atk_apgd_linf for API compat
    atk_apgd_l2 = _build_train_pgd_attack(model, norm='L2', eps=0.5, steps=10)          # PGD-10

    model.train()
    pbar = tqdm(trainloader, leave=True, desc=f"Train {epoch}")
    global_iter = global_iter_start

    # Scale color/light attack learning rates
    lr_scale = 1.0

    for step_idx, (img, tgt) in enumerate(pbar, start=1):
        img = img.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        bsz = img.size(0)
        source_ids_per_attack = torch.arange(len(domains), device=device, dtype=torch.long).repeat_interleave(bsz)

        # Generate adversarial examples
        model.eval()

        # Enable BPDA surrogate gradient to bypass hard routing
        if hasattr(model, "set_bpda"):
            model.set_bpda(True)

        attack_inputs = generate_attack_batch(
            img, tgt, model, device,
            attack_names=domains,
            atk_apgd_linf=atk_apgd_linf,
            atk_apgd_l2=atk_apgd_l2,
            lr_scale=lr_scale,
            ncls=n_classes,
            subspace_bases=getattr(args, 'subspace_bases', None)
        )

        # Disable BPDA surrogate after attack generation to train with hard routing
        if hasattr(model, "set_bpda"):
            model.set_bpda(False)

        imgs_mix = [attack_inputs[d] for d in domains]

        # Update global mapping every map_update_every steps
        global_iter += 1
        if epoch < 10:
            if global_iter % args.map_update_every == 0 or global_iter == 1:
                with torch.no_grad():
                    model.extract_spectral_features(torch.cat(imgs_mix, dim=0), source_ids_per_attack)

        # Update domain mappings
        if global_iter % args.map_update_every == 0 or global_iter == 1:
            model.update_domain_mappings(epoch, args.end_epoch)

        # Assign domain labels
        domain_labels = source_ids_per_attack
        true_domain_labels = model.get_domain_labels(domain_labels)

        # Forward pass with domain assignments
        model.train()
        combined = torch.cat(imgs_mix, 0).detach()

        outputs = model(combined, domain_ids=domain_labels)
        cls_logits, _, domain_logits, _ = outputs

        # Domain loss
        domain_loss = F.cross_entropy(domain_logits, true_domain_labels)
        losses['domain'].update(domain_loss, combined.size(0))

        with torch.no_grad():
            _, pred_domain = domain_logits.max(1)
            domain_acc = (pred_domain == true_domain_labels).float().mean() * 100
            domain_acc_meter.update(domain_acc, combined.size(0))

        # Classification loss and accuracy
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

        # Update tqdm progress bar
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
#  Validation
# ══════════════════════════════════════════════════════════════════════════
def validation_pgd(epoch, testloader, criterion, model, n_cls,
                   acc_hist=None, domain_hist=None, args=None):
    """
    Validation function with configurable attack selection.

    Args:
        args.attacks: List of attacks to evaluate, e.g. ['Clean', 'Linf_APGD', 'L2_APGD', ...]
                      Use 'Linf_APGD'/'L2_APGD' for APGD-100 evaluation (recommended).
                      Use 'Linf_PGD'/'L2_PGD' for standard PGD-10 (training-time attack only).
    """
    # Get selected attacks
    selected_attacks = args.attacks if args and hasattr(args, 'attacks') else ALL_ATTACKS
    if 'GPGD' in selected_attacks and getattr(args, 'subspace_bases', None) is None:
        raise RuntimeError("GPGD evaluation selected but PCA bases not prepared")

    print(f"\n[Validation] Selected attacks: {selected_attacks}")

    selected_attacks_runtime = list(selected_attacks)

    atk_apgd_linf = None
    atk_apgd_l2 = None
    # Evaluation: Auto-PGD-100 (stronger than training PGD-10)
    if 'Linf_APGD' in selected_attacks_runtime:
        atk_apgd_linf = _build_apgd_attack(model, norm='Linf', eps=8 / 255, steps=100)
    if 'L2_APGD' in selected_attacks_runtime:
        atk_apgd_l2 = _build_apgd_attack(model, norm='L2', eps=0.5, steps=100)

    metrics = {k: AverageMeter() for k in selected_attacks_runtime}
    domain_accs = {k: AverageMeter() for k in selected_attacks_runtime if k in DOMAIN_ATTACKS}

    cm = torch.zeros(model.num_threat_domains, model.num_threat_domains, dtype=torch.long)
    model.eval()

    pbar = tqdm(testloader, desc=f"Val Ep{epoch}", leave=True)

    lr_scale = 1.0
    batch_eval_forwards = True

    for step_idx, (img, tgt) in enumerate(pbar, start=1):
        img = img.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        bsz = img.size(0)

        # Enable BPDA surrogate gradient to bypass hard routing
        if hasattr(model, "set_bpda"):
            model.set_bpda(True)

        # ─── Generate adversarial examples ──────────────────────────────
        attack_inputs = generate_attack_batch(
            img, tgt, model, device,
            attack_names=selected_attacks_runtime,
            atk_apgd_linf=atk_apgd_linf,
            atk_apgd_l2=atk_apgd_l2,

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

        # Disable BPDA surrogate for evaluation forward pass
        if hasattr(model, "set_bpda"):
            model.set_bpda(False)

        # ─── Evaluate each attack ────────────────────────────────────────────
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
                    out_tuple = model(x_in, domain_ids=None)  # Uniformly use predicted routing
                    logits = out_tuple[0] if isinstance(out_tuple, tuple) else out_tuple
                    acc1 = accuracy(logits, tgt, (1,))[0]
                    metrics[name].update(acc1.item(), bsz)

                    # Domain accuracy (only if ground-truth domain label exists)
                    if name in domain_ids and isinstance(out_tuple, tuple) and len(out_tuple) > 2:
                        domain_logit = out_tuple[2]
                        pred_domain = domain_logit.max(1)[1]
                        true_domain = model.get_domain_labels(domain_ids[name])
                        d_acc = (pred_domain == true_domain).float().mean() * 100
                        domain_accs[name].update(d_acc.item(), bsz)

                        _update_confusion_matrix(cm, true_domain, pred_domain)

        # ─── Build tqdm postfix: Attack=ClsAcc/DomAcc ────────────────────────
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

    # ─── Print validation summary ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Validation Summary (Epoch {epoch})")
    print("=" * 70)
    print(f"{'Attack':<15} {'Cls Acc':>12} {'Dom Acc':>12}")
    print("-" * 70)
    for k in selected_attacks_runtime:
        if k in metrics:
            cls_acc = metrics[k].avg
            if k in domain_accs and domain_accs[k].count > 0:
                dom_acc = domain_accs[k].avg
                print(f"{k:<15} {cls_acc:>11.2f}% {dom_acc:>11.2f}%")
            else:
                print(f"{k:<15} {cls_acc:>11.2f}% {'N/A':>12}")
    print("=" * 70)

    if cm.sum() > 0:
        print("\nDomain confusion matrix (rows=true, cols=pred):")
        print(cm.cpu().numpy())

    # Update history records
    if acc_hist is not None:
        for k, v in metrics.items():
            acc_hist.setdefault(k, []).append(v.avg)

    if domain_hist is not None:
        for k, v in domain_accs.items():
            if v.count > 0:
                domain_hist.setdefault(k, []).append(v.avg)

    clean_acc = metrics['Clean'].avg if 'Clean' in metrics else 0.0
    apgd_linf_acc = metrics['Linf_APGD'].avg if 'Linf_APGD' in metrics else 0.0

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

    torch.save(state, os.path.join(path, "latest_model.pth"))


def load_checkpoint(model, optimizer, ckpt_path, base_main_lr=0.001, base_domain_lr=0.003):
    load_dev = device if torch.cuda.is_available() else torch.device("cpu")
    try:
        ckpt = torch.load(ckpt_path, map_location=load_dev, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=load_dev)
    # Allow slight architecture mismatches (e.g., domain count changes)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        print(f"[WARN] state_dict mismatch, missing={len(missing)}, unexpected={len(unexpected)}")
    try:
        optimizer.load_state_dict(ckpt["optimizer"])
    except Exception as e:
        print(f"[WARN] Failed to load optimizer state (different group structure?). Ignored:{e}")
    set_group_lrs(optimizer, base_main_lr, base_domain_lr)
    print(f"=> Resumed from epoch {ckpt['epoch']}, learning rates reset to main={base_main_lr} / domain={base_domain_lr}")
    return ckpt["epoch"] + 1, ckpt.get("accuracy_history"), ckpt.get("domain_acc_history")


# ══════════════════════════════════════════════════════════════════════════
#  Main entry point
# ══════════════════════════════════════════════════════════════════════════
def _resume_eval_history_len(acc_hist, args):
    if not acc_hist:
        return 0

    preferred_attacks = ['Clean']
    if args is not None and hasattr(args, 'attacks'):
        preferred_attacks.extend([atk for atk in args.attacks if atk != 'Clean'])

    seen = set()
    for attack_name in preferred_attacks:
        if attack_name in seen:
            continue
        seen.add(attack_name)
        if attack_name in acc_hist:
            return len(acc_hist[attack_name])

    all_lengths = [len(vals) for vals in acc_hist.values()]
    if all_lengths:
        return max(all_lengths)
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
                                            './datasets/',
                                            num_workers=args.num_workers,
                                            pin_memory=not args.disable_pin_memory,
                                            persistent_workers=not args.disable_persistent_workers,
                                            prefetch_factor=args.prefetch_factor)

    # Prepare PCA bases for GPGD attack if needed
    prepare_subspace_bases(args, trainloader)

    print("==> Building model ...")
    # Use create_encoder factory to build model based on backbone and attack_config
    # Pass dataset parameter so model auto-selects correct normalization stats
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
        model_kwargs['size'] = 32  # CIFAR-10/100

    model = create_encoder(**model_kwargs).to(device)
    model.count_frequency_convolutions()

    # ── Group optimizer: higher lr for threat_domain_classifier ─────────────────────
    domain_params = list(model.threat_domain_classifier.parameters())
    domain_param_ids = set(id(p) for p in domain_params)
    main_params = [p for p in model.parameters() if id(p) not in domain_param_ids]

    optimizer = optim.Adam([
        {"params": main_params, "lr": args.lr, "weight_decay": args.weight_decay, "name": "main"},
        {"params": domain_params, "lr": args.lr_domain, "weight_decay": args.weight_decay, "name": "domain"},
    ])
    set_group_lrs(optimizer, args.lr, args.lr_domain)

    # Initialize history records based on selected attacks
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
    print(f"Initial learning rate main={args.lr} | domain={args.lr_domain} | Schedule: 0-49x1, 50-69x0.1, >=70x0.01")
    print(f"domain_loss_weight: {args.domain_loss_weight}")
    print(f"map_update_every: {args.map_update_every} iters")
    print(f"domains(clusters/domains)={args.domains}")
    print(f"Training attacks: {args.train_attacks}")
    print("attack_lr_scale: 1.0")
    print(f"Selected evaluation attacks: {args.attacks}")
    if getattr(args, 'subspace_bases', None) is not None:
        print(f"GPGD PCA bases: {args.subspace_basis_path}")
    print("=" * 60)

    last_completed_epoch = start_epoch - 1
    if args.resume and _should_run_resume_eval(last_completed_epoch, acc_hist, args):
        print(f"[Resume] Detected missing evaluation/curves for epoch {last_completed_epoch} , running initial validation.")
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
    parser = argparse.ArgumentParser("TaFD: Threat-aware Frequency Decoupling — Adversarial Training")
    parser.add_argument("--dataset", type=str, default="CIFAR100",
                        help="Dataset name (e.g., CIFAR10 / CIFAR100)")
    parser.add_argument("--lr", type=float, default=0.001, help="Main backbone learning rate")
    parser.add_argument("--lr_domain", type=float, default=0.001, help="Domain classifier learning rate")
    parser.add_argument("--weight_decay", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=16, help="Validation/test batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="DataLoader prefetch_factor")
    parser.add_argument("--disable_pin_memory", action="store_true", help="Disable DataLoader pin_memory")
    parser.add_argument("--disable_persistent_workers", action="store_true", help="Disable DataLoader persistent_workers")
    parser.add_argument("--log_every", type=int, default=10, help="Tqdm update interval (iterations)")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--end_epoch", type=int, default=76)
    parser.add_argument("--eval_freq", type=int, default=15)
    parser.add_argument("--resume", type=str,
                        default="")
    parser.add_argument("--result_dir", type=str,
                        default="")
    # Keep n_cls parameter, but it will be overwritten by auto-inference
    parser.add_argument("--n_cls", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)

    # Number of domains/clusters (controls domain classifier, clusters, experts)
    parser.add_argument("--domains", type=int, default=6, help="Number of domains/clusters")

    # Domain supervision weight
    parser.add_argument("--domain_loss_weight", type=float, default=1.0,
                        help="Weight of domain supervision in total loss")
    # Dynamic clustering frequency (iterations)
    parser.add_argument("--map_update_every", type=int, default=50)

    # ══════════════════════════════════════════════════════════════════════════
    # Attack selection parameters
    # ══════════════════════════════════════════════════════════════════════════
    parser.add_argument("--attacks", type=str, nargs='+',
                        default=None,
                        choices=['Clean',
                                 'Linf_PGD', 'L2_PGD',          # training PGD-10
                                 'Linf_APGD', 'L2_APGD',        # evaluation APGD-100
                                 'ACE', 'ReColorAdv', 'HSVAdv', 'ALA', 'RetouchUAA', 'GPGD', 'StAdv'],
                        help="List of attacks to evaluate. Linf_APGD/L2_APGD use APGD-100 (eval); Linf_PGD/L2_PGD use PGD-10 (train).")

    parser.add_argument("--subspace_basis_path", type=str, default="",
                        help="Path to GPGD attack PCA bases; auto-placed in result_dir if empty")
    parser.add_argument("--subspace_rank", type=int, default=128,
                        help="GPGD attack PCA rank")
    parser.add_argument("--subspace_max_per_class", type=int, default=600,
                        help="Max samples per class when building GPGD PCA bases")

    # ══════════════════════════════════════════════════════════════════════════
    # Attack configuration and model architecture selection
    # ══════════════════════════════════════════════════════════════════════════
    parser.add_argument("--attack_config", type=str, default='v10',
                        choices=['v10', 'v20'],
                        help="Attack config: v10 (7 attacks) or v20 (5 attacks)")
    parser.add_argument("--backbone", type=str, default='resnet',
                        choices=['resnet', 'mobilevit'],
                        help="Model backbone: resnet or mobilevit")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device index (e.g., 0, 1, 2 ...)")

    args = parser.parse_args()

    # Set GPU
    if torch.cuda.is_available() and args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    # Automatically set num_sources based on attack config
    attack_cfg = ATTACK_CONFIGS[args.attack_config]
    args.num_sources = attack_cfg['num_sources']
    args.train_attacks = list(attack_cfg['train_attacks'])
    args.test_attacks = list(attack_cfg['test_attacks'])
    args.domain_names = list(attack_cfg['domain_names'])
    args.subspace_bases = None

    if not args.attacks:
        args.attacks = list(args.test_attacks)

    # Automatically infer n_cls (CIFAR-100->100; CIFAR-10->10)
    inferred_n = _infer_num_classes(args.dataset, fallback=args.n_cls)
    if inferred_n != args.n_cls:
        print(f"[Info] n_cls auto-updated from {args.n_cls} to {inferred_n} (based on dataset='{args.dataset}')")
        args.n_cls = inferred_n

    # Automatically generate result_dir if not specified
    if not args.result_dir:
        args.result_dir = (
            f"./results/tafd_{args.backbone}_{args.dataset}_d{args.domains}"
            f"_{args.attack_config}_lr{args.lr}_dlr{args.lr_domain}"
            f"_dw{args.domain_loss_weight}_bs{args.batch_size}"
            f"_ep{args.end_epoch}_seed{args.seed}"
        )

    print(f"[Auto-Config] Dataset: {args.dataset}")
    print(f"[Auto-Config] n_cls: {args.n_cls}")
    print(f"[Auto-Config] Attack config: {args.attack_config} ({attack_cfg['num_sources']} attacks)")
    print(f"[Auto-Config] Training attacks: {args.train_attacks}")
    print(f"[Auto-Config] Model backbone: {args.backbone}")
    print(f"[Auto-Config] Selected evaluation attacks: {args.attacks}")
    print(f"[Auto-Config] result_dir: {args.result_dir}")

    os.makedirs(args.result_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    main(args)
