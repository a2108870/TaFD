#!/usr/bin/env python3
"""Core TaFD training and evaluation implementation."""

import os

import random
import argparse
import numpy as np
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


from utils.utils import *
from utils.datasets_utils import GetDataLoader
from torchattacks import APGD, PGD, PGDL2


from attacks.ace import ace_attack
from attacks.recoloradv import recoloradv_attack
from attacks.ala import ala_attack
from attacks.hsvadv import hsvadv_attack
from attacks.retouch_uaa import retouch_uaa_attack


from attacks.gpgd import gpgd_attack, build_gpgd_bases, save_gpgd_bases, load_gpgd_bases
from attacks.stadv import stadv_attack


from models.tafd import build_tafd_model
from training.protocols import (
    ATTACK_SHORT_NAMES,
    ATTACK_UNIONS,
    CANONICAL_ATTACK_TEST_ORDER,
    DIAGNOSIS_ATTACKS,
    DIAGNOSIS_SUPERVISED_ATTACKS,
    migrate_metric_history,
)

# -------------------------------------------------------------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# -------------------------------------------------------------------------




def _cuda_sync_if_needed():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _update_confusion_matrix(cm: torch.Tensor, target_threat_domain_indices: torch.Tensor, predicted_threat_domains: torch.Tensor):
    if target_threat_domain_indices.numel() == 0:
        return
    num_threat_domains = cm.size(0)
    true_cpu = target_threat_domain_indices.detach().view(-1).to(torch.long).cpu()
    pred_cpu = predicted_threat_domains.detach().view(-1).to(torch.long).cpu()
    flat = true_cpu * num_threat_domains + pred_cpu
    counts = torch.bincount(flat, minlength=num_threat_domains * num_threat_domains).view(num_threat_domains, num_threat_domains)
    cm += counts





def set_group_lrs(optimizer: optim.Optimizer, lr_main: float, diagnosis_lr: float):
    """Set learning rates for the main and diagnosis parameter groups."""
    for pg in optimizer.param_groups:
        if pg.get("name") == "diagnosis":
            pg["lr"] = diagnosis_lr
            pg["initial_lr"] = diagnosis_lr
        else:
            pg["lr"] = lr_main
            pg["initial_lr"] = lr_main


def adjust_learning_rate(optimizer, epoch, lr_main_init=0.001, diagnosis_lr_init=0.003):
    """Decay both parameter groups with the same schedule."""
    if epoch < 50:
        factor = 1.0
    elif epoch < 70:
        factor = 0.1
    else:
        factor = 0.01
    for pg in optimizer.param_groups:
        if pg.get("name") == "diagnosis":
            pg["lr"] = diagnosis_lr_init * factor
        else:
            pg["lr"] = lr_main_init * factor


def get_current_lrs(optimizer):
    """Return the current main and diagnosis learning rates."""
    lrs = {}
    for i, pg in enumerate(optimizer.param_groups):
        name = pg.get("name", f"group{i}")
        lrs[name] = pg["lr"]
    lr_main = lrs.get("main", list(lrs.values())[0])
    diagnosis_lr = lrs.get("diagnosis", lr_main)
    return lr_main, diagnosis_lr


def _normalize_optimizer_group_names(optimizer: optim.Optimizer):
    """Migrate historical optimizer metadata to paper-aligned group names."""
    for group in optimizer.param_groups:
        if group.get("name") == "domain":
            group["name"] = "diagnosis"


class AverageMeter:
    def __init__(self): self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val

        if isinstance(val, torch.Tensor):
            val = val.detach()

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



def compute_balanced_ce_weight(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float().to(labels.device)
    counts = torch.clamp(counts, min=1.0)
    inv = 1.0 / counts
    weights = inv / inv.mean()
    return weights





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





def generate_single_attack(attack_func, img, tgt, model, device,
                           attack_name="Unknown", **kwargs):
    try:
        if attack_name in ('APGD_Linf', 'APGD_L2'):
            return attack_func(img, tgt)
        else:
            return attack_func(img, tgt, model, device, **kwargs)
    except Exception as e:
        print(f"Attack {attack_name} failed: {e}")
        return img.clone()


def _source_id_from_attack(attack_name: str, attack_names: list):
    if attack_name in attack_names:
        return attack_names.index(attack_name)
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
                          lr_scale=1.0, num_classes=100, gpgd_bases=None):
    """Generate adversarial examples keyed by attack name for either attack union."""
    attack_inputs = {}

    for name in attack_names:
        if name == 'Clean':
            attack_inputs['Clean'] = img
        elif name == 'APGD_Linf':
            if atk_apgd_linf is None:
                raise ValueError("APGD_Linf attack is not initialized")
            attack_inputs['APGD_Linf'] = generate_single_attack(atk_apgd_linf, img, tgt, model, device, "APGD_Linf")
        elif name == 'APGD_L2':
            if atk_apgd_l2 is None:
                raise ValueError("APGD_L2 attack is not initialized")
            attack_inputs['APGD_L2'] = generate_single_attack(atk_apgd_l2, img, tgt, model, device, "APGD_L2")
        elif name == 'ACE':
            attack_inputs['ACE'] = generate_single_attack(
                ace_attack, img, tgt, model, device, "ACE",
                lr=1 * lr_scale, max_iterations=10, num_classes=num_classes
            )
        elif name == 'HSVAdv':
            attack_inputs['HSVAdv'] = generate_single_attack(
                hsvadv_attack, img, tgt, model, device, "HSVAdv",
                lr=1 * lr_scale, max_iterations=10, num_classes=num_classes
            )
        elif name == 'ReColorAdv':
            attack_inputs['ReColorAdv'] = generate_single_attack(
                recoloradv_attack, img, tgt, model, device, "ReColorAdv",
                lr=0.01 * lr_scale, max_iterations=10, num_classes=num_classes
            )
        elif name == 'ALA':
            attack_inputs['ALA'] = generate_single_attack(
                ala_attack, img, tgt, model, device, "ALA",
                lr=1 * lr_scale, max_iterations=10, num_classes=num_classes
            )
        elif name == 'RetouchUAA':
            attack_inputs['RetouchUAA'] = generate_single_attack(
                retouch_uaa_attack, img, tgt, model, device, "RetouchUAA",
                lr=0.1 * lr_scale, max_iterations=10, num_classes=num_classes
            )
        elif name == 'GPGD':
            if gpgd_bases is None:
                raise ValueError("GPGD requires precomputed PCA bases")
            attack_inputs['GPGD'] = generate_single_attack(
                gpgd_attack, img, tgt, model, device, "GPGD",
                bases_dict=gpgd_bases, steps=10, epsilon=2.0, num_classes=num_classes, proj='l2'
            )
        elif name == 'StAdv':
            attack_inputs['StAdv'] = generate_single_attack(
                stadv_attack, img, tgt, model, device, "StAdv",
                eps=0.045, steps=10, mode='linf'
            )
        else:
            print(f"[WARN] Unsupported attack {name}; skipping")

    return attack_inputs





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
    """Infer the number of classes from a dataset or result identifier."""
    if dataset_name is None:
        return fallback
    name = str(dataset_name).lower()

    if '200class' in name:
        return 200
    if 'imagenette' in name:
        return 10
    if ('cifar100' in name) or ('cifar-100' in name):
        return 100

    if '10class' in name:
        return 10
    if ('cifar10' in name) or ('cifar-10' in name):
        return 10


    if 'tiny' in name:
        return 200

    return fallback


def mapping_status_str(status_dict: dict, order=None) -> str:
    if order is None:
        order = ['APGD_Linf', 'APGD_L2', 'ACE', 'HSVAdv', 'ReColorAdv', 'ALA', 'RetouchUAA']
    parts = []
    for k in order:
        short_k = ATTACK_SHORT_NAMES.get(k, k)
        v = status_dict.get(k, 'Unassigned')
        if isinstance(v, str) and v.startswith('ThreatDomain-'):
            parts.append(f"{short_k}:{v.split('-')[-1]}")
        elif isinstance(v, int):
            parts.append(f"{short_k}:{v}")
        else:
            parts.append(f"{short_k}:{v}")
    return " ".join(parts)


def prepare_gpgd_bases(args, trainloader):
    """Build or load the PCA bases required by GPGD."""
    needs_gpgd = ('GPGD' in getattr(args, 'train_attacks', [])) or ('GPGD' in getattr(args, 'test_attacks', []))
    if not needs_gpgd:
        args.gpgd_bases = None
        return

    if args.gpgd_basis_path:
        basis_path = args.gpgd_basis_path
    else:
        basis_path = os.path.join(
            args.result_dir,
            f"gpgd_pca_bases_{args.dataset}_c{args.num_classes}_r{args.gpgd_rank}.pth"
        )
    args.gpgd_basis_path = basis_path

    if os.path.isfile(basis_path):
        print(f"[GPGD] Loaded PCA bases: {basis_path}")
        args.gpgd_bases = load_gpgd_bases(basis_path, device=device)
        return

    print("[GPGD] PCA bases not found; building them now")
    args.gpgd_bases = build_gpgd_bases(
        trainloader,
        n_classes=args.num_classes,
        rank=args.gpgd_rank,
        max_per_class=args.gpgd_max_per_class,
        device=torch.device('cpu')
    )
    os.makedirs(os.path.dirname(basis_path) or '.', exist_ok=True)
    save_gpgd_bases(basis_path, args.gpgd_bases)
    print(f"[GPGD] Saved PCA bases: {basis_path}")


def _should_batch_eval_forwards(dataset_name: str) -> bool:
    return _infer_input_size(dataset_name) <= 64





def train_one_epoch(epoch, trainloader, criterion, optimizer, n_classes, model,
          initial_lr_main=0.001, initial_diagnosis_lr=0.003, args=None, global_iter_start=0):
    adjust_learning_rate(optimizer, epoch, lr_main_init=initial_lr_main, diagnosis_lr_init=initial_diagnosis_lr)
    lr_main, diagnosis_lr = get_current_lrs(optimizer)
    print(
        f"\nEpoch {epoch:03d} | lr_main={lr_main:.6f} | diagnosis_lr={diagnosis_lr:.6f} | threat_domains={args.num_threat_domains}")

    losses = {k: AverageMeter() for k in ['cls', 'diagnosis', 'total']}
    training_attacks = list(args.train_attacks)


    train_cls_accs = {d: AverageMeter() for d in training_attacks}
    train_diagnosis_accuracies = {d: AverageMeter() for d in training_attacks}

    diagnosis_accuracy_meter = AverageMeter()




    atk_apgd_linf = _build_train_pgd_attack(model, norm='Linf', eps=8 / 255, steps=10)
    atk_apgd_l2 = _build_train_pgd_attack(model, norm='L2', eps=0.5, steps=10)

    model.train()
    pbar = tqdm(trainloader, leave=True, desc=f"Train {epoch}")
    global_iter = global_iter_start


    lr_scale = 0.1 if _is_tiny_dataset(getattr(args, "dataset", None)) else 1.0

    for step_idx, (img, tgt) in enumerate(pbar, start=1):
        img = img.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        bsz = img.size(0)
        attack_source_ids = torch.arange(len(training_attacks), device=device, dtype=torch.long).repeat_interleave(bsz)


        model.eval()




        if hasattr(model, "set_bpda"):
            model.set_bpda(True)
        # ===================================

        attack_inputs = generate_attack_batch(
            img, tgt, model, device,
            attack_names=training_attacks,
            atk_apgd_linf=atk_apgd_linf,
            atk_apgd_l2=atk_apgd_l2,
            lr_scale=lr_scale,
            num_classes=n_classes,
            gpgd_bases=getattr(args, 'gpgd_bases', None)
        )



        if hasattr(model, "set_bpda"):
            model.set_bpda(False)

        imgs_mix = [attack_inputs[d] for d in training_attacks]

        global_iter += 1
        if epoch < 10:
            if global_iter % args.assignment_update_interval == 0 or global_iter == 1:
                with torch.no_grad():
                    model.update_spectral_prototypes(torch.cat(imgs_mix, dim=0), attack_source_ids)

        if global_iter % args.assignment_update_interval == 0 or global_iter == 1:
            model.update_threat_domain_assignments(epoch, args.end_epoch)

        target_threat_domain_indices = model.get_threat_domain_indices(attack_source_ids)

        model.train()
        combined = torch.cat(imgs_mix, 0).detach()

        outputs = model(combined, attack_source_ids=attack_source_ids)
        cls_logits, _, diagnosis_logits, _ = outputs

        # Supervise the diagnosis module with the current source-to-domain assignment.
        diagnosis_loss = F.cross_entropy(diagnosis_logits, target_threat_domain_indices)
        losses['diagnosis'].update(diagnosis_loss, combined.size(0))

        with torch.no_grad():
            _, predicted_threat_domains = diagnosis_logits.max(1)
            diagnosis_accuracy = (predicted_threat_domains == target_threat_domain_indices).float().mean() * 100
            diagnosis_accuracy_meter.update(diagnosis_accuracy, combined.size(0))

        repeated_tgt = tgt.repeat(len(training_attacks))
        for i, dname in enumerate(training_attacks):
            attack_logits = cls_logits[bsz * i:bsz * (i + 1)]
            diagnosis_logits_slice = diagnosis_logits[bsz * i:bsz * (i + 1)]
            target_threat_domain_slice = target_threat_domain_indices[bsz * i:bsz * (i + 1)]

            with torch.no_grad():
                acc_cls = accuracy(attack_logits, tgt, (1,))[0]
                train_cls_accs[dname].update(acc_cls, bsz)
                diagnosis_accuracy = (diagnosis_logits_slice.max(1)[1] == target_threat_domain_slice).float().mean() * 100
                train_diagnosis_accuracies[dname].update(diagnosis_accuracy, bsz)

        cls_loss = F.cross_entropy(cls_logits, repeated_tgt)
        losses['cls'].update(cls_loss, combined.size(0))

        total_loss = cls_loss + args.diagnosis_loss_weight * diagnosis_loss
        losses['total'].update(total_loss, combined.size(0))

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()


        if step_idx % args.log_every == 0:
            status = model.get_threat_domain_assignment_status()
            pbar.set_postfix({
                "loss": f"{losses['cls'].avg:.3f}",
                "map": mapping_status_str(status, order=args.attack_names),
                **{ATTACK_SHORT_NAMES.get(d, d): (f"{train_cls_accs[d].avg:.1f}/{train_diagnosis_accuracies[d].avg:.1f}" if d in DIAGNOSIS_SUPERVISED_ATTACKS and train_diagnosis_accuracies[d].count > 0 else f"{train_cls_accs[d].avg:.1f}/-") for d in training_attacks},
            })

    pbar.close()
    return global_iter






def validate_attack_union(epoch, testloader, criterion, model, num_classes,
                   classification_accuracy_history=None, diagnosis_accuracy_history=None, args=None):
    """Evaluate the configured clean and adversarial conditions."""

    selected_attacks = (
        args.test_attacks
        if args and hasattr(args, 'test_attacks')
        else CANONICAL_ATTACK_TEST_ORDER
    )
    if 'GPGD' in selected_attacks and getattr(args, 'gpgd_bases', None) is None:
        raise RuntimeError("GPGD evaluation was requested before PCA bases were prepared")

    print(f"\n[Validation] Selected attacks: {selected_attacks}")



    selected_attacks_runtime = list(selected_attacks)

    atk_apgd_linf = None
    atk_apgd_l2 = None
    if 'APGD_Linf' in selected_attacks_runtime:
        atk_apgd_linf = _build_apgd_attack(model, norm='Linf', eps=8 / 255, steps=100)
    if 'APGD_L2' in selected_attacks_runtime:
        atk_apgd_l2 = _build_apgd_attack(model, norm='L2', eps=0.5, steps=100)
    print(f"\n{'='*70}")
    print(f"{'='*70}")


    metrics = {k: AverageMeter() for k in selected_attacks_runtime}
    diagnosis_accuracies = {k: AverageMeter() for k in selected_attacks_runtime if k in DIAGNOSIS_ATTACKS}

    cm = torch.zeros(model.num_threat_domains, model.num_threat_domains, dtype=torch.long)
    model.eval()

    pbar = tqdm(testloader, desc=f"Val Ep{epoch}", leave=True)

    lr_scale = 0.1 if _is_tiny_dataset(getattr(args, "dataset", None)) else 1.0
    batch_eval_forwards = _should_batch_eval_forwards(getattr(args, "dataset", None))

    for step_idx, (img, tgt) in enumerate(pbar, start=1):
        img = img.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        bsz = img.size(0)


        if hasattr(model, "set_bpda"):
            model.set_bpda(True)


        attack_inputs = generate_attack_batch(
            img, tgt, model, device,
            attack_names=selected_attacks_runtime,
            atk_apgd_linf=atk_apgd_linf,
            atk_apgd_l2=atk_apgd_l2,
            lr_scale=lr_scale,
            num_classes=num_classes,
            gpgd_bases=getattr(args, 'gpgd_bases', None)
        )

        attack_source_ids_by_name = {}
        for name in attack_inputs.keys():
            if name not in DIAGNOSIS_ATTACKS:
                continue
            attack_source_id = _source_id_from_attack(name, args.attack_names)
            if attack_source_id is None:
                continue
            attack_source_ids_by_name[name] = torch.full((bsz,), attack_source_id, device=device, dtype=torch.long)


        if hasattr(model, "set_bpda"):
            model.set_bpda(False)


        with torch.no_grad():
            if batch_eval_forwards and attack_inputs:
                ordered_names = list(attack_inputs.keys())
                combined_inputs = torch.cat([attack_inputs[name] for name in ordered_names], dim=0)
                out_tuple = model(combined_inputs, attack_source_ids=None)
                logits_all = out_tuple[0] if isinstance(out_tuple, tuple) else out_tuple
                diagnosis_logits_all = out_tuple[2] if isinstance(out_tuple, tuple) and len(out_tuple) > 2 else None

                for idx, name in enumerate(ordered_names):
                    start = idx * bsz
                    end = start + bsz
                    logits = logits_all[start:end]
                    acc1 = accuracy(logits, tgt, (1,))[0]
                    metrics[name].update(acc1.item(), bsz)

                    if name in attack_source_ids_by_name and diagnosis_logits_all is not None:
                        diagnosis_logits_slice = diagnosis_logits_all[start:end]
                        predicted_threat_domains = diagnosis_logits_slice.max(1)[1]
                        target_threat_domain_indices = model.get_threat_domain_indices(attack_source_ids_by_name[name])
                        diagnosis_accuracy = (predicted_threat_domains == target_threat_domain_indices).float().mean() * 100
                        diagnosis_accuracies[name].update(diagnosis_accuracy.item(), bsz)

                        _update_confusion_matrix(cm, target_threat_domain_indices, predicted_threat_domains)
            else:
                for name, x_in in attack_inputs.items():
                    out_tuple = model(x_in, attack_source_ids=None)
                    logits = out_tuple[0] if isinstance(out_tuple, tuple) else out_tuple
                    acc1 = accuracy(logits, tgt, (1,))[0]
                    metrics[name].update(acc1.item(), bsz)


                    if name in attack_source_ids_by_name and isinstance(out_tuple, tuple) and len(out_tuple) > 2:
                        diagnosis_logits_slice = out_tuple[2]
                        predicted_threat_domains = diagnosis_logits_slice.max(1)[1]
                        target_threat_domain_indices = model.get_threat_domain_indices(attack_source_ids_by_name[name])
                        diagnosis_accuracy = (predicted_threat_domains == target_threat_domain_indices).float().mean() * 100
                        diagnosis_accuracies[name].update(diagnosis_accuracy.item(), bsz)

                        _update_confusion_matrix(cm, target_threat_domain_indices, predicted_threat_domains)


        if step_idx % args.log_every == 0:
            postfix_dict = {}
            for k in selected_attacks_runtime:
                if k in metrics:
                    cls_acc_str = f"{metrics[k].avg:.1f}"
                    if k in diagnosis_accuracies and diagnosis_accuracies[k].count > 0:
                        diagnosis_accuracy_text = f"{diagnosis_accuracies[k].avg:.1f}"
                        postfix_dict[k] = f"{cls_acc_str}/{diagnosis_accuracy_text}"
                    else:
                        postfix_dict[k] = cls_acc_str
            pbar.set_postfix(postfix_dict)

    pbar.close()


    print(f"\n{'='*70}")
    print(f"Validation summary (classification / diagnosis accuracy)")
    print("=" * 70)
    print(f"{'Attack':<15} {'Cls Acc':>12} {'Diagnosis Acc':>12} {'Format':>15}")
    print("-" * 70)
    for k in selected_attacks_runtime:
        if k in metrics:
            cls_acc = metrics[k].avg
            if k in diagnosis_accuracies and diagnosis_accuracies[k].count > 0:
                diagnosis_accuracy = diagnosis_accuracies[k].avg
                fmt_str = f"{cls_acc:.1f}/{diagnosis_accuracy:.1f}"
                print(f"{k:<15} {cls_acc:>11.2f}% {diagnosis_accuracy:>11.2f}% {fmt_str:>15}")
            else:
                print(f"{k:<15} {cls_acc:>11.2f}% {'N/A':>12} {cls_acc:>15.1f}")
    print("=" * 70)

    if cm.sum() > 0:
        print(f"\nThreat-domain confusion matrix (rows=true, columns=predicted):")
        print(cm.cpu().numpy())


    print(f"\n{'#'*80}")
    print(f"  Validation summary (Epoch {epoch})")
    print(f"{'#'*80}")

    print(f"{'Attack':<12} | {'Cls%':>9}")
    print('-' * 24)
    for k in selected_attacks_runtime:
        if k in metrics:
            print(f"{k:<12} | {metrics[k].avg:>9.2f}%")

    print(f"{'#'*80}\n")


    if classification_accuracy_history is not None:
        for k, v in metrics.items():
            classification_accuracy_history.setdefault(k, []).append(v.avg)

    if diagnosis_accuracy_history is not None:
        for k, v in diagnosis_accuracies.items():
            if v.count > 0:
                diagnosis_accuracy_history.setdefault(k, []).append(v.avg)

    clean_acc = metrics['Clean'].avg if 'Clean' in metrics else 0.0
    apgd_linf_acc = metrics['APGD_Linf'].avg if 'APGD_Linf' in metrics else 0.0

    return clean_acc, apgd_linf_acc, classification_accuracy_history, diagnosis_accuracy_history



#  Checkpoint I/O

def save_checkpoint(model, optimizer, epoch, path, classification_accuracy_history=None, diagnosis_accuracy_history=None):
    os.makedirs(path, exist_ok=True)
    state = {"model": model.state_dict(),
             "optimizer": optimizer.state_dict(),
             "epoch": epoch,
             "classification_accuracy_history": classification_accuracy_history,
             "diagnosis_accuracy_history": diagnosis_accuracy_history}
    # torch.save(state, os.path.join(path, f"checkpoint_epoch_{epoch}.pth"))
    torch.save(state, os.path.join(path, "latest_model.pth"))


def load_checkpoint(model, optimizer, ckpt_path, base_main_lr=0.001, base_diagnosis_lr=0.003):
    load_dev = device if torch.cuda.is_available() else torch.device("cpu")
    try:
        ckpt = torch.load(ckpt_path, map_location=load_dev, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=load_dev)

    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        print(f"[WARN] State-dict mismatch: missing={len(missing)}, unexpected={len(unexpected)}")
    try:
        optimizer.load_state_dict(ckpt["optimizer"])
        _normalize_optimizer_group_names(optimizer)
    except Exception as e:
        print(f"[WARN] Optimizer state could not be restored and was ignored: {e}")
    set_group_lrs(optimizer, base_main_lr, base_diagnosis_lr)
    print(f"=> Resumed from epoch {ckpt['epoch']}; learning rates reset to main={base_main_lr} / diagnosis={base_diagnosis_lr}")
    classification_accuracy_history = migrate_metric_history(
        ckpt.get("classification_accuracy_history", ckpt.get("accuracy_history"))
    )
    diagnosis_accuracy_history = migrate_metric_history(
        ckpt.get("diagnosis_accuracy_history", ckpt.get("domain_acc_history"))
    )
    return ckpt["epoch"] + 1, classification_accuracy_history, diagnosis_accuracy_history





def _resume_eval_history_len(classification_accuracy_history, args):
    if not classification_accuracy_history:
        return 0

    preferred_attacks = ['Clean']
    if args is not None and hasattr(args, 'test_attacks'):
        preferred_attacks.extend([atk for atk in args.test_attacks if atk != 'Clean'])

    seen = set()
    for attack_name in preferred_attacks:
        if attack_name in seen:
            continue
        seen.add(attack_name)
        if attack_name in classification_accuracy_history:
            return len(classification_accuracy_history[attack_name])

    evaluation_lengths = [len(values) for values in classification_accuracy_history.values()]
    if evaluation_lengths:
        return max(evaluation_lengths)
    return 0


def _should_run_resume_eval(last_completed_epoch, classification_accuracy_history, args):
    if args is None or last_completed_epoch <= 0:
        return False
    if args.eval_freq <= 0 or last_completed_epoch % args.eval_freq != 0:
        return False

    expected_eval_points = last_completed_epoch // args.eval_freq
    recorded_eval_points = _resume_eval_history_len(classification_accuracy_history, args)
    if recorded_eval_points < expected_eval_points:
        return True

    curve_path = os.path.join(
        args.result_dir,
        'accuracy_curves',
        f'acc_curve_epoch_{last_completed_epoch}.png'
    )
    return not os.path.isfile(curve_path)


def main(args):
    print("==> Preparing data")
    trainloader, testloader = GetDataLoader(args.dataset,
                                            args.batch_size,
                                            args.test_batch_size,
                                            args.dataset_path,
                                            num_workers=args.num_workers,
                                            pin_memory=not args.disable_pin_memory,
                                            persistent_workers=not args.disable_persistent_workers,
                                            prefetch_factor=args.prefetch_factor)


    prepare_gpgd_bases(args, trainloader)

    print("==> Building model")


    model_kwargs = dict(
        backbone=args.backbone,
        dataset=args.dataset,
        num_attack_sources=args.num_attack_sources,
        num_threat_domains=args.num_threat_domains,
        num_frequency_experts=args.num_threat_domains,
        num_classes=args.num_classes,
        attack_names=args.attack_names
    )
    if args.backbone == 'mobilevit':
        model_kwargs['size'] = _infer_input_size(args.dataset)

    model = build_tafd_model(**model_kwargs).to(device)
    model.count_frequency_convolutions()


    diagnosis_params = list(model.threat_domain_classifier.parameters())
    diagnosis_param_ids = set(id(p) for p in diagnosis_params)
    main_params = [p for p in model.parameters() if id(p) not in diagnosis_param_ids]

    optimizer = optim.Adam([
        {"params": main_params, "lr": args.lr, "weight_decay": args.weight_decay, "name": "main"},
        {"params": diagnosis_params, "lr": args.diagnosis_lr, "weight_decay": args.weight_decay, "name": "diagnosis"},
    ])
    set_group_lrs(optimizer, args.lr, args.diagnosis_lr)


    classification_accuracy_history = {k: [] for k in args.test_attacks}
    diagnosis_accuracy_history = {k: [] for k in args.test_attacks if k in DIAGNOSIS_ATTACKS}

    start_epoch = args.start_epoch
    if args.resume and os.path.isfile(args.resume):
        start_epoch, loaded_classification_accuracy_history, loaded_diagnosis_accuracy_history = load_checkpoint(
            model, optimizer, args.resume,
            base_main_lr=args.lr,
            base_diagnosis_lr=args.diagnosis_lr
        )
        if loaded_classification_accuracy_history:  classification_accuracy_history = loaded_classification_accuracy_history
        if loaded_diagnosis_accuracy_history: diagnosis_accuracy_history = loaded_diagnosis_accuracy_history

    criterion = nn.CrossEntropyLoss()

    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"num_classes (auto): {args.num_classes}")
    print(f"Initial learning rates: main={args.lr} | diagnosis={args.diagnosis_lr} | schedule: 0-49 x1, 50-74 x0.1, >=75 x0.01")
    print(f"diagnosis_loss_weight: {args.diagnosis_loss_weight}")
    print(f"assignment_update_interval: {args.assignment_update_interval} iters")
    print(f"num_threat_domains: {args.num_threat_domains}")
    print(f"Training attacks: {args.train_attacks}")
    print(
        f"attack_lr_scale: {'0.1 (small-scale datasets)' if _is_tiny_dataset(args.dataset) else '1.0'} | applied to ACE, HSVAdv, ReColorAdv, ALA, and RetouchUAA")
    print(f"Evaluation attacks: {args.test_attacks}")
    if getattr(args, 'gpgd_bases', None) is not None:
        print(f"GPGD PCA bases: {args.gpgd_basis_path}")
    print("=" * 60)
    print("=" * 60)

    last_completed_epoch = start_epoch - 1
    if args.resume and _should_run_resume_eval(last_completed_epoch, classification_accuracy_history, args):
        print(f"[Resume] Evaluation history is missing for epoch {last_completed_epoch}; running validation first")
        validate_attack_union(last_completed_epoch, testloader, criterion,
                       model, args.num_classes,
                       classification_accuracy_history, diagnosis_accuracy_history, args=args)
        plot_accuracy_curves(classification_accuracy_history, last_completed_epoch, args.result_dir, "acc")
        plot_accuracy_curves(diagnosis_accuracy_history, last_completed_epoch, args.result_dir, "diagnosis_accuracy")
        save_checkpoint(model, optimizer, last_completed_epoch,
                        args.result_dir, classification_accuracy_history, diagnosis_accuracy_history)

    global_iter = 0
    for epoch in range(start_epoch, args.end_epoch):
        global_iter = train_one_epoch(epoch, trainloader, criterion, optimizer,
                            args.num_classes, model,
                            initial_lr_main=args.lr, initial_diagnosis_lr=args.diagnosis_lr,
                            args=args, global_iter_start=global_iter)

        save_checkpoint(model, optimizer, epoch,
                        args.result_dir, classification_accuracy_history, diagnosis_accuracy_history)

        if epoch % args.eval_freq == 0 and epoch != 0:
            validate_attack_union(epoch, testloader, criterion,
                           model, args.num_classes,
                           classification_accuracy_history, diagnosis_accuracy_history, args=args)

            plot_accuracy_curves(classification_accuracy_history, epoch, args.result_dir, "acc")
            plot_accuracy_curves(diagnosis_accuracy_history, epoch, args.result_dir, "diagnosis_accuracy")
            save_checkpoint(model, optimizer, epoch,
                            args.result_dir, classification_accuracy_history, diagnosis_accuracy_history)
