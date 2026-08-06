#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from attacks.ace_adaptive_diagnosis import ace_adaptive_diagnosis_attack
from attacks.hsvadv_adaptive_diagnosis import hsvadv_attack as hsvadv_attack_adaptive_diagnosis
from attacks.ala_adaptive_diagnosis import ala_attack as ala_attack_adaptive_diagnosis
from attacks.recoloradv_adaptive_diagnosis import recoloradv_adaptive_diagnosis_attack
from attacks.stadv_adaptive_diagnosis import stadv_attack as stadv_attack_adaptive_diagnosis
from attacks.gpgd import build_gpgd_bases, load_gpgd_bases, save_gpgd_bases
from attacks.gpgd_adaptive_diagnosis import gpgd_attack as gpgd_attack_adaptive_diagnosis
from attacks.retouch_uaa_adaptive_diagnosis import (
    retouch_uaa_attack as retouch_uaa_attack_adaptive_diagnosis,
)
from training.tafd import (
    AverageMeter,
    _build_apgd_attack,
    _infer_input_size,
    _infer_num_classes,
    _is_tiny_dataset,
    accuracy,
    generate_attack_batch,
)
from training.protocols import ATTACK_UNIONS
from models.tafd import build_tafd_model
from torchattacks.attacks.apgd_adaptive_diagnosis import AdaptiveDiagnosisAPGD
from utils.datasets_utils import GetDataLoader


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_scales(scales_text):
    values = []
    for item in scales_text.split(','):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError('`--scales` must not be empty')
    return values


def attack_source_id_from_name(attack_name, attack_names):
    if attack_name in attack_names:
        return attack_names.index(attack_name)
    return None


def build_attack_source_ids(attack_names, batch_size, device, configured_attack_names):
    attack_source_ids_by_name = {}
    for name in attack_names:
        attack_source_id = attack_source_id_from_name(name, configured_attack_names)
        if attack_source_id is not None:
            attack_source_ids_by_name[name] = torch.full(
                (batch_size,), attack_source_id, device=device, dtype=torch.long
            )
    return attack_source_ids_by_name


def infer_domains_from_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model_state = ckpt.get('model', {})
    classifier_key = 'threat_domain_classifier.classifier.5.weight'
    if classifier_key in model_state and hasattr(model_state[classifier_key], 'shape'):
        return int(model_state[classifier_key].shape[0])
    td_state = model_state.get('threat_domain_diagnosis_state', {})
    mapping = td_state.get('source_to_domain', {})
    if isinstance(mapping, dict) and mapping:
        return max(int(v) for v in mapping.values()) + 1
    return None


def resolve_gpgd_basis_path(args):
    if args.gpgd_basis_path:
        if os.path.isabs(args.gpgd_basis_path):
            return args.gpgd_basis_path
        return os.path.join(BASE_DIR, args.gpgd_basis_path)

    resume_dir = os.path.dirname(args.resume)
    filename = f'gpgd_pca_bases_{args.dataset}_c{args.num_classes}_r{args.gpgd_rank}.pth'
    return os.path.join(resume_dir, filename)


def prepare_evaluation_gpgd_bases(args, trainloader, device):
    if 'GPGD' not in args.test_attacks:
        args.gpgd_bases = None
        return

    basis_path = resolve_gpgd_basis_path(args)
    args.gpgd_basis_path = basis_path

    if os.path.isfile(basis_path):
        print(f'[GPGD] loading PCA bases: {basis_path}')
        args.gpgd_bases = load_gpgd_bases(basis_path, device=device)
        return

    if trainloader is None:
        raise RuntimeError('GPGD evaluation requires trainloader to build PCA bases, but trainloader is unavailable')

    print(f'[GPGD] PCA bases not found, building: {basis_path}')
    args.gpgd_bases = build_gpgd_bases(
        trainloader,
        n_classes=args.num_classes,
        rank=args.gpgd_rank,
        max_per_class=args.gpgd_max_per_class,
        device=torch.device('cpu'),
    )
    os.makedirs(os.path.dirname(basis_path) or '.', exist_ok=True)
    save_gpgd_bases(basis_path, args.gpgd_bases)
    print(f'[GPGD] PCA bases saved: {basis_path}')


def evaluate_attack_inputs(model, attack_inputs, tgt, attack_source_ids_by_name, cls_metrics, diagnosis_metrics):
    if not attack_inputs:
        return

    batch_size = tgt.size(0)
    ordered_names = list(attack_inputs.keys())
    combined_inputs = torch.cat([attack_inputs[name] for name in ordered_names], dim=0)

    with torch.no_grad():
        out_tuple = model(combined_inputs, attack_source_ids=None)
        logits_all = out_tuple[0] if isinstance(out_tuple, tuple) else out_tuple
        diagnosis_logits_all = out_tuple[2] if isinstance(out_tuple, tuple) and len(out_tuple) > 2 else None

        for idx, name in enumerate(ordered_names):
            start = idx * batch_size
            end = start + batch_size
            logits = logits_all[start:end]
            acc1 = accuracy(logits, tgt, (1,))[0]
            cls_metrics[name].update(acc1.item(), batch_size)

            if name in attack_source_ids_by_name and diagnosis_logits_all is not None:
                diagnosis_logits_slice = diagnosis_logits_all[start:end]
                predicted_threat_domains = diagnosis_logits_slice.max(1)[1]
                target_threat_domain_indices = model.get_threat_domain_indices(attack_source_ids_by_name[name])
                diagnosis_accuracy = (
                    predicted_threat_domains == target_threat_domain_indices
                ).float().mean() * 100
                diagnosis_metrics[name].update(diagnosis_accuracy.item(), batch_size)


def validation_baseline(testloader, model, device, num_classes, args):
    model.eval()
    cls_metrics = {k: AverageMeter() for k in args.test_attacks}
    diagnosis_metrics = {k: AverageMeter() for k in args.test_attacks if k in args.attack_names}

    atk_apgd_linf = _build_apgd_attack(model, norm='Linf', eps=8 / 255, steps=args.apgd_steps) if 'APGD_Linf' in args.test_attacks else None
    atk_apgd_l2 = _build_apgd_attack(model, norm='L2', eps=0.5, steps=args.apgd_steps) if 'APGD_L2' in args.test_attacks else None

    pbar = tqdm(testloader, desc='Baseline', leave=True)
    for step_idx, (img, tgt) in enumerate(pbar, start=1):
        img = img.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        if hasattr(model, 'set_bpda'):
            model.set_bpda(True)

        lr_scale = 0.1 if _is_tiny_dataset(args.dataset) else 1.0
        attack_inputs = generate_attack_batch(
            img,
            tgt,
            model,
            device,
            attack_names=args.test_attacks,
            atk_apgd_linf=atk_apgd_linf,
            atk_apgd_l2=atk_apgd_l2,
            lr_scale=lr_scale,
            num_classes=num_classes,
            gpgd_bases=getattr(args, 'gpgd_bases', None),
        )
        attack_source_ids_by_name = build_attack_source_ids(
            attack_inputs.keys(), img.size(0), device, args.attack_names
        )

        if hasattr(model, 'set_bpda'):
            model.set_bpda(False)

        evaluate_attack_inputs(model, attack_inputs, tgt, attack_source_ids_by_name, cls_metrics, diagnosis_metrics)

        postfix = {}
        for name in args.test_attacks:
            if cls_metrics[name].count == 0:
                continue
            cls_val = f"{cls_metrics[name].avg:.1f}"
            if name in diagnosis_metrics and diagnosis_metrics[name].count > 0:
                postfix[name] = f"{cls_val}/{diagnosis_metrics[name].avg:.1f}"
            else:
                postfix[name] = cls_val
        pbar.set_postfix(postfix)

        if args.max_batches > 0 and step_idx >= args.max_batches:
            break

    pbar.close()
    return cls_metrics, diagnosis_metrics


def generate_adaptive_diagnosis_batch(img, tgt, model, device, attack_names, attack_source_ids_by_name, num_classes, diagnosis_loss_scale,
                           lr_scale=1.0,
                           atk_apgd_linf=None, atk_apgd_l2=None, gpgd_bases=None):
    attack_inputs = {}

    for name in attack_names:
        if name == 'Clean':
            attack_inputs[name] = img
            continue

        target_threat_domain_indices = None
        if name in attack_source_ids_by_name:
            target_threat_domain_indices = model.get_threat_domain_indices(attack_source_ids_by_name[name])

        if name == 'APGD_Linf':
            if atk_apgd_linf is None or target_threat_domain_indices is None:
                raise ValueError('Adaptive APGD_Linf is not initialized correctly')
            attack_inputs[name] = atk_apgd_linf(img, tgt, target_threat_domain_indices)
        elif name == 'APGD_L2':
            if atk_apgd_l2 is None or target_threat_domain_indices is None:
                raise ValueError('Adaptive APGD_L2 is not initialized correctly')
            attack_inputs[name] = atk_apgd_l2(img, tgt, target_threat_domain_indices)
        elif name == 'ACE':
            attack_inputs[name] = ace_adaptive_diagnosis_attack(img, tgt, target_threat_domain_indices, model, device, lr=1 * lr_scale, max_iterations=10, num_classes=num_classes, diagnosis_loss_scale=diagnosis_loss_scale)
        elif name == 'HSVAdv':
            attack_inputs[name] = hsvadv_attack_adaptive_diagnosis(img, tgt, target_threat_domain_indices, model, device, lr=1 * lr_scale, max_iterations=10, num_classes=num_classes, diagnosis_loss_scale=diagnosis_loss_scale)
        elif name == 'ReColorAdv':
            attack_inputs[name] = recoloradv_adaptive_diagnosis_attack(img, tgt, target_threat_domain_indices, model, device, lr=0.01 * lr_scale, max_iterations=10, num_classes=num_classes, diagnosis_loss_scale=diagnosis_loss_scale)
        elif name == 'ALA':
            attack_inputs[name] = ala_attack_adaptive_diagnosis(img, tgt, target_threat_domain_indices, model, device, lr=1 * lr_scale, max_iterations=10, num_classes=num_classes, diagnosis_loss_scale=diagnosis_loss_scale)
        elif name == 'RetouchUAA':
            attack_inputs[name] = retouch_uaa_attack_adaptive_diagnosis(
                img,
                tgt,
                target_threat_domain_indices,
                model,
                device,
                lr=0.1 * lr_scale,
                max_iterations=10,
                num_classes=num_classes,
                diagnosis_loss_scale=diagnosis_loss_scale,
            )
        elif name == 'GPGD':
            if gpgd_bases is None:
                raise ValueError('Adaptive GPGD requires PCA bases, but `gpgd_bases` is None')
            attack_inputs[name] = gpgd_attack_adaptive_diagnosis(
                img,
                tgt,
                target_threat_domain_indices,
                model,
                device,
                bases_dict=gpgd_bases,
                steps=10,
                epsilon=2.0,
                num_classes=num_classes,
                proj='l2',
                diagnosis_loss_scale=diagnosis_loss_scale,
            )
        elif name == 'StAdv':
            attack_inputs[name] = stadv_attack_adaptive_diagnosis(
                img,
                tgt,
                target_threat_domain_indices,
                model,
                device,
                eps=0.045,
                steps=10,
                mode='linf',
                diagnosis_loss_scale=diagnosis_loss_scale,
            )
        else:
            raise ValueError(f'Unsupported adaptive diagnosis attack: {name}')

    return attack_inputs


def evaluate_adaptive_diagnosis(testloader, model, device, num_classes, args, diagnosis_loss_scale):
    model.eval()
    cls_metrics = {k: AverageMeter() for k in args.test_attacks}
    diagnosis_metrics = {k: AverageMeter() for k in args.test_attacks if k in args.attack_names}

    atk_apgd_linf = None
    atk_apgd_l2 = None
    if 'APGD_Linf' in args.test_attacks:
        atk_apgd_linf = AdaptiveDiagnosisAPGD(model, norm='Linf', eps=8 / 255, steps=args.apgd_steps, n_restarts=1,
                                     seed=args.seed, loss='ce', eot_iter=1, rho=.75, verbose=False,
                                     diagnosis_loss_scale=diagnosis_loss_scale)
    if 'APGD_L2' in args.test_attacks:
        atk_apgd_l2 = AdaptiveDiagnosisAPGD(model, norm='L2', eps=0.5, steps=args.apgd_steps, n_restarts=1,
                                   seed=args.seed, loss='ce', eot_iter=1, rho=.75, verbose=False,
                                   diagnosis_loss_scale=diagnosis_loss_scale)

    pbar = tqdm(testloader, desc=f'Adaptive diagnosis scale={diagnosis_loss_scale}', leave=True)
    for step_idx, (img, tgt) in enumerate(pbar, start=1):
        img = img.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        attack_source_ids_by_name = build_attack_source_ids(
            args.test_attacks, img.size(0), device, args.attack_names
        )
        lr_scale = 0.1 if _is_tiny_dataset(args.dataset) else 1.0

        if hasattr(model, 'set_bpda'):
            model.set_bpda(True)

        attack_inputs = generate_adaptive_diagnosis_batch(
            img,
            tgt,
            model,
            device,
            args.test_attacks,
            attack_source_ids_by_name,
            num_classes,
            diagnosis_loss_scale,
            lr_scale=lr_scale,
            atk_apgd_linf=atk_apgd_linf,
            atk_apgd_l2=atk_apgd_l2,
            gpgd_bases=getattr(args, 'gpgd_bases', None),
        )

        if hasattr(model, 'set_bpda'):
            model.set_bpda(False)

        evaluate_attack_inputs(model, attack_inputs, tgt, attack_source_ids_by_name, cls_metrics, diagnosis_metrics)

        postfix = {}
        for name in args.test_attacks:
            if cls_metrics[name].count == 0:
                continue
            cls_val = f"{cls_metrics[name].avg:.1f}"
            if name in diagnosis_metrics and diagnosis_metrics[name].count > 0:
                postfix[name] = f"{cls_val}/{diagnosis_metrics[name].avg:.1f}"
            else:
                postfix[name] = cls_val
        pbar.set_postfix(postfix)

        if args.max_batches > 0 and step_idx >= args.max_batches:
            break

    pbar.close()
    return cls_metrics, diagnosis_metrics


def print_scale_summary(baseline_cls, baseline_diagnosis_accuracy, results_by_scale, attack_list):
    scales = sorted(results_by_scale.keys())
    header = f"{'Attack':<12} | {'Baseline':<10}"
    for s in scales:
        header += f" | {f'Scale={s}':<10}"
    sep_line = '-' * len(header)

    print('\n' + '=' * max(90, len(header)))
    print('Adaptive Diagnosis Multi-Scale Summary')
    print('=' * max(90, len(header)))

    print('\n[1] Classification Accuracy')
    print(sep_line)
    print(header)
    print(sep_line)
    for domain in attack_list:
        row = f"{domain:<12} | {baseline_cls[domain].avg:>9.2f}%"
        for s in scales:
            row += f" | {results_by_scale[s][0][domain].avg:>9.2f}%"
        print(row)
    print(sep_line)

    print('\n[2] Domain Accuracy')
    print(sep_line)
    print(header)
    print(sep_line)
    for domain in attack_list:
        row = f"{domain:<12} | {baseline_diagnosis_accuracy[domain].avg:>9.2f}%"
        for s in scales:
            row += f" | {results_by_scale[s][1][domain].avg:>9.2f}%"
        print(row)
    print(sep_line)


def build_model(args, device):
    model_kwargs = dict(
        backbone=args.backbone,
        dataset=args.dataset,
        num_attack_sources=args.num_attack_sources,
        num_threat_domains=args.num_threat_domains,
        num_frequency_experts=args.num_threat_domains,
        num_classes=args.num_classes,
        attack_names=args.attack_names,
    )
    if args.backbone == 'mobilevit':
        model_kwargs['size'] = _infer_input_size(args.dataset)
    model = build_tafd_model(**model_kwargs).to(device)
    return model


def load_checkpoint_to_model(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'model' not in ckpt:
        raise RuntimeError('Checkpoint does not contain a `model` field')
    if 'threat_domain_diagnosis_state' not in ckpt['model']:
        raise RuntimeError('Checkpoint does not contain `threat_domain_diagnosis_state`')
    td_state = ckpt['model']['threat_domain_diagnosis_state']
    if 'source_to_domain' not in td_state:
        raise RuntimeError('Checkpoint diagnosis state does not contain `source_to_domain`')
    mapping = td_state['source_to_domain']
    if not isinstance(mapping, dict) or len(mapping) == 0:
        raise RuntimeError('Checkpoint `source_to_domain` mapping is empty or invalid')
    max_domain_id = max(int(v) for v in mapping.values())
    if max_domain_id >= int(model.num_threat_domains):
        raise RuntimeError(
            f'Checkpoint domain ID {max_domain_id} is incompatible with num_threat_domains={model.num_threat_domains}'
        )

    missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
    if missing or unexpected:
        print(f'[WARN] State-dict mismatch: missing={len(missing)}, unexpected={len(unexpected)}')
    return ckpt, td_state


def main(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    attack_cfg = ATTACK_UNIONS[args.attack_union]
    args.num_attack_sources = attack_cfg['num_attack_sources']
    args.attack_names = list(attack_cfg['attack_names'])
    args.test_attacks = list(attack_cfg['test_attacks'])

    args.scales = parse_scales(args.scales)
    args.num_classes = _infer_num_classes(args.dataset, fallback=args.num_classes)
    if not os.path.isabs(args.resume):
        resume_candidate = os.path.join(BASE_DIR, args.resume)
        if os.path.exists(resume_candidate):
            args.resume = resume_candidate

    inferred_domains = infer_domains_from_checkpoint(args.resume, device)
    if inferred_domains is not None and inferred_domains != args.num_threat_domains:
        print(f'[INFO] override --num_threat_domains from {args.num_threat_domains} to checkpoint value {inferred_domains}')
        args.num_threat_domains = inferred_domains

    print('==> Preparing data')
    trainloader, testloader = GetDataLoader(
        args.dataset,
        args.batch_size,
        args.test_batch_size,
        getattr(args, 'dataset_path', os.path.join(BASE_DIR, 'datasets')),
        num_workers=args.num_workers,
        pin_memory=not args.disable_pin_memory,
        persistent_workers=not args.disable_persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )

    prepare_evaluation_gpgd_bases(args, trainloader, device)

    print('==> Building model')
    model = build_model(args, device)
    model.count_frequency_convolutions()

    print('==> Loading checkpoint')
    _, td_state = load_checkpoint_to_model(model, args.resume, device)
    print(f'==> Restored source_to_domain: {td_state["source_to_domain"]}')
    if hasattr(model, 'threat_domain_diagnosis'):
        print(f'==> Current diagnosis mapping: {model.threat_domain_diagnosis.get_assignment_status()}')

    print('\n' + '=' * 80)
    print('Adaptive threat-domain diagnosis evaluation across loss scales')
    print('=' * 80)
    print(f'Dataset: {args.dataset}')
    print(f'Backbone: {args.backbone}')
    print(f'Checkpoint: {args.resume}')
    print(f'Domains: {args.num_threat_domains}')
    print(f'Test attacks: {args.test_attacks}')
    print(f'Scales: {args.scales}')
    if args.max_batches > 0:
        print(f'Max Batches: {args.max_batches}')
    print('=' * 80)

    results_by_scale = {}
    for scale in args.scales:
        print(f'\n>>> Running adaptive diagnosis attacks with diagnosis_loss_scale = {scale}')
        adaptive_diagnosis_cls, adaptive_diagnosis_accuracy = evaluate_adaptive_diagnosis(testloader, model, device, args.num_classes, args, diagnosis_loss_scale=scale)
        results_by_scale[scale] = (adaptive_diagnosis_cls, adaptive_diagnosis_accuracy)

    print('\n>>> Running Baseline')
    baseline_cls, baseline_diagnosis_accuracy = validation_baseline(testloader, model, device, args.num_classes, args)

    attack_list = [atk for atk in args.test_attacks if atk in args.attack_names]
    print_scale_summary(baseline_cls, baseline_diagnosis_accuracy, results_by_scale, attack_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Adaptive diagnosis multi-scale supplemental evaluation')
    parser.add_argument(
        '--dataset',
        type=str,
        default='CIFAR100',
        choices=['CIFAR10', 'CIFAR100', 'Imagenette'],
    )
    parser.add_argument('--dataset_path', type=str, default=os.path.join(BASE_DIR, 'datasets'))
    parser.add_argument('--backbone', type=str, default='resnet', choices=['resnet', 'mobilevit'])
    parser.add_argument('--attack_union', type=str, default='canonical', choices=['canonical', 'broader'])
    parser.add_argument('--num_threat_domains', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--disable_pin_memory', action='store_true')
    parser.add_argument('--disable_persistent_workers', action='store_true')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--apgd_steps', type=int, default=100)
    parser.add_argument('--scales', type=str, default='0.1,1,5,10')
    parser.add_argument('--max_batches', type=int, default=0)
    parser.add_argument('--gpgd_basis_path', type=str, default='')
    parser.add_argument('--gpgd_rank', type=int, default=128)
    parser.add_argument('--gpgd_max_per_class', type=int, default=600)

    main(parser.parse_args())
