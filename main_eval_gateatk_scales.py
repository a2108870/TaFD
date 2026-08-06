#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from attacks.ace_gateatk import ACE as ACE_gateatk
from attacks.hue_gateatk import hue_atk as hue_atk_gateatk
from attacks.light_gateatk import light_atk as light_atk_gateatk
from attacks.recoloradv_gateatk import ReColorAdv as ReColorAdv_gateatk
from attacks.stadv_gateatk import stadv_attack as stadv_attack_gateatk
from attacks.subspace import build_pca_bases, load_pca_bases, save_pca_bases
from attacks.subspace_gateatk import subspace_atk as subspace_atk_gateatk
from attacks.uaa_gateatk import UAA_atk as UAA_atk_gateatk
from main_train_pgdtrain import (
    ATTACK_CONFIGS,
    AverageMeter,
    _build_apgd_attack,
    _infer_input_size,
    _infer_num_classes,
    _is_tiny_dataset,
    accuracy,
    generate_attack_batch,
)
from models.encoder import create_encoder
from torchattacks.attacks.apgd_gateatk import APGD_GateAtk
from utils.datasets_utils import GetDataLoader


SUPPORTED_GATE_ATTACKS = ['Clean', 'APGD_Linf', 'APGD_L2', 'ACE', 'ReColorAdv', 'Hue', 'Light', 'UAA', 'SUB', 'STADV']
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_scales(scales_text):
    values = []
    for item in scales_text.split(','):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError('`--scales` 不能为空')
    return values


def domain_id_from_attack(attack_name, domain_names):
    if attack_name in domain_names:
        return domain_names.index(attack_name)
    return None


def build_domain_source_ids(attack_names, batch_size, device, domain_names):
    domain_ids = {}
    for name in attack_names:
        idx = domain_id_from_attack(name, domain_names)
        if idx is not None:
            domain_ids[name] = torch.full((batch_size,), idx, device=device, dtype=torch.long)
    return domain_ids


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


def resolve_subspace_basis_path(args):
    if args.subspace_basis_path:
        if os.path.isabs(args.subspace_basis_path):
            return args.subspace_basis_path
        return os.path.join(BASE_DIR, args.subspace_basis_path)

    resume_dir = os.path.dirname(args.resume)
    filename = f'subspace_pca_bases_{args.dataset}_c{args.n_cls}_r{args.subspace_rank}.pth'
    return os.path.join(resume_dir, filename)


def prepare_eval_subspace_bases(args, trainloader, device):
    if 'SUB' not in args.attacks:
        args.subspace_bases = None
        return

    basis_path = resolve_subspace_basis_path(args)
    args.subspace_basis_path = basis_path

    if os.path.isfile(basis_path):
        print(f'[SUB] loading PCA bases: {basis_path}')
        args.subspace_bases = load_pca_bases(basis_path, device=device)
        return

    if trainloader is None:
        raise RuntimeError('SUB evaluation requires trainloader to build PCA bases, but trainloader is unavailable')

    print(f'[SUB] PCA bases not found, building: {basis_path}')
    args.subspace_bases = build_pca_bases(
        trainloader,
        n_classes=args.n_cls,
        rank=args.subspace_rank,
        max_per_class=args.subspace_max_per_class,
        device=torch.device('cpu'),
    )
    os.makedirs(os.path.dirname(basis_path) or '.', exist_ok=True)
    save_pca_bases(basis_path, args.subspace_bases)
    print(f'[SUB] PCA bases saved: {basis_path}')


def evaluate_attack_inputs(model, attack_inputs, tgt, domain_ids, cls_metrics, gate_metrics):
    if not attack_inputs:
        return

    batch_size = tgt.size(0)
    ordered_names = list(attack_inputs.keys())
    combined_inputs = torch.cat([attack_inputs[name] for name in ordered_names], dim=0)

    with torch.no_grad():
        out_tuple = model(combined_inputs, domain_ids=None)
        logits_all = out_tuple[0] if isinstance(out_tuple, tuple) else out_tuple
        domain_logits_all = out_tuple[2] if isinstance(out_tuple, tuple) and len(out_tuple) > 2 else None

        for idx, name in enumerate(ordered_names):
            start = idx * batch_size
            end = start + batch_size
            logits = logits_all[start:end]
            acc1 = accuracy(logits, tgt, (1,))[0]
            cls_metrics[name].update(acc1.item(), batch_size)

            if name in domain_ids and domain_logits_all is not None:
                domain_logit = domain_logits_all[start:end]
                pred_domain = domain_logit.max(1)[1]
                true_gate = model.get_domain_labels(domain_ids[name])
                d_acc = (pred_domain == true_gate).float().mean() * 100
                gate_metrics[name].update(d_acc.item(), batch_size)


def validation_baseline(testloader, model, device, n_cls, args):
    model.eval()
    cls_metrics = {k: AverageMeter() for k in args.attacks}
    gate_metrics = {k: AverageMeter() for k in args.attacks if k in args.domain_names}

    atk_apgd_linf = _build_apgd_attack(model, norm='Linf', eps=8 / 255, steps=args.apgd_steps) if 'APGD_Linf' in args.attacks else None
    atk_apgd_l2 = _build_apgd_attack(model, norm='L2', eps=0.5, steps=args.apgd_steps) if 'APGD_L2' in args.attacks else None

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
            attack_names=args.attacks,
            atk_apgd_linf=atk_apgd_linf,
            atk_apgd_l2=atk_apgd_l2,
            atk_spsa=None,
            lr_scale=lr_scale,
            ncls=n_cls,
            subspace_bases=getattr(args, 'subspace_bases', None),
        )
        domain_ids = build_domain_source_ids(attack_inputs.keys(), img.size(0), device, args.domain_names)

        if hasattr(model, 'set_bpda'):
            model.set_bpda(False)

        evaluate_attack_inputs(model, attack_inputs, tgt, domain_ids, cls_metrics, gate_metrics)

        postfix = {}
        for name in args.attacks:
            if cls_metrics[name].count == 0:
                continue
            cls_val = f"{cls_metrics[name].avg:.1f}"
            if name in gate_metrics and gate_metrics[name].count > 0:
                postfix[name] = f"{cls_val}/{gate_metrics[name].avg:.1f}"
            else:
                postfix[name] = cls_val
        pbar.set_postfix(postfix)

        if args.max_batches > 0 and step_idx >= args.max_batches:
            break

    pbar.close()
    return cls_metrics, gate_metrics


def generate_gateatk_batch(img, tgt, model, device, attack_names, domain_ids, ncls, gate_loss_scale,
                           lr_scale=1.0,
                           atk_apgd_linf=None, atk_apgd_l2=None, subspace_bases=None):
    attack_inputs = {}

    for name in attack_names:
        if name == 'Clean':
            attack_inputs[name] = img
            continue

        true_gate = None
        if name in domain_ids:
            true_gate = model.get_domain_labels(domain_ids[name])

        if name == 'APGD_Linf':
            if atk_apgd_linf is None or true_gate is None:
                raise ValueError('APGD_Linf GateAtk is not initialized correctly')
            attack_inputs[name] = atk_apgd_linf(img, tgt, true_gate)
        elif name == 'APGD_L2':
            if atk_apgd_l2 is None or true_gate is None:
                raise ValueError('APGD_L2 GateAtk is not initialized correctly')
            attack_inputs[name] = atk_apgd_l2(img, tgt, true_gate)
        elif name == 'ACE':
            attack_inputs[name] = ACE_gateatk(img, tgt, true_gate, model, device, lr=1 * lr_scale, max_iterations=10, ncls=ncls, gate_loss_scale=gate_loss_scale)
        elif name == 'Hue':
            attack_inputs[name] = hue_atk_gateatk(img, tgt, true_gate, model, device, lr=1 * lr_scale, max_iterations=10, ncls=ncls, gate_loss_scale=gate_loss_scale)
        elif name == 'ReColorAdv':
            attack_inputs[name] = ReColorAdv_gateatk(img, tgt, true_gate, model, device, lr=0.01 * lr_scale, max_iterations=10, ncls=ncls, gate_loss_scale=gate_loss_scale)
        elif name == 'Light':
            attack_inputs[name] = light_atk_gateatk(img, tgt, true_gate, model, device, lr=1 * lr_scale, max_iterations=10, ncls=ncls, gate_loss_scale=gate_loss_scale)
        elif name == 'UAA':
            attack_inputs[name] = UAA_atk_gateatk(img, tgt, true_gate, model, device, lr=0.1 * lr_scale, max_iterations=10, ncls=ncls, gate_loss_scale=gate_loss_scale)
        elif name == 'SUB':
            if subspace_bases is None:
                raise ValueError('SUB GateAtk requires PCA bases, but `subspace_bases` is None')
            attack_inputs[name] = subspace_atk_gateatk(
                img,
                tgt,
                true_gate,
                model,
                device,
                bases_dict=subspace_bases,
                steps=10,
                epsilon=2.0,
                ncls=ncls,
                proj='l2',
                gate_loss_scale=gate_loss_scale,
            )
        elif name == 'STADV':
            attack_inputs[name] = stadv_attack_gateatk(
                img,
                tgt,
                true_gate,
                model,
                device,
                eps=0.045,
                steps=10,
                mode='linf',
                gate_loss_scale=gate_loss_scale,
            )
        else:
            raise ValueError(f'Unsupported GateAtk type: {name}')

    return attack_inputs


def validation_gateatk(testloader, model, device, n_cls, args, gate_loss_scale):
    model.eval()
    cls_metrics = {k: AverageMeter() for k in args.attacks}
    gate_metrics = {k: AverageMeter() for k in args.attacks if k in args.domain_names}

    atk_apgd_linf = None
    atk_apgd_l2 = None
    if 'APGD_Linf' in args.attacks:
        atk_apgd_linf = APGD_GateAtk(model, norm='Linf', eps=8 / 255, steps=args.apgd_steps, n_restarts=1,
                                     seed=args.seed, loss='ce', eot_iter=1, rho=.75, verbose=False,
                                     gate_loss_scale=gate_loss_scale)
    if 'APGD_L2' in args.attacks:
        atk_apgd_l2 = APGD_GateAtk(model, norm='L2', eps=0.5, steps=args.apgd_steps, n_restarts=1,
                                   seed=args.seed, loss='ce', eot_iter=1, rho=.75, verbose=False,
                                   gate_loss_scale=gate_loss_scale)

    pbar = tqdm(testloader, desc=f'GateAtk scale={gate_loss_scale}', leave=True)
    for step_idx, (img, tgt) in enumerate(pbar, start=1):
        img = img.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        domain_ids = build_domain_source_ids(args.attacks, img.size(0), device, args.domain_names)
        lr_scale = 0.1 if _is_tiny_dataset(args.dataset) else 1.0

        if hasattr(model, 'set_bpda'):
            model.set_bpda(True)

        attack_inputs = generate_gateatk_batch(
            img,
            tgt,
            model,
            device,
            args.attacks,
            domain_ids,
            n_cls,
            gate_loss_scale,
            lr_scale=lr_scale,
            atk_apgd_linf=atk_apgd_linf,
            atk_apgd_l2=atk_apgd_l2,
            subspace_bases=getattr(args, 'subspace_bases', None),
        )

        if hasattr(model, 'set_bpda'):
            model.set_bpda(False)

        evaluate_attack_inputs(model, attack_inputs, tgt, domain_ids, cls_metrics, gate_metrics)

        postfix = {}
        for name in args.attacks:
            if cls_metrics[name].count == 0:
                continue
            cls_val = f"{cls_metrics[name].avg:.1f}"
            if name in gate_metrics and gate_metrics[name].count > 0:
                postfix[name] = f"{cls_val}/{gate_metrics[name].avg:.1f}"
            else:
                postfix[name] = cls_val
        pbar.set_postfix(postfix)

        if args.max_batches > 0 and step_idx >= args.max_batches:
            break

    pbar.close()
    return cls_metrics, gate_metrics


def print_scale_summary(baseline_cls, baseline_gate, results_by_scale, attack_list):
    scales = sorted(results_by_scale.keys())
    header = f"{'Attack':<12} | {'Baseline':<10}"
    for s in scales:
        header += f" | {f'Scale={s}':<10}"
    sep_line = '-' * len(header)

    print('\n' + '=' * max(90, len(header)))
    print('GateAtk Multi-Scale Summary')
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
        row = f"{domain:<12} | {baseline_gate[domain].avg:>9.2f}%"
        for s in scales:
            row += f" | {results_by_scale[s][1][domain].avg:>9.2f}%"
        print(row)
    print(sep_line)


def build_model(args, device):
    model_kwargs = dict(
        backbone=args.backbone,
        dataset=args.dataset,
        num_sources=args.num_sources,
        num_domains=args.domains,
        fd_num_experts=args.domains,
        num_classes=args.n_cls,
        source_names=args.domain_names,
    )
    if args.backbone == 'mobilevit':
        model_kwargs['size'] = _infer_input_size(args.dataset)
    model = create_encoder(**model_kwargs).to(device)
    return model


def load_checkpoint_to_model(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'model' not in ckpt:
        raise RuntimeError('Checkpoint 中缺少 `model` 字段')
    if 'threat_domain_diagnosis_state' not in ckpt['model']:
        raise RuntimeError('Checkpoint 中缺少 `threat_domain_diagnosis_state`，无法安全构造 true_gate')
    td_state = ckpt['model']['threat_domain_diagnosis_state']
    if 'source_to_domain' not in td_state:
        raise RuntimeError('Checkpoint 中缺少 `source_to_domain`，无法安全构造 true_gate')
    mapping = td_state['source_to_domain']
    if not isinstance(mapping, dict) or len(mapping) == 0:
        raise RuntimeError('Checkpoint 中的 `source_to_domain` 非法或为空，无法安全构造 true_gate')
    max_domain_id = max(int(v) for v in mapping.values())
    if max_domain_id >= int(model.num_threat_domains):
        raise RuntimeError(
            f'Checkpoint 中 `source_to_domain` 最大域ID={max_domain_id}，但当前模型 domains={model.num_threat_domains}，不匹配'
        )

    missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
    if missing or unexpected:
        print(f'[WARN] state_dict 不完全匹配: missing={len(missing)}, unexpected={len(unexpected)}')
    return ckpt, td_state


def main(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    attack_cfg = ATTACK_CONFIGS[args.attack_config]
    args.num_sources = attack_cfg['num_sources']
    args.domain_names = list(attack_cfg['domain_names'])
    if not args.attacks:
        args.attacks = [atk for atk in attack_cfg['test_attacks'] if atk in SUPPORTED_GATE_ATTACKS]

    unsupported = [atk for atk in args.attacks if atk not in SUPPORTED_GATE_ATTACKS]
    if unsupported:
        raise ValueError(f'当前补充实验仅支持 {SUPPORTED_GATE_ATTACKS}，收到不支持项: {unsupported}')

    args.scales = parse_scales(args.scales)
    args.n_cls = _infer_num_classes(args.dataset, fallback=args.n_cls)
    if not os.path.isabs(args.resume):
        resume_candidate = os.path.join(BASE_DIR, args.resume)
        if os.path.exists(resume_candidate):
            args.resume = resume_candidate

    inferred_domains = infer_domains_from_checkpoint(args.resume, device)
    if inferred_domains is not None and inferred_domains != args.domains:
        print(f'[INFO] override --domains from {args.domains} to checkpoint value {inferred_domains}')
        args.domains = inferred_domains

    invalid_for_config = [atk for atk in args.attacks if atk not in attack_cfg['test_attacks']]
    if invalid_for_config:
        raise ValueError(f'Attacks {invalid_for_config} are not in attack_config={args.attack_config} test set {attack_cfg["test_attacks"]}')

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

    prepare_eval_subspace_bases(args, trainloader, device)

    print('==> 构建模型')
    model = build_model(args, device)
    model.count_frequency_convolutions()

    print('==> 加载 checkpoint')
    _, td_state = load_checkpoint_to_model(model, args.resume, device)
    print(f'==> 恢复的 source_to_domain: {td_state["source_to_domain"]}')
    if hasattr(model, 'threat_domain_diagnosis'):
        print(f'==> 当前映射状态: {model.threat_domain_diagnosis.get_mapping_status()}')

    print('\n' + '=' * 80)
    print('GateAtk 多尺度补充实验')
    print('=' * 80)
    print(f'Dataset: {args.dataset}')
    print(f'Backbone: {args.backbone}')
    print(f'Checkpoint: {args.resume}')
    print(f'Domains: {args.domains}')
    print(f'Attacks: {args.attacks}')
    print(f'Scales: {args.scales}')
    if args.max_batches > 0:
        print(f'Max Batches: {args.max_batches}')
    print('=' * 80)

    results_by_scale = {}
    for scale in args.scales:
        print(f'\n>>> Running GateAtk with gate_loss_scale = {scale}')
        gateatk_cls, gateatk_gate = validation_gateatk(testloader, model, device, args.n_cls, args, gate_loss_scale=scale)
        results_by_scale[scale] = (gateatk_cls, gateatk_gate)

    print('\n>>> Running Baseline')
    baseline_cls, baseline_gate = validation_baseline(testloader, model, device, args.n_cls, args)

    attack_list = [atk for atk in args.attacks if atk in args.domain_names]
    print_scale_summary(baseline_cls, baseline_gate, results_by_scale, attack_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('GateAtk multi-scale supplemental evaluation')
    parser.add_argument('--dataset', type=str, default='CIFAR100')
    parser.add_argument('--dataset_path', type=str, default=os.path.join(BASE_DIR, 'datasets'))
    parser.add_argument('--backbone', type=str, default='resnet', choices=['resnet', 'mobilevit'])
    parser.add_argument('--attack_config', type=str, default='v10', choices=['v10', 'v20'])
    parser.add_argument('--domains', type=int, default=2)
    parser.add_argument('--n_cls', type=int, default=100)
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
    parser.add_argument('--attacks', type=str, nargs='+', default=None,
                        choices=SUPPORTED_GATE_ATTACKS)
    parser.add_argument('--max_batches', type=int, default=0)
    parser.add_argument('--subspace_basis_path', type=str, default='')
    parser.add_argument('--subspace_rank', type=int, default=128)
    parser.add_argument('--subspace_max_per_class', type=int, default=600)

    main(parser.parse_args())
