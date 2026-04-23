#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval.py
-------

Usage:
  python eval.py --checkpoint ./results/.../latest_model.pth \
                 --dataset CIFAR100 --backbone resnet --attack_config v10 --gpu 0
"""

import os
import sys
import argparse
import numpy as np
import random

import torch
import torch.nn as nn

# Reuse core components from train.py
from train import (
    ATTACK_CONFIGS, ALL_ATTACKS,
    validation_pgd, load_checkpoint,
    _infer_num_classes,
    prepare_subspace_bases,
)
from utils.datasets_utils import GetDataLoader
from models.encoder import create_encoder


def main():
    parser = argparse.ArgumentParser(
        "TaFD: Threat-aware Frequency Decoupling — Evaluation"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint (e.g., latest_model.pth)")
    parser.add_argument("--dataset", type=str, default="CIFAR100",
                        help="Dataset name (e.g., CIFAR10 / CIFAR100)")
    parser.add_argument("--backbone", type=str, default="resnet",
                        choices=["resnet", "mobilevit"],
                        help="Model backbone: resnet or mobilevit")
    parser.add_argument("--attack_config", type=str, default="v10",
                        choices=["v10", "v20"],
                        help="Attack configuration: v10 (7 attacks) or v20 (5 attacks)")
    parser.add_argument("--attacks", type=str, nargs="+", default=None,
                        choices=['Clean',
                                 'Linf_PGD', 'L2_PGD',       # training PGD-10
                                 'Linf_APGD', 'L2_APGD',     # evaluation APGD-100
                                 'ACE', 'ReColorAdv', 'HSVAdv', 'ALA', 'RetouchUAA', 'GPGD', 'StAdv'],
                        help="Specific attacks to evaluate; defaults to full test set from attack_config")
    parser.add_argument("--domains", type=int, default=6,
                        help="Number of threat domains")
    parser.add_argument("--n_cls", type=int, default=10,
                        help="Number of classes (auto-inferred from dataset)")
    parser.add_argument("--test_batch_size", type=int, default=16,
                        help="Test batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--log_every", type=int, default=10,
                        help="Log interval (batches)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index")
    parser.add_argument("--seed", type=int, default=0)

    # GPGD attack options
    parser.add_argument("--subspace_basis_path", type=str, default="")
    parser.add_argument("--subspace_rank", type=int, default=128)
    parser.add_argument("--subspace_max_per_class", type=int, default=600)

    args = parser.parse_args()

    # ── Device ──
    global device
    if torch.cuda.is_available() and args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    # Patch the module-level device used by train.py functions
    import train as train_module
    train_module.device = device

    # ── Seed ──
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Attack config ──
    attack_cfg = ATTACK_CONFIGS[args.attack_config]
    args.num_sources = attack_cfg['num_sources']
    args.train_attacks = list(attack_cfg['train_attacks'])
    args.test_attacks = list(attack_cfg['test_attacks'])
    args.domain_names = list(attack_cfg['domain_names'])
    args.subspace_bases = None

    if not args.attacks:
        args.attacks = list(args.test_attacks)

    # ── Auto-infer n_cls ──
    inferred_n = _infer_num_classes(args.dataset, fallback=args.n_cls)
    if inferred_n != args.n_cls:
        print(f"[Info] n_cls auto-updated: {args.n_cls} → {inferred_n}")
        args.n_cls = inferred_n

    # ── Result dir (for GPGD bases) ──
    args.result_dir = os.path.dirname(args.checkpoint) or "."

    # ── Data ──
    print(f"==> Loading dataset: {args.dataset}")
    _, testloader = GetDataLoader(
        args.dataset,
        train_batch_size=128,  # unused
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
    )

    # ── Model ──
    print(f"==> Building model: {args.backbone}")
    model_kwargs = dict(
        backbone=args.backbone,
        num_sources=args.num_sources,
        num_domains=args.domains,
        num_classes=args.n_cls,
        dataset=args.dataset,
        source_names=args.domain_names,
    )
    if args.backbone == 'mobilevit':
        model_kwargs['size'] = 32  # CIFAR-10/100
    model = create_encoder(**model_kwargs).to(device)

    # ── Dummy optimizer (needed for load_checkpoint interface) ──
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # ── Load checkpoint ──
    if not os.path.isfile(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    print(f"==> Loading checkpoint: {args.checkpoint}")
    epoch, acc_hist, domain_hist = load_checkpoint(
        model, optimizer, args.checkpoint
    )
    print(f"==> Checkpoint from epoch {epoch - 1}")

    # ── Prepare GPGD bases if needed ──
    if 'GPGD' in args.attacks:
        trainloader, _ = GetDataLoader(
            args.dataset,
            train_batch_size=128,
            test_batch_size=args.test_batch_size,
            num_workers=args.num_workers,
        )
        prepare_subspace_bases(args, trainloader)

    # ── Evaluate ──
    print(f"\n{'='*60}")
    print(f"  Evaluating: {args.dataset} | {args.backbone} | {args.attack_config}")
    print(f"  Attacks: {args.attacks}")
    print(f"{'='*60}")

    criterion = nn.CrossEntropyLoss()
    clean_acc, apgd_linf_acc, _, _ = validation_pgd(
        epoch - 1, testloader, criterion, model, args.n_cls,
        acc_hist={}, domain_hist={}, args=args
    )

    print(f"\n{'='*60}")
    print("  Final Results")
    print(f"  Clean Accuracy:      {clean_acc:.2f}%")
    print(f"  APGD-Linf Accuracy:  {apgd_linf_acc:.2f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
