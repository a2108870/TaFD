"""Shared command-line helpers for the public TaFD release.

The original training files are kept close to the experiment code used for the
paper.  This module provides clean, paper-aligned entrypoints without changing
the underlying training and evaluation logic.
"""

from __future__ import annotations

import argparse
import os
import random
from types import ModuleType

import numpy as np
import torch


ATTACK_CHOICES = [
    "Clean",
    "APGD_Linf",
    "APGD_L2",
    "SPSA",
    "ACE",
    "ReColorAdv",
    "Hue",
    "Light",
    "UAA",
    "SUB",
    "STADV",
]


def build_parser(
    description: str,
    *,
    include_domain_route_ablation: bool = False,
    include_mvit_fdconv: bool = False,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR100",
        help="Dataset name: CIFAR10, CIFAR100, Tiny_32_10class, Tiny_32_200class, or Imagenette.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./datasets/",
        help="Root directory containing the datasets.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet",
        choices=["resnet", "mobilevit"],
        help="Classifier backbone.",
    )
    parser.add_argument(
        "--attack_config",
        type=str,
        default="v10",
        choices=["v10", "v20"],
        help="Heterogeneous attack set used for training and evaluation.",
    )
    parser.add_argument(
        "--domains",
        type=int,
        default=2,
        help="Number of threat domains K. The paper default is K=2.",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Main-network learning rate.")
    parser.add_argument(
        "--lr_domain",
        type=float,
        default=0.001,
        help="Threat-domain diagnosis learning rate.",
    )
    parser.add_argument("--weight_decay", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--disable_pin_memory", action="store_true")
    parser.add_argument("--disable_persistent_workers", action="store_true")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--end_epoch", type=int, default=76)
    parser.add_argument("--eval_freq", type=int, default=25)
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Optional checkpoint path for resuming training or evaluating a model.",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="",
        help="Output directory. A paper-style directory is generated when omitted.",
    )
    parser.add_argument(
        "--n_cls",
        type=int,
        default=10,
        help="Number of classes. Inferred automatically for known datasets.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--domain_loss_weight",
        type=float,
        default=1.0,
        help="Weight of the threat-domain diagnosis loss.",
    )
    parser.add_argument(
        "--map_update_every",
        type=int,
        default=50,
        help="Iteration interval for updating threat-domain assignments.",
    )
    if include_domain_route_ablation:
        parser.add_argument(
            "--ablate_domain_route",
            type=str,
            default="none",
            choices=["none", "fixed0", "uniform"],
            help="Optional route-dispatch ablation used by the no-diagnosis script.",
        )
        parser.add_argument(
            "--ablate_domain_route_id",
            type=int,
            default=0,
            help="Route id used when --ablate_domain_route=fixed0.",
        )
    parser.add_argument(
        "--attacks",
        type=str,
        nargs="+",
        default=None,
        choices=ATTACK_CHOICES,
        help="Evaluation attacks. Defaults to the selected attack_config.",
    )
    parser.add_argument(
        "--subspace_basis_path",
        type=str,
        default="",
        help="Optional PCA-basis file for SUB/GPGD-style subspace attacks.",
    )
    parser.add_argument("--subspace_rank", type=int, default=128)
    parser.add_argument("--subspace_max_per_class", type=int, default=600)
    if include_mvit_fdconv:
        parser.add_argument(
            "--mvit_fdconv",
            action="store_true",
            help="Use FC-Conv in MobileViT block input convolutions.",
        )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="CUDA device id. Use a negative value for CPU.",
    )
    return parser


def prepare_args(args: argparse.Namespace, impl: ModuleType, *, result_prefix: str) -> argparse.Namespace:
    if torch.cuda.is_available() and args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        impl.device = torch.device(f"cuda:{args.gpu}")
    else:
        impl.device = torch.device("cpu")

    attack_cfg = impl.ATTACK_CONFIGS[args.attack_config]
    args.num_sources = attack_cfg["num_sources"]
    args.train_attacks = list(attack_cfg["train_attacks"])
    args.test_attacks = list(attack_cfg["test_attacks"])
    args.domain_names = list(attack_cfg["domain_names"])
    args.subspace_bases = None

    if not args.attacks:
        args.attacks = list(args.test_attacks)

    args.n_cls = impl._infer_num_classes(args.dataset, fallback=args.n_cls)

    if not hasattr(args, "mvit_fdconv"):
        args.mvit_fdconv = False

    if not args.result_dir:
        args.result_dir = (
            f"./results/{result_prefix}_{args.backbone}_{args.dataset}_d{args.domains}"
            f"_{args.attack_config}_lr{args.lr}_dlr{args.lr_domain}"
            f"_dw{args.domain_loss_weight}_bs{args.batch_size}"
            f"_ep{args.end_epoch}_seed{args.seed}"
        )

    os.makedirs(args.result_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    print("[Config] dataset:", args.dataset)
    print("[Config] classes:", args.n_cls)
    print("[Config] backbone:", args.backbone)
    print("[Config] attack_config:", args.attack_config)
    print("[Config] train_attacks:", args.train_attacks)
    print("[Config] eval_attacks:", args.attacks)
    print("[Config] threat_domains:", args.domains)
    print("[Config] device:", impl.device)
    print("[Config] result_dir:", args.result_dir)
    return args
