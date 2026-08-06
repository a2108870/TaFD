"""Shared command-line helpers for the public TaFD release."""

from __future__ import annotations

import argparse
import os
import random
from types import ModuleType

import numpy as np
import torch

def build_parser(
    description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR100",
        choices=["CIFAR10", "CIFAR100", "Imagenette"],
        help="Dataset used in the paper experiments.",
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
        "--attack_union",
        type=str,
        default="canonical",
        choices=["canonical", "broader"],
        help="Heterogeneous attack set used for training and evaluation.",
    )
    parser.add_argument(
        "--num_threat_domains",
        type=int,
        default=2,
        help="Number of threat domains K. The paper default is K=2.",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Main-network learning rate.")
    parser.add_argument(
        "--diagnosis_lr",
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
        "--num_classes",
        type=int,
        default=10,
        help="Number of classes. Inferred automatically for known datasets.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--diagnosis_loss_weight",
        type=float,
        default=1.0,
        help="Weight of the threat-domain diagnosis loss.",
    )
    parser.add_argument(
        "--assignment_update_interval",
        type=int,
        default=50,
        help="Iteration interval for updating threat-domain assignments.",
    )
    parser.add_argument(
        "--gpgd_basis_path",
        type=str,
        default="",
        help="Optional PCA-basis file for the GPGD attack.",
    )
    parser.add_argument("--gpgd_rank", type=int, default=128)
    parser.add_argument("--gpgd_max_per_class", type=int, default=600)
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

    attack_cfg = impl.ATTACK_UNIONS[args.attack_union]
    args.num_attack_sources = attack_cfg["num_attack_sources"]
    args.train_attacks = list(attack_cfg["train_attacks"])
    args.test_attacks = list(attack_cfg["test_attacks"])
    args.attack_names = list(attack_cfg["attack_names"])
    args.gpgd_bases = None

    args.num_classes = impl._infer_num_classes(args.dataset, fallback=args.num_classes)

    if not args.result_dir:
        args.result_dir = (
            f"./results/{result_prefix}_{args.backbone}_{args.dataset}"
            f"_k{args.num_threat_domains}_{args.attack_union}_lr{args.lr}"
            f"_diagnosislr{args.diagnosis_lr}"
            f"_diagnosisw{args.diagnosis_loss_weight}_bs{args.batch_size}"
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
    print("[Config] classes:", args.num_classes)
    print("[Config] backbone:", args.backbone)
    print("[Config] attack_union:", args.attack_union)
    print("[Config] train_attacks:", args.train_attacks)
    print("[Config] test_attacks:", args.test_attacks)
    print("[Config] threat_domains:", args.num_threat_domains)
    print("[Config] device:", impl.device)
    print("[Config] result_dir:", args.result_dir)
    return args
