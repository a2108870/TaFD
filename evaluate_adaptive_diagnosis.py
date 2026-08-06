"""Evaluate adaptive attacks against TaFD threat-domain diagnosis."""

import argparse

import main_eval_gateatk_scales as eval_impl
from tafd_cli import to_internal_attack_name, to_public_attack_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate adaptive attacks against TaFD threat-domain diagnosis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default="CIFAR100")
    parser.add_argument("--dataset_path", type=str, default="./datasets/")
    parser.add_argument("--backbone", type=str, default="resnet", choices=["resnet", "mobilevit"])
    parser.add_argument("--attack_config", type=str, default="v10", choices=["v10", "v20"])
    parser.add_argument("--domains", type=int, default=2)
    parser.add_argument("--n_cls", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--disable_pin_memory", action="store_true")
    parser.add_argument("--disable_persistent_workers", action="store_true")
    parser.add_argument("--resume", type=str, required=True, help="Checkpoint path to evaluate.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--apgd_steps", type=int, default=100)
    parser.add_argument("--scales", type=str, default="0.1,1,5,10")
    parser.add_argument(
        "--attacks",
        type=str,
        nargs="+",
        default=None,
        choices=[to_public_attack_name(name) for name in eval_impl.SUPPORTED_GATE_ATTACKS],
    )
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--subspace_basis_path", type=str, default="")
    parser.add_argument("--subspace_rank", type=int, default=128)
    parser.add_argument("--subspace_max_per_class", type=int, default=600)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.attacks:
        args.attacks = [to_internal_attack_name(name) for name in args.attacks]
    eval_impl.main(args)


if __name__ == "__main__":
    main()
