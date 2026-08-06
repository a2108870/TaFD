"""Ablation: generate frequency masks directly."""

import training.without_basis_parameterized_mask as tafd_impl
from tafd_cli import build_parser, prepare_args


def main() -> None:
    parser = build_parser(
        "Train the TaFD ablation with direct frequency-mask generation.",
    )
    args = prepare_args(
        parser.parse_args(),
        tafd_impl,
        result_prefix="tafd_direct_frequency_mask",
    )
    tafd_impl.main(args)


if __name__ == "__main__":
    main()
