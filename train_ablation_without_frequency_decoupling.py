"""Ablation: replace FC-Conv with standard convolution."""

import training.without_frequency_decoupling as tafd_impl
from tafd_cli import build_parser, prepare_args


def main() -> None:
    parser = build_parser(
        "Train the TaFD ablation with standard convolution instead of FC-Conv.",
    )
    args = prepare_args(
        parser.parse_args(),
        tafd_impl,
        result_prefix="tafd_standard_convolution",
    )
    tafd_impl.main(args)


if __name__ == "__main__":
    main()
