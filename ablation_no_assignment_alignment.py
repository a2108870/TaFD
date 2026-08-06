"""Ablation: remove assignment alignment in threat-domain diagnosis."""

import main_train_pgdtrain_woHungary as tafd_impl
from tafd_cli import build_parser, prepare_args


def main() -> None:
    parser = build_parser(
        "Train the TaFD ablation without assignment alignment.",
        include_mvit_fdconv=False,
    )
    args = prepare_args(
        parser.parse_args(),
        tafd_impl,
        result_prefix="tafd_no_assignment_alignment",
    )
    tafd_impl.main(args)


if __name__ == "__main__":
    main()
