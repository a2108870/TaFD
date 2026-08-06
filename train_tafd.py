"""Train or evaluate TaFD with paper-aligned terminology."""

import main_train_pgdtrain as tafd_impl
from tafd_cli import build_parser, prepare_args


def main() -> None:
    parser = build_parser(
        "Train or evaluate TaFD with threat-domain diagnosis and diagnosis-dispatch.",
        include_mvit_fdconv=True,
    )
    args = prepare_args(
        parser.parse_args(),
        tafd_impl,
        result_prefix="tafd_pgdtrain_apgdtest",
    )
    tafd_impl.main(args)


if __name__ == "__main__":
    main()
