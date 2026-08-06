"""Ablation: remove threat-domain diagnosis supervision."""

import main_train_pgdtrain_woDomainUniformMix as tafd_impl
from tafd_cli import build_parser, prepare_args


def main() -> None:
    parser = build_parser(
        "Train the TaFD ablation without threat-domain diagnosis supervision.",
        include_domain_route_ablation=True,
        include_mvit_fdconv=True,
    )
    args = prepare_args(
        parser.parse_args(),
        tafd_impl,
        result_prefix="tafd_no_threat_domain_diagnosis",
    )
    tafd_impl.main(args)


if __name__ == "__main__":
    main()
