"""Ablation: remove threat-domain diagnosis and use uniform expert aggregation."""

import main_train_pgdtrain_woDomainUniformMix as tafd_impl
from tafd_cli import build_parser, prepare_args


def main() -> None:
    parser = build_parser(
        "Train the TaFD ablation without threat-domain diagnosis.",
        include_mvit_fdconv=True,
    )
    parser.set_defaults(domain_loss_weight=0.0)
    parsed_args = parser.parse_args()
    parsed_args.ablate_domain_route = "uniform"
    parsed_args.ablate_domain_route_id = 0
    parsed_args.domain_loss_weight = 0.0
    args = prepare_args(
        parsed_args,
        tafd_impl,
        result_prefix="tafd_no_threat_domain_diagnosis",
    )
    tafd_impl.main(args)


if __name__ == "__main__":
    main()
