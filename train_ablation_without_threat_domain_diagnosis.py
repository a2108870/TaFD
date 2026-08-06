"""Ablation: remove threat-domain diagnosis and use uniform expert aggregation."""

import training.without_threat_domain_diagnosis as tafd_impl
from tafd_cli import build_parser, prepare_args


def main() -> None:
    parser = build_parser(
        "Train the TaFD ablation without threat-domain diagnosis.",
    )
    parser.set_defaults(diagnosis_loss_weight=0.0)
    parsed_args = parser.parse_args()
    parsed_args.diagnosis_ablation_mode = "uniform"
    parsed_args.forced_threat_domain_index = 0
    parsed_args.diagnosis_loss_weight = 0.0
    args = prepare_args(
        parsed_args,
        tafd_impl,
        result_prefix="tafd_no_threat_domain_diagnosis",
    )
    tafd_impl.main(args)


if __name__ == "__main__":
    main()
