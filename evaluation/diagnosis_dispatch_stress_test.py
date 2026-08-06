"""Stress-test TaFD's diagnosis-dispatch mechanism with targeted diagnosis losses.

The evaluation compares each standard attack with a per-sample union that also
targets every incorrect threat domain. The model always uses its original hard
diagnosis-dispatch forward pass; no route is assigned externally.
"""

import argparse
import csv
import os
import traceback

import matplotlib
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evaluation.adaptive_diagnosis import generate_adaptive_diagnosis_batch
from torchattacks.attacks.apgd_adaptive_diagnosis import AdaptiveDiagnosisAPGD
from training.protocols import ATTACK_UNIONS


ATTACKS_BY_UNION = {
    union_name: [
        attack_name
        for attack_name in union_config['test_attacks']
        if attack_name != 'Clean'
    ]
    for union_name, union_config in ATTACK_UNIONS.items()
}


@torch.no_grad()
def _predict_classes_and_domains(model, inputs):
    if hasattr(model, 'set_bpda'):
        model.set_bpda(True)
    outputs = model(inputs, attack_source_ids=None)
    predicted_classes = outputs[0].argmax(1)
    predicted_threat_domains = None
    if isinstance(outputs, tuple) and len(outputs) > 2 and outputs[2] is not None:
        predicted_threat_domains = outputs[2].argmax(1)
    return predicted_classes, predicted_threat_domains


def run_diagnosis_dispatch_stress_test(
    model,
    testloader,
    args,
    device,
    epoch,
    output_dir,
    n_batches=32,
    steps=50,
    diagnosis_loss_scales=(1.0, 5.0),
):
    """Evaluate robustness when attacks also target incorrect threat domains."""
    was_training = model.training
    model.eval()

    attack_union = getattr(args, 'attack_union', 'broader')
    attacks = [
        name
        for name in ATTACKS_BY_UNION.get(attack_union, ATTACKS_BY_UNION['broader'])
        if name in args.attack_names
    ]
    num_classes = args.num_classes
    num_threat_domains = int(model.num_threat_domains)
    attack_step_scale = 0.1 if 'tiny' in str(args.dataset).lower() else 1.0
    gpgd_bases = getattr(args, 'gpgd_bases', None)

    representative_source_by_domain = {}
    for source_id in range(len(args.attack_names)):
        domain_id = int(
            model.get_threat_domain_indices(torch.tensor([source_id], device=device)).item()
        )
        representative_source_by_domain.setdefault(domain_id, source_id)

    input_batches = []
    label_batches = []
    target_samples = n_batches * testloader.batch_size
    for inputs, labels in testloader:
        input_batches.append(inputs)
        label_batches.append(labels)
        if sum(batch.size(0) for batch in input_batches) >= target_samples:
            break

    batch_size = testloader.batch_size
    inputs = torch.cat(input_batches, 0)[:target_samples].to(device)
    labels = torch.cat(label_batches, 0)[:target_samples].to(device)
    num_samples = inputs.size(0)

    def make_apgd(norm, eps, diagnosis_loss_scale):
        return AdaptiveDiagnosisAPGD(
            model,
            norm=norm,
            eps=eps,
            steps=steps,
            n_restarts=1,
            seed=0,
            loss='ce',
            eot_iter=1,
            rho=0.75,
            verbose=False,
            diagnosis_loss_scale=diagnosis_loss_scale,
        )

    apgd_attacks = {
        ('Linf', 0.0): make_apgd('Linf', 8 / 255, 0.0),
        ('L2', 0.0): make_apgd('L2', 0.5, 0.0),
    }
    for scale in diagnosis_loss_scales:
        apgd_attacks[('Linf', scale)] = make_apgd('Linf', 8 / 255, -scale)
        apgd_attacks[('L2', scale)] = make_apgd('L2', 0.5, -scale)

    def generate(inputs_batch, labels_batch, attack_name, target_source_id, scale):
        if scale == 0.0:
            source_id = args.attack_names.index(attack_name)
        else:
            source_id = target_source_id
        source_ids_by_attack = {
            attack_name: torch.full(
                (inputs_batch.size(0),),
                source_id,
                device=device,
                dtype=torch.long,
            )
        }
        effective_scale = 0.0 if scale == 0.0 else -scale
        apgd_linf = apgd_attacks[('Linf', scale)]
        apgd_l2 = apgd_attacks[('L2', scale)]
        if hasattr(model, 'set_bpda'):
            model.set_bpda(True)
        adversarial_inputs = generate_adaptive_diagnosis_batch(
            inputs_batch,
            labels_batch,
            model,
            device,
            [attack_name],
            source_ids_by_attack,
            num_classes,
            effective_scale,
            lr_scale=attack_step_scale,
            atk_apgd_linf=apgd_linf,
            atk_apgd_l2=apgd_l2,
            gpgd_bases=gpgd_bases,
        )[attack_name]
        return adversarial_inputs.detach()

    rows = []
    for attack_name in attacks:
        source_id = args.attack_names.index(attack_name)
        assigned_domain = int(
            model.get_threat_domain_indices(torch.tensor([source_id], device=device)).item()
        )
        target_domains = [
            domain_id
            for domain_id in range(num_threat_domains)
            if domain_id != assigned_domain
        ]
        baseline_failures = torch.zeros(num_samples, dtype=torch.bool, device=device)
        stress_failures = torch.zeros(num_samples, dtype=torch.bool, device=device)
        target_domain_hits = 0
        target_domain_attempts = 0

        for start in range(0, num_samples, batch_size):
            inputs_batch = inputs[start:start + batch_size]
            labels_batch = labels[start:start + batch_size]

            adversarial_inputs = generate(
                inputs_batch,
                labels_batch,
                attack_name,
                None,
                0.0,
            )
            predicted_classes, _ = _predict_classes_and_domains(model, adversarial_inputs)
            failures = predicted_classes != labels_batch
            baseline_failures[start:start + batch_size] |= failures
            stress_failures[start:start + batch_size] |= failures
            del adversarial_inputs

            for target_threat_domain_indices in target_domains:
                target_source_id = int(
                    representative_source_by_domain.get(target_threat_domain_indices, 0)
                )
                for scale in diagnosis_loss_scales:
                    adversarial_inputs = generate(
                        inputs_batch,
                        labels_batch,
                        attack_name,
                        target_source_id,
                        scale,
                    )
                    predicted_classes, predicted_threat_domains = _predict_classes_and_domains(
                        model,
                        adversarial_inputs,
                    )
                    stress_failures[start:start + batch_size] |= (
                        predicted_classes != labels_batch
                    )
                    if predicted_threat_domains is not None:
                        target_domain_hits += (
                            predicted_threat_domains == target_threat_domain_indices
                        ).sum().item()
                        target_domain_attempts += inputs_batch.size(0)
                    del adversarial_inputs
            torch.cuda.empty_cache()

        baseline_ra = 100.0 * (~baseline_failures).float().mean().item()
        stress_test_ra = 100.0 * (~stress_failures).float().mean().item()
        rows.append(
            (
                attack_name,
                baseline_ra,
                stress_test_ra,
                baseline_ra - stress_test_ra,
                100.0 * target_domain_hits / max(1, target_domain_attempts),
            )
        )

    _save_results(
        output_dir,
        epoch,
        rows,
        num_threat_domains,
        attack_union,
        num_samples,
        steps,
        diagnosis_loss_scales,
    )
    if hasattr(model, 'set_bpda'):
        model.set_bpda(False)
    if was_training:
        model.train()
    return rows


def _save_results(
    output_dir,
    epoch,
    rows,
    num_threat_domains,
    attack_union,
    num_samples,
    steps,
    diagnosis_loss_scales,
):
    stress_test_dir = os.path.join(output_dir, 'diagnosis_dispatch_stress_test')
    os.makedirs(stress_test_dir, exist_ok=True)

    epoch_csv = os.path.join(stress_test_dir, f'stress_test_epoch_{epoch}.csv')
    with open(epoch_csv, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                'attack',
                'baseline_ra',
                'stress_test_ra',
                'ra_drop',
                'target_domain_success',
            ]
        )
        for row in rows:
            writer.writerow(
                [row[0], f'{row[1]:.2f}', f'{row[2]:.2f}', f'{row[3]:.2f}', f'{row[4]:.2f}']
            )

    epoch_txt = os.path.join(stress_test_dir, f'stress_test_epoch_{epoch}.txt')
    with open(epoch_txt, 'w') as handle:
        handle.write(
            'Diagnosis-dispatch stress test | '
            f'epoch={epoch} | K={num_threat_domains} | attack_union={attack_union} | '
            f'n={num_samples} | steps={steps} | '
            f'diagnosis_loss_scales={list(diagnosis_loss_scales)}\n'
        )
        handle.write(
            f'{"attack":<12}{"baseline_ra":>13}{"stress_ra":>12}'
            f'{"ra_drop":>10}{"target_success":>16}\n'
        )
        for row in rows:
            handle.write(
                f'{row[0]:<12}{row[1]:>13.2f}{row[2]:>12.2f}'
                f'{row[3]:>10.2f}{row[4]:>16.2f}\n'
            )
        mean_drop = sum(row[3] for row in rows) / max(1, len(rows))
        max_drop = max((row[3] for row in rows), default=0.0)
        handle.write(f'\nmean_ra_drop={mean_drop:.2f}  max_ra_drop={max_drop:.2f}\n')
        handle.write(
            'ra_drop = baseline_ra - stress_test_ra; lower values indicate greater '
            'resistance to diagnosis-targeted dispatch manipulation.\n'
        )

    summary_csv = os.path.join(stress_test_dir, 'stress_test_summary.csv')
    new_summary = not os.path.exists(summary_csv)
    with open(summary_csv, 'a', newline='') as handle:
        writer = csv.writer(handle)
        if new_summary:
            writer.writerow(
                [
                    'epoch',
                    'attack',
                    'baseline_ra',
                    'stress_test_ra',
                    'ra_drop',
                    'target_domain_success',
                ]
            )
        for row in rows:
            writer.writerow(
                [
                    epoch,
                    row[0],
                    f'{row[1]:.2f}',
                    f'{row[2]:.2f}',
                    f'{row[3]:.2f}',
                    f'{row[4]:.2f}',
                ]
            )

    try:
        _plot_epoch_results(
            rows,
            os.path.join(stress_test_dir, f'stress_test_epoch_{epoch}.png'),
            epoch,
            num_threat_domains,
            attack_union,
        )
        _plot_epoch_comparison(
            summary_csv,
            os.path.join(stress_test_dir, 'stress_test_across_epochs.png'),
            num_threat_domains,
            attack_union,
        )
    except Exception as error:
        print(f'[DIAGNOSIS-DISPATCH] plot generation failed: {error}')
        traceback.print_exc()

    mean_drop = sum(row[3] for row in rows) / max(1, len(rows))
    print(
        f'[DIAGNOSIS-DISPATCH] epoch {epoch}: results saved to '
        f'{stress_test_dir} | mean_ra_drop={mean_drop:.2f}',
        flush=True,
    )


def _plot_epoch_results(rows, path, epoch, num_threat_domains, attack_union):
    import numpy as np

    attack_names = [row[0] for row in rows]
    baseline_ra = [row[1] for row in rows]
    stress_test_ra = [row[2] for row in rows]
    positions = np.arange(len(attack_names))
    width = 0.38

    fig, axis = plt.subplots(figsize=(max(6, 1.3 * len(attack_names)), 4.5))
    axis.bar(
        positions - width / 2,
        baseline_ra,
        width,
        label='standard attack',
        color='#4C72B0',
    )
    axis.bar(
        positions + width / 2,
        stress_test_ra,
        width,
        label='diagnosis-dispatch stress test',
        color='#C44E52',
    )
    for index, row in enumerate(rows):
        axis.text(
            index,
            max(baseline_ra[index], stress_test_ra[index]) + 1,
            f'drop {row[3]:.1f}',
            ha='center',
            fontsize=8,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(attack_names, rotation=20)
    axis.set_ylabel('robust accuracy (%)')
    axis.set_ylim(0, max(100, max(baseline_ra + stress_test_ra) + 8))
    axis.set_title(
        f'K={num_threat_domains} {attack_union} | epoch {epoch} | '
        'diagnosis-dispatch stress test'
    )
    axis.legend(fontsize=8)
    axis.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_epoch_comparison(
    summary_csv,
    path,
    num_threat_domains,
    attack_union,
):
    import collections

    drops_by_attack = collections.defaultdict(dict)
    epochs = set()
    with open(summary_csv) as handle:
        for row in csv.DictReader(handle):
            epoch = int(row['epoch'])
            drops_by_attack[row['attack']][epoch] = float(row['ra_drop'])
            epochs.add(epoch)

    epochs = sorted(epochs)
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for attack_name, values in drops_by_attack.items():
        y_values = [values.get(epoch, float('nan')) for epoch in epochs]
        axis.plot(epochs, y_values, marker='o', label=attack_name)
    axis.set_xlabel('epoch')
    axis.set_ylabel('robust-accuracy drop (%)')
    axis.set_title(
        f'K={num_threat_domains} {attack_union} | diagnosis-dispatch stress test'
    )
    axis.axhline(0, color='gray', linestyle=':', linewidth=1)
    axis.legend(fontsize=8)
    axis.grid(linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(args):
    from evaluation.adaptive_diagnosis import (
        build_model,
        infer_domains_from_checkpoint,
        load_checkpoint_to_model,
        prepare_evaluation_gpgd_bases,
    )
    import training.tafd as training_module
    from utils.datasets_utils import GetDataLoader

    device = torch.device(
        f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    )
    attack_config = training_module.ATTACK_UNIONS[args.attack_union]
    args.num_attack_sources = attack_config['num_attack_sources']
    args.attack_names = list(attack_config['attack_names'])
    args.test_attacks = list(attack_config['test_attacks'])
    args.num_classes = training_module._infer_num_classes(args.dataset, fallback=100)
    args.num_threat_domains = (
        infer_domains_from_checkpoint(args.checkpoint, device) or 2
    )
    args.resume = args.checkpoint

    trainloader, testloader = GetDataLoader(
        args.dataset,
        args.batch_size,
        args.test_batch_size,
        args.dataset_path,
        num_workers=args.num_workers,
    )
    args.gpgd_bases = None
    prepare_evaluation_gpgd_bases(args, trainloader, device)

    model = build_model(args, device)
    model.count_frequency_convolutions()
    load_checkpoint_to_model(model, args.checkpoint, device)
    model.eval()
    run_diagnosis_dispatch_stress_test(
        model,
        testloader,
        args,
        device,
        args.epoch,
        args.output_dir,
        n_batches=args.n_batches,
        steps=args.steps,
        diagnosis_loss_scales=args.diagnosis_loss_scales,
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description='Evaluate TaFD under diagnosis-targeted dispatch manipulation.'
    )
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument(
        '--dataset',
        default='CIFAR100',
        choices=['CIFAR10', 'CIFAR100', 'Imagenette'],
    )
    parser.add_argument('--dataset_path', default='./datasets')
    parser.add_argument('--backbone', choices=['resnet', 'mobilevit'], default='resnet')
    parser.add_argument(
        '--attack_union',
        choices=['canonical', 'broader'],
        default='broader',
    )
    parser.add_argument('--n_batches', type=int, default=32)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--diagnosis_loss_scales', type=float, nargs='+', default=[1.0, 5.0])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpgd_basis_path', default='')
    parser.add_argument('--gpgd_rank', type=int, default=128)
    parser.add_argument('--gpgd_max_per_class', type=int, default=600)
    return parser


if __name__ == '__main__':
    main(build_parser().parse_args())
