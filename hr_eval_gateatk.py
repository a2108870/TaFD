# -*- coding: utf-8 -*-
"""Harmful-routing evaluation via gateatk (per-attack std vs route-manipulated).
Runs at training epochs 25/50/75 (inline, model in memory, training paused -> no GPU contention),
or standalone on a checkpoint. Outputs per-attack + harmful-routing TABLE (csv/txt) and PLOTS
(per-epoch std-vs-hr bar + cross-epoch gap trend) into <result_dir>/hr_eval/ for direct viewing.

Harmful routing definition (input-space, threat-model-faithful; NO forced routing):
  per weapon W: std = classification-only attack (gate_loss_scale=0);
  harmful = push routing toward each WRONG domain j (true_gate=source(j), gate_loss_scale=-gls),
            sweep gls; per-sample union over {std} U {(j,gls)} -> hr(woj); gap=std-hr.
  Eval uses the model's OWN hard router (domain_ids=None). gap small => K resists harmful routing.
"""
import os, csv, traceback
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from main_eval_gateatk_scales import generate_gateatk_batch
from torchattacks.attacks.apgd_gateatk import APGD_GateAtk

WEAPONS = {
    'v10': ['APGD_Linf', 'APGD_L2', 'ACE', 'Hue', 'ReColorAdv', 'Light', 'UAA'],
    'v20': ['APGD_Linf', 'APGD_L2', 'ACE', 'SUB', 'STADV'],
}


@torch.no_grad()
def _pred_route(model, x):
    if hasattr(model, 'set_bpda'):
        model.set_bpda(True)
    out = model(x, domain_ids=None)
    cls = out[0].argmax(1)
    dom = out[2].argmax(1) if (isinstance(out, tuple) and len(out) > 2 and out[2] is not None) else None
    return cls, dom


def run_hr_eval(model, testloader, args, device, epoch, out_dir,
                n_batches=32, steps=50, gls_list=(1.0, 5.0)):
    was_training = model.training
    model.eval()
    cfg = getattr(args, 'attack_config', 'v20')
    weapons = [w for w in WEAPONS.get(cfg, WEAPONS['v20']) if w in args.domain_names]
    ncls = args.n_cls
    D = int(model.num_threat_domains)
    lr_scale = 0.1 if ('tiny' in str(args.dataset).lower()) else 1.0
    sub_bases = getattr(args, 'subspace_bases', None)

    dom2src = {}
    for s in range(len(args.domain_names)):
        d = int(model.get_domain_labels(torch.tensor([s], device=device)).item())
        dom2src.setdefault(d, s)

    xs, ys = [], []
    for xb, yb in testloader:
        xs.append(xb); ys.append(yb)
        if sum(t.size(0) for t in xs) >= n_batches * testloader.batch_size:
            break
    BS = testloader.batch_size
    X = torch.cat(xs, 0)[:n_batches * BS].to(device)
    Y = torch.cat(ys, 0)[:n_batches * BS].to(device)
    N = X.size(0)

    def mk(norm, eps, g):
        return APGD_GateAtk(model, norm=norm, eps=eps, steps=steps, n_restarts=1, seed=0,
                            loss='ce', eot_iter=1, rho=.75, verbose=False, gate_loss_scale=g)
    ATK = {('Linf', 0.0): mk('Linf', 8 / 255, 0.0), ('L2', 0.0): mk('L2', 0.5, 0.0)}
    for g in gls_list:
        ATK[('Linf', g)] = mk('Linf', 8 / 255, -g)
        ATK[('L2', g)] = mk('L2', 0.5, -g)

    def gen(img, tgt, W, src_target, gls):
        if gls == 0.0:
            dids = {W: torch.full((img.size(0),), args.domain_names.index(W), device=device, dtype=torch.long)}
        else:
            dids = {W: torch.full((img.size(0),), src_target, device=device, dtype=torch.long)}
        eff = 0.0 if gls == 0.0 else -gls
        al = ATK[('Linf', gls)]; a2 = ATK[('L2', gls)]
        if hasattr(model, 'set_bpda'):
            model.set_bpda(True)
        adv = generate_gateatk_batch(img, tgt, model, device, [W], dids, ncls, eff,
                                     lr_scale=lr_scale, atk_apgd_linf=al, atk_apgd_l2=a2,
                                     subspace_bases=sub_bases)[W]
        return adv.detach()

    rows = []
    for W in weapons:
        j_nat = int(model.get_domain_labels(torch.tensor([args.domain_names.index(W)], device=device)).item())
        wrong = [j for j in range(D) if j != j_nat]
        broken_std = torch.zeros(N, dtype=torch.bool, device=device)
        broken_union = torch.zeros(N, dtype=torch.bool, device=device)
        flip_hit = 0; flip_tot = 0
        for s in range(0, N, BS):
            xb = X[s:s + BS]; yb = Y[s:s + BS]
            adv = gen(xb, yb, W, None, 0.0); p, _ = _pred_route(model, adv)
            bad = (p != yb); broken_std[s:s + BS] |= bad; broken_union[s:s + BS] |= bad; del adv
            for j in wrong:
                for g in gls_list:
                    adv = gen(xb, yb, W, int(dom2src.get(j, 0)), g); p, r = _pred_route(model, adv)
                    broken_union[s:s + BS] |= (p != yb)
                    if r is not None:
                        flip_hit += (r == j).sum().item(); flip_tot += xb.size(0)
                    del adv
            torch.cuda.empty_cache()
        std = 100.0 * (~broken_std).float().mean().item()
        hr = 100.0 * (~broken_union).float().mean().item()
        rows.append((W, std, hr, std - hr, 100.0 * flip_hit / max(1, flip_tot)))

    _save_table(out_dir, epoch, rows, D, cfg, N, steps, gls_list)
    if hasattr(model, 'set_bpda'):
        model.set_bpda(False)
    if was_training:
        model.train()
    return rows


def _save_table(out_dir, epoch, rows, D, cfg, N, steps, gls_list):
    hrd = os.path.join(out_dir, 'hr_eval'); os.makedirs(hrd, exist_ok=True)
    with open(os.path.join(hrd, f'hr_ep{epoch}.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['weapon', 'std_PA', 'hr_woj', 'gap', 'flipSucc'])
        for r in rows:
            w.writerow([r[0], f'{r[1]:.2f}', f'{r[2]:.2f}', f'{r[3]:.2f}', f'{r[4]:.2f}'])
    with open(os.path.join(hrd, f'hr_ep{epoch}.txt'), 'w') as f:
        f.write(f'Harmful-routing eval (gateatk) | epoch {epoch} | K(domains)={D} | config={cfg} | '
                f'n={N} steps={steps} gls={list(gls_list)}\n')
        f.write(f'{"weapon":<12}{"std_PA":>9}{"hr(woj)":>9}{"gap":>8}{"flipSucc":>10}\n')
        for r in rows:
            f.write(f'{r[0]:<12}{r[1]:>9.2f}{r[2]:>9.2f}{r[3]:>8.2f}{r[4]:>10.2f}\n')
        mg = sum(r[3] for r in rows) / max(1, len(rows))
        f.write(f'\nmean_gap={mg:.2f}  max_gap={max(r[3] for r in rows):.2f}\n')
        f.write('gap = std_PA - hr(woj) = extra damage from manipulating routing.\n')
        f.write(f'small gap => K={D} resists harmful routing; large gap => harmful routing is real.\n')
    summ = os.path.join(hrd, 'hr_summary.csv'); newfile = not os.path.exists(summ)
    with open(summ, 'a', newline='') as f:
        w = csv.writer(f)
        if newfile:
            w.writerow(['epoch', 'weapon', 'std_PA', 'hr_woj', 'gap', 'flipSucc'])
        for r in rows:
            w.writerow([epoch, r[0], f'{r[1]:.2f}', f'{r[2]:.2f}', f'{r[3]:.2f}', f'{r[4]:.2f}'])
    try:
        _bar_plot(rows, os.path.join(hrd, f'hr_ep{epoch}.png'), epoch, D, cfg)
        _compare_plot(summ, os.path.join(hrd, 'hr_compare_over_epochs.png'), D, cfg)
    except Exception as e:
        print(f'[HR-EVAL] plot failed: {e}'); traceback.print_exc()
    print(f'[HR-EVAL] epoch {epoch}: table+plot -> {hrd} | mean_gap='
          f'{sum(r[3] for r in rows) / max(1, len(rows)):.2f}', flush=True)


def _bar_plot(rows, path, epoch, D, cfg):
    import numpy as np
    weapons = [r[0] for r in rows]; std = [r[1] for r in rows]; hr = [r[2] for r in rows]
    x = np.arange(len(weapons)); wd = 0.38
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(weapons)), 4.5))
    ax.bar(x - wd / 2, std, wd, label='per-attack std (route not attacked)', color='#4C72B0')
    ax.bar(x + wd / 2, hr, wd, label='harmful-routing (gateatk worst-over wrong routes)', color='#C44E52')
    for i, r in enumerate(rows):
        ax.text(i, max(std[i], hr[i]) + 1, f'gap {r[3]:.1f}', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(weapons, rotation=20)
    ax.set_ylabel('robust accuracy (%)'); ax.set_ylim(0, max(100, max(std + hr) + 8))
    ax.set_title(f'K={D} {cfg} | epoch {epoch} | per-attack vs harmful-routing (gap=extra harm)')
    ax.legend(fontsize=8); ax.grid(axis='y', ls='--', alpha=.5)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _compare_plot(summ_csv, path, D, cfg):
    import collections
    data = collections.defaultdict(dict)
    epochs = set()
    with open(summ_csv) as f:
        for row in csv.DictReader(f):
            ep = int(row['epoch']); data[row['weapon']][ep] = float(row['gap']); epochs.add(ep)
    epochs = sorted(epochs)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for W, d in data.items():
        ys = [d.get(e, float('nan')) for e in epochs]
        ax.plot(epochs, ys, marker='o', label=W)
    ax.set_xlabel('epoch'); ax.set_ylabel('harmful-routing gap (std - hr, %)')
    ax.set_title(f'K={D} {cfg} | harmful-routing gap vs epoch (lower=more resistant)')
    ax.axhline(0, color='gray', ls=':', lw=1); ax.legend(fontsize=8); ax.grid(ls='--', alpha=.5)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == '__main__':
    import argparse
    from main_eval_gateatk_scales import build_model, load_checkpoint_to_model, infer_domains_from_checkpoint
    import main_train_pgdtrain as T
    from utils.datasets_utils import GetDataLoader
    p = argparse.ArgumentParser('standalone harmful-routing gateatk eval')
    p.add_argument('--ckpt', required=True); p.add_argument('--out', required=True)
    p.add_argument('--epoch', type=int, default=0)
    p.add_argument('--dataset', default='CIFAR100'); p.add_argument('--attack_config', default='v20')
    p.add_argument('--n_batches', type=int, default=32); p.add_argument('--steps', type=int, default=50)
    p.add_argument('--gpu', type=int, default=0); p.add_argument('--test_batch_size', type=int, default=16)
    a = p.parse_args()
    dev = torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    cfg = T.ATTACK_CONFIGS[a.attack_config]
    a.num_sources = cfg['num_sources']; a.domain_names = list(cfg['domain_names'])
    a.n_cls = T._infer_num_classes(a.dataset, fallback=100)
    a.backbone = 'resnet'
    a.domains = infer_domains_from_checkpoint(a.ckpt, dev) or 2
    a.subspace_basis_path = ''; a.subspace_rank = 128
    a.attacks = list(cfg['test_attacks'])
    trainloader, testloader = GetDataLoader(a.dataset, 128, a.test_batch_size, './datasets/', num_workers=4)
    a.subspace_bases = None
    if 'SUB' in a.domain_names:
        try:
            from main_eval_gateatk_scales import prepare_eval_subspace_bases
            prepare_eval_subspace_bases(a, trainloader, dev)
        except Exception as e:
            print(f'[SUB bases] {e}')
    model = build_model(a, dev); model.count_frequency_convolutions()
    load_checkpoint_to_model(model, a.ckpt, dev); model.eval()
    run_hr_eval(model, testloader, a, dev, a.epoch, a.out, n_batches=a.n_batches, steps=a.steps)
