#!/usr/bin/env python3
"""The paper's tables and depth figure, from the rollout dirs the pipeline/ stages write.

  --table position   Table 1: success per axis (cube ±5/±7 cm) and unshifted, per arm
  --table perturb    Table 2: base -> head retrained per policy, task and perturbation, with
                     Wilson 95% intervals and the decision rule's verdict; writes grid.json
  --table severity   Table 3: Square under weaker perturbations, dev sweep and test cells
  --table depth      Figure 3: figures/fig_depth.pdf

    python3 scripts/interp_grid.py --table all
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import wilson  # noqa: E402

O = 'outputs/interp/'
N = O + 'nuis/'
PANELS = (('ACT, Lift', N, ''), ('Diffusion Policy, Lift', N, 'dp_'), ('SmolVLA, Lift', N, 'vla_'),
          ('ACT, Square', N, 'sq_'), ('Diffusion Policy, Square', N, 'dpsq_'), ('ACT, ToolHang', N, 'th_'))
# in-distribution base success, the failure rule's reference: <prefix>eval3_base_none, except ACT
# Lift's, which is Table 1's unshifted base cell
INDIST_DIRS = {'ACT, Lift': O + 'lift_act_base_s3_shift_0,0'}
# Table 1 arms: (label, rollout name, axis or None for both); dirs are lift_<name>_s3_shift_<offset>
POSITION = (('ACT base', 'act_base', None),
            ('full retrain, 400 shifted demos', 'act_xaug', 'x'), ('', 'act_yaug', 'y'),
            ('head retrain, DAgger expert labels', 'act_edit_x_onpolicy_head_r2', 'x'),
            ('', 'act_edit_y_onpolicy_head_r2', 'y'),
            ('head retrain, pseudo-labels, no demos', 'act_edit_slfull_b3', None),
            ('head retrain, in-distribution labels', 'act_edit_x_head_r10_b1_unsh', None),
            ('SmolVLA base', 'vla_base', None),
            ('head retrain, DAgger expert labels', 'vla_rep_xpos', 'x'), ('', 'vla_rep_ypos', 'y'))
AXES = {'x': ('-0.07,0', '-0.05,0', '0.05,0', '0.07,0'), 'y': ('0,-0.07', '0,-0.05', '0,0.05', '0,0.07')}
# Table 3: dev sweep tags (sqbase_ / dpsqbase_, seed 0) and the cells taken to test
SEVERITY = (('none', 'none'), ('faint', 'object opacity 0.1'), ('faint03', 'object opacity 0.3'),
            ('faint05', 'object opacity 0.5'), ('wroll90', 'wrist roll 90 only'),
            ('cam10w30', 'turn 10 + roll 30'), ('light05', 'light x0.5'))
SHIFTS = (('bothcam', 'turn + roll'), ('darkredwroll', 'red, dim + roll'), ('bothflip', 'upside down'),
          ('bothzoom', 'zoomed'), ('faint', 'faint object'), ('invisible', 'invisible object'))


def rate(d: str):
    """(successes, episodes) of an eval dir pooled with its extra-seed twin (eval3_ -> eval4_), or None."""
    fs = sorted(glob.glob(f'{d}/ep*.npz'))
    if 'eval3_' in d:  # without it the replace is a no-op and the dir would count twice
        fs += sorted(glob.glob(f"{d.replace('eval3_', 'eval4_')}/ep*.npz"))
    if not fs:
        return None
    return int(sum(bool(np.load(f)['success']) for f in fs)), len(fs)


def indist(panel: str, root: str, pre: str):
    for d in (INDIST_DIRS.get(panel, ''), f'{root}{pre}eval3_base_none'):
        if d and rate(d):
            return rate(d)
    return None


def collect(panels=PANELS, shifts=SHIFTS) -> list:
    rows = []
    for name, root, pre in panels:
        for tag, label in shifts:
            r = {k: rate(f'{root}{pre}eval3_{k}_{tag}') for k in ('base', 'rep')}
            r['rep_none'] = rate(f'{root}{pre}eval3_rep_{tag}_none')
            if r['base'] is None or r['rep'] is None:
                continue
            row = dict(panel=name, shift=tag, label=label)
            for k, v in r.items():
                if v is not None:
                    lo, hi = wilson(*v)
                    row[k] = dict(k=v[0], n=v[1], p=v[0] / v[1], lo=lo, hi=hi)
            rows.append(row)
    ctl = {r['panel']: r['rep'] for r in rows if r['shift'] == 'invisible'}
    ref = {name: indist(name, root, pre) for name, root, pre in panels}
    for r in rows:
        if r['shift'] != 'invisible':
            r.update(verdict(r['base'], r['rep'], ctl.get(r['panel']), ref.get(r['panel'])))
    return rows


def pct(v) -> str:
    return '  --' if v is None else f'{100 * v[0] / v[1]:4.0f}'


def pooled(dirs) -> tuple | None:
    vs = [v for v in map(rate, dirs) if v]
    return (sum(k for k, _ in vs), sum(n for _, n in vs)) if vs else None


def position() -> None:
    print(f'{"Table 1: object position, Lift (%)":44s} {"x":>4s} {"y":>4s} {"in-dist.":>8s}')
    for label, name, axis in POSITION:
        cell = {a: pooled([f'{O}lift_{name}_s3_shift_{s}' for s in AXES[a]]) for a in AXES}
        if axis:
            cell = {a: (v if a == axis else None) for a, v in cell.items()}
        print(f'{label or "":44s} {pct(cell["x"])} {pct(cell["y"])} {pct(rate(f"{O}lift_{name}_s3_shift_0,0")):>8s}'
              f'   ({name})')


def severity() -> None:
    print(f'{"Table 3: Square, weaker perturbations (%)":36s} {"ACT dev":>7s} {"test":>9s}  {"DP dev":>6s} {"test":>9s}')
    for tag, label in SEVERITY:
        cells = []
        for dev, pre in (('sqbase_', 'sq_'), ('dpsqbase_', 'dpsq_')):
            base, rep = rate(f'{N}{pre}eval3_base_{tag}'), rate(f'{N}{pre}eval3_rep_{tag}')
            test = f'{pct(base).strip()} -> {pct(rep).strip()}' if base and rep else '--'
            cells.append(f'{pct(rate(N + dev + tag)):>7s} {test:>9s}')
        print(f'{label:36s} ' + '  '.join(cells))


def perturb(json_out: str) -> None:
    rows = collect()
    for r in rows:
        print(f"{r['panel']:26s} {r['shift']:13s} base {r['base']['k']:3d}/{r['base']['n']:<3d} "
              f"repaired {r['rep']['k']:3d}/{r['rep']['n']:<3d}"
              + (f" unshifted {r['rep_none']['k']}/{r['rep_none']['n']}" if 'rep_none' in r else '')
              + ('' if 'failure' not in r else ' ' + ('head-limited' if r['head_limited'] else 'not head-limited')
                 + ('' if r['failure'] else ' (no failure)')))
    with open(json_out, 'w') as f:
        json.dump(rows, f, indent=1)


def verdict(base: dict, rep: dict, ctl: dict | None, indist: tuple | None) -> dict:
    """The paper's rule on Wilson 95% bounds: a failure if base's upper bound is below the in-distribution
    lower bound; head-limited if the retrained lower bound clears base's and the control's upper bounds."""
    failure = indist is not None and base['hi'] < wilson(*indist)[0]
    head = rep['lo'] > base['hi'] and (ctl is None or rep['lo'] > ctl['hi'])
    return dict(failure=bool(failure), head_limited=bool(head))


BLUE, SINGLE = '#0072B2', 3.4                  # perturbed colour; column width, in


def style(ax) -> None:
    """Light y-only grid, no top/right spines."""
    ax.grid(axis='y', color='0.87', lw=0.5)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2.5, pad=2)


def legend_handle(color: str, label: str, inside: bool = True):
    from matplotlib.lines import Line2D
    return Line2D([], [], ls='none', marker='o', ms=3.5, color=color, markeredgewidth=0.8,
                  markerfacecolor=color if inside else 'none', label=label)


def fig_depth(out: str, root: str = N) -> None:
    """Square, both cameras shifted: success under the shift and unshifted over DAgger rounds of
    the full refit (left), and the refit depth on the same four rounds of labels (right)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7})
    rounds = [('0', 'sq_eval3_base_bothcam', 'sq_eval3_base_none')] + [
        (str(k), f'sq_depth_full{"" if k == 1 else f"_r{k}"}_eval3_bothcam',
         f'sq_depth_full{"" if k == 1 else f"_r{k}"}_eval3_none') for k in range(1, 6)]
    depth = [('head only\n5K', 'sq_depth_head_r4'), ('LoRA r=8, all\n457K', 'sq_depth_all8_r4'),
             ('full\n51.6M', 'sq_depth_full_r4')]
    fig, (a, b) = plt.subplots(1, 2, figsize=(SINGLE * 2, 2.1), sharey=True,
                               constrained_layout=True)
    for j, (name, color, fill) in enumerate((('perturbed', BLUE, True), ('in distribution', '0.45', False))):
        pts = [(k, rate(root + (sh if j == 0 else un))) for k, sh, un in rounds]
        pts = [(int(k), v) for k, v in pts if v]
        for x, v in pts:
            p_ = v[0] / v[1]
            lo, hi = wilson(*v)
            a.errorbar(x, p_, yerr=[[max(0, p_ - lo)], [max(0, hi - p_)]], fmt='o', color=color, ms=3.5,
                       lw=0.8, capsize=1.5, markerfacecolor=color if fill else 'none', zorder=3)
        a.plot([x for x, _ in pts], [v[0] / v[1] for _, v in pts], color=color, lw=0.8, zorder=2)
        for i, (lab, n) in enumerate(depth):
            v = rate(root + n + ('_eval3_bothcam' if j == 0 else '_eval3_none'))
            if v:
                p_ = v[0] / v[1]
                lo, hi = wilson(*v)
                b.errorbar(i + (0.1 if j else -0.1), p_, yerr=[[max(0, p_ - lo)], [max(0, hi - p_)]],
                           fmt='o', color=color, ms=3.5, lw=0.8, capsize=1.5,
                           markerfacecolor=color if fill else 'none', zorder=3)
    a.set_xlabel('DAgger iteration (full fine-tuning)')
    a.set_ylabel('success')
    a.set_ylim(-0.03, 1.03)
    a.set_title('DAgger iterations')
    b.set_xticks(range(len(depth)), [d for d, _ in depth])
    b.set_xlim(-0.5, len(depth) - 0.5)
    b.set_title('fine-tuning depth, same 4 iterations')
    for ax in (a, b):
        style(ax)
    fig.legend(handles=[legend_handle(BLUE, 'perturbed'), legend_handle('0.45', 'in distribution', inside=False)],
               loc='outside lower center', ncol=2, frameon=False, fontsize=7)
    fig.savefig(os.path.join(out, 'fig_depth.pdf'))
    plt.close(fig)
    print(f'[fig] {out}/fig_depth.pdf')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--table', default='all', choices=('all', 'position', 'perturb', 'severity', 'depth'))
    p.add_argument('--json', default=O + 'grid.json')
    p.add_argument('--figs', default='figures')
    args = p.parse_args()
    if args.table in ('all', 'position'):
        position()
    if args.table in ('all', 'perturb'):
        perturb(args.json)
    if args.table in ('all', 'severity'):
        severity()
    if args.table in ('all', 'depth') and glob.glob(N + 'sq_depth_full_r4_eval3_bothcam/ep*.npz'):
        fig_depth(args.figs)


if __name__ == '__main__':
    main()
