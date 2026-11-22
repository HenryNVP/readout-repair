#!/usr/bin/env python3
"""Figure 2: the perturbations on one Lift scene, agentview and wrist, original first.

    python scripts/fig_shifts.py --out figures/fig_shifts.pdf --spec 'roll=180;wroll=180' --titles 'upside down'
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from readout_repair.nuisance import Nuisance
from readout_repair.sim import TASKS, make_env, observe


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--spec', required=True, action='append', help='repeat for one column per shift')
    p.add_argument('--titles', nargs='*', help='column titles (default: the specs)')
    p.add_argument('--out', required=True, help='png/pdf: agentview and wrist, unshifted | shifted ...')
    args = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    env, _ = make_env(TASKS['lift'])
    views = [observe(env)]
    for spec in args.spec:
        n = Nuisance(spec)
        n.on(env)
        views.append(observe(env))
        n.off(env)
        restored = observe(env)
        assert np.array_equal(restored['agentview_image'], views[0]['agentview_image']), 'off() must restore'
    titles = ['original'] + (args.titles or args.spec)
    fig, axes = plt.subplots(2, len(views), figsize=(1.1 * len(views), 2.3), squeeze=False)
    for c, o in enumerate(views):
        for r, cam in enumerate(('agentview_image', 'robot0_eye_in_hand_image')):
            axes[r, c].imshow(np.flipud(o[cam]))
            axes[r, c].axis('off')
        axes[0, c].set_title(titles[c], fontsize=7)
    for r, name in enumerate(('agentview', 'wrist')):
        axes[r, 0].text(-0.08, 0.5, name, transform=axes[r, 0].transAxes, rotation=90, ha='right', va='center', fontsize=7)
    fig.subplots_adjust(wspace=0.04, hspace=0.04)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches='tight', pad_inches=0.02)
    env.close()


if __name__ == '__main__':
    main()
