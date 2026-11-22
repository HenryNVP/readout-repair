#!/usr/bin/env python3
"""Retrain Diffusion Policy's readout: the FiLM conditioning layers through which the observation
features (global_cond) enter the denoiser, everything else frozen, with DP's own denoising loss.

Data: recorded decisions (collect.py --backend dp --selflabel cf): 'feats' at each labelled tick
and 'teacher', the unedited DP's full-horizon chunk on the perturbation-free render. Saved as a
workspace .ckpt holding only the edited weights (as 'ema_model'), which --weights loads.

    python scripts/repair_dp.py --parent checkpoints/dp_lift_ph_image_cnn.ckpt \
        --data 'outputs/interp/nuis/dp_cf1_bothcam,outputs/interp/nuis/dp_cf1_none' \
        --out outputs/interp/edit/dp_rep_bothcam.ckpt
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from readout_repair.data import episode_files
from readout_repair.repair import train_dp


def load(patterns: str):
    """(global_cond (N, D), label trajectories abs10 (N, H, 10)) over every labelled tick."""
    C, A = [], []
    for f in episode_files(patterns):
        e = np.load(f)
        if 'label_tick' in e:                              # labelled every n-th tick only
            C.append(e['feats'][e['label_tick']])
            A.append(e['teacher'])
            continue
        n = min(len(e['feats']), len(e['teacher']))
        C.append(e['feats'][:n])
        A.append(e['teacher'][:n])
    return np.concatenate(C).astype(np.float32), np.concatenate(A).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--parent', required=True, help='DP workspace .ckpt')
    p.add_argument('--data', required=True, help='comma-separated globs of recorded rollout dirs')
    p.add_argument('--out', required=True, help='edited .ckpt')
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    import dill
    import torch

    from evh_controller.policy import make_policy
    torch.manual_seed(args.seed)
    dp = make_policy('dp', args.parent)._policy
    cond, traj = load(args.data)
    n, loss = train_dp(dp, cond, traj, steps=args.steps, batch=args.batch, lr=args.lr, seed=args.seed)
    payload = torch.load(open(args.parent, 'rb'), pickle_module=dill, map_location='cpu')
    # the edited weights only (load_dp_checkpoint reads ema_model): the workspace payload's
    # optimizer and second model copy would make each file 3.2 GB
    payload['state_dicts'] = {'ema_model': {k: v.detach().cpu() for k, v in dp.state_dict().items()}}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(payload, open(args.out, 'wb'), pickle_module=dill)
    stats = dict(vars(args), decisions=len(cond), params=n, final_loss=loss)
    with open(args.out + '.json', 'w') as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats))


if __name__ == '__main__':
    main()
