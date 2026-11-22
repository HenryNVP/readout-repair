#!/usr/bin/env python3
"""Retrain an ACT policy's head, or deeper, on labelled states; saved as a pretrained_model dir
(plus edit.json) that every script loads with --weights.

  --onpolicy GLOBS  closed form: the head refit on recorded decisions (collect.py --teacher /
                    --selflabel: head inputs + label chunks), change truncated to --rank
  --images GLOBS    SGD on what the policy saw (collect.py --save-images) and its labels, at
                    --site: head (rank r), all (LoRA on every linear layer) or full
  neither           SGD on Lift demo replays (scripts/augment_lift.py): --labels unshifted is the
                    in-distribution-labels control

Preservation (--pres lift): the Lift base replays' pool sources keep in-range behaviour, weighed
by --beta (SGD: L1 to the parent's outputs; closed form: change 0 on their head inputs).

    python scripts/repair_act.py --parent <pretrained_model> --out outputs/interp/edit/rep_bothcam \
        --onpolicy 'outputs/interp/nuis/cf1_bothcam,outputs/interp/nuis/cf1_none' --rank 10 --beta 3
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from readout_repair.act import EDIT_SITES
from readout_repair.data import episode_files, load_replays, recorded_frames, replay_frames, subset
from readout_repair.repair import (
    head_closed_form,
    head_inputs,
    predict_frames,
    save_pretrained,
    train_act,
)

AUG = 'outputs/interp/aug'
POOL = 160                                   # sources below this rank: edit pool; rest: validation


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--parent', required=True, help='ACT pretrained_model dir')
    p.add_argument('--out', required=True)
    p.add_argument('--onpolicy', default='', help='comma-separated globs of recorded rollout dirs')
    p.add_argument('--images', default='', help='comma-separated globs of --save-images rollout dirs')
    p.add_argument('--stride', type=int, default=2, help='--images: every n-th labelled tick')
    p.add_argument('--site', default='head', choices=sorted(EDIT_SITES) + ['all', 'full'])
    p.add_argument('--rank', type=int, default=2)
    p.add_argument('--axis', choices=('x', 'y'), default='x', help='replays: which shifted set')
    p.add_argument('--n', type=int, default=40, help='replays: edit episodes')
    p.add_argument('--labels', choices=('shifted', 'unshifted'), default='shifted')
    p.add_argument('--pres', choices=('lift', 'none'), default='lift',
                   help="preservation data: the Lift base replays, or none (another task: the "
                        "unperturbed on-policy episodes preserve)")
    p.add_argument('--beta', type=float, default=1.0, help='preservation weight')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    import torch

    from evh_controller.policy import make_policy
    torch.manual_seed(args.seed)
    act = make_policy('act', args.parent)._model.eval()
    H = act.config.chunk_size
    pres = None
    if args.pres == 'lift':
        pres = replay_frames([e for e in load_replays(f'{AUG}/lift_base.hdf5', lambda k, s: True)
                              if e['rank'] < POOL], H)

    if args.onpolicy:
        if pres is not None:
            keep_idx = np.random.default_rng(args.seed).choice(len(pres[1]), min(2000, len(pres[1])),
                                                               replace=False)
            keep = head_inputs(act, subset(pres, keep_idx)).reshape(-1, act.model.action_head.in_features)
        else:
            keep = np.zeros((0, act.model.action_head.in_features))
        eps = [np.load(f) for f in episode_files(args.onpolicy)]
        stats = head_closed_form(act, eps, keep, args.beta, args.rank, args.seed)
    else:
        if args.images:
            edit = recorded_frames(args.images, H, args.stride)
        elif args.labels == 'shifted':
            edit = replay_frames(load_replays(f'{AUG}/lift_{args.axis}.hdf5',
                                              lambda k, s: k < args.n // 2 and s.any()), H)
        else:
            edit = replay_frames(load_replays(f'{AUG}/lift_{args.axis}.hdf5',
                                              lambda k, s: k < args.n and not s.any()), H)
        parent_pres = predict_frames(act, pres) if pres is not None else None
        n = train_act(act, args.site, args.rank, edit, pres, parent_pres, beta=args.beta,
                      steps=args.steps, batch=args.batch, lr=args.lr, seed=args.seed)
        stats = dict(params=n, edit_frames=len(edit[1]))
    stats = dict(vars(args), **stats)
    print(json.dumps(stats))
    save_pretrained(act, args.parent, args.out, stats)


if __name__ == '__main__':
    main()
