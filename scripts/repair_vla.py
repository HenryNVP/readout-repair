#!/usr/bin/env python3
"""Retrain SmolVLA's readout: action_out_proj, the layer that turns the action expert's output
into the flow-matching velocity (--site expert also trains the expert), with its own loss.

Data: recorded decisions (collect.py --backend smolvla --save-images with --selflabel cf or
--teacher): what the policy saw and the label chunks. Saved as a pretrained_model dir.

    python scripts/repair_vla.py --parent <smolvla pretrained_model> \
        --data 'outputs/interp/nuis/vla_cf1_bothflip,outputs/interp/nuis/vla_cf1_none' --out ...
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from readout_repair.data import episode_files
from readout_repair.repair import save_pretrained, train_vla


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--parent', required=True)
    p.add_argument('--data', required=True, help='comma-separated globs of recorded rollout dirs')
    p.add_argument('--out', required=True)
    p.add_argument('--site', choices=('out', 'expert'), default='out')
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    import torch

    from evh_controller.policy import make_policy
    torch.manual_seed(args.seed)
    policy = make_policy('smolvla', args.parent)
    eps = [np.load(f) for f in episode_files(args.data)]
    agent, wrist = (np.concatenate([e[k] for e in eps]) for k in ('img_agent', 'img_wrist'))
    prop, lab = (np.concatenate([e[k] for e in eps]).astype(np.float32) for k in ('proprio', 'teacher'))
    n, loss = train_vla(policy, agent, wrist, prop, lab, site=args.site, steps=args.steps,
                        batch=args.batch, lr=args.lr, seed=args.seed)
    stats = dict(vars(args), decisions=len(lab), params=n, final_loss=loss)
    print(json.dumps(stats))
    save_pretrained(policy._model, args.parent, args.out, stats)


if __name__ == '__main__':
    main()
