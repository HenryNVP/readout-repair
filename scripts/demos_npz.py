#!/usr/bin/env python3
"""Write robomimic image demos in the recorded-rollout layout repair_act.py --images reads
(img_agent, img_wrist, proprio, teacher = the next 16 absolute actions, abs10), so a repair can mix
the original demonstrations in, as standard fine-tuning does, and keep in-range behaviour.

Actions are converted exactly as to_lerobot.py --actions abs-derived does; chunks past the
episode end repeat the last action.

    python scripts/demos_npz.py --dataset .../square/ph/image.hdf5 --out outputs/interp/nuis/sq_demos
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from readout_repair.data import STATE_KEYS, chunks, derive_absolute, osc_scales, sorted_demos


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--dataset', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max-demos', type=int, default=0)
    p.add_argument('--chunk', type=int, default=16)
    args = p.parse_args()
    import h5py

    from evh_controller.rotation import abs7_to_abs10
    os.makedirs(args.out, exist_ok=True)
    with h5py.File(args.dataset, 'r') as f:
        data = f['data']
        pos_scale, rot_scale = osc_scales(json.loads(data.attrs['env_args']))
        for i, name in enumerate(sorted_demos(data)[:args.max_demos or None]):
            ep = data[name]
            o = ep['obs']
            abs7 = derive_absolute(ep['actions'][:].astype(np.float64), o[STATE_KEYS[0]][:],
                                   o[STATE_KEYS[1]][:], pos_scale, rot_scale)
            np.savez_compressed(
                os.path.join(args.out, f'ep{i:04d}.npz'),
                img_agent=o['agentview_image'][:], img_wrist=o['robot0_eye_in_hand_image'][:],
                proprio=np.concatenate([o[k][:] for k in STATE_KEYS], 1).astype(np.float32),
                teacher=chunks(abs7_to_abs10(abs7).astype(np.float32), args.chunk))
    print(f'wrote {i + 1} demos -> {args.out}')


if __name__ == '__main__':
    main()
