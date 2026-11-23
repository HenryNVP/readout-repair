#!/usr/bin/env python3
"""A robomimic image HDF5 with absolute actions, for training Diffusion Policy (abs_action=True).

The actions are derived from the recorded EE pose exactly as to_lerobot.py --actions abs-derived
does for ACT (data.derive_absolute: [pos, axis-angle, gripper], 7-dim), so both policies train on
the same targets; DP's dataset turns the axis-angle into rot_6d itself.

    python scripts/dp_abs_hdf5.py --dataset outputs/interp/toolhang/tool_hang_84.hdf5 \
        --out outputs/interp/toolhang/tool_hang_84_abs.hdf5
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from readout_repair.data import STATE_KEYS, derive_absolute, osc_scales

KEYS = ('agentview_image', 'robot0_eye_in_hand_image', 'robot0_eef_pos', 'robot0_eef_quat',
        'robot0_gripper_qpos')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--dataset', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    import h5py
    with h5py.File(args.dataset, 'r') as src, h5py.File(args.out, 'w') as dst:
        env_meta = json.loads(src['data'].attrs['env_args'])
        pos_scale, rot_scale = osc_scales(env_meta)
        out = dst.create_group('data')
        for k, v in src['data'].attrs.items():
            out.attrs[k] = v
        for demo in src['data']:
            ep = src['data'][demo]
            act = derive_absolute(ep['actions'][:].astype(np.float64), ep['obs'][STATE_KEYS[0]][:],
                                  ep['obs'][STATE_KEYS[1]][:], pos_scale, rot_scale)
            assert np.isfinite(act).all() and act.shape == ep['actions'].shape, demo
            g = out.create_group(demo)
            for k, v in ep.attrs.items():
                g.attrs[k] = v
            g.create_dataset('actions', data=act)
            for k in KEYS:
                g.create_dataset(f'obs/{k}', data=ep['obs'][k][:])
    print('wrote', args.out)


if __name__ == '__main__':
    main()
