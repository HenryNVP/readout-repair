#!/usr/bin/env python3
"""The grasp-point probe: under a perturbation, is the grasp point still linearly decodable from
the head's input?

From recorded rollouts (collect.py: ACT's 'dec', DP's 'feats', plus object and gripper poses) the
grasp point's xy relative to the gripper, or with --absolute its table position before the object
is touched, is ridge-decoded from the head input, every --stride-th tick:
  joint     one readout fit on unperturbed + perturbed episodes (5-fold CV by episode), R² on the
            perturbed rows: is the information there for a single linear map? (the paper's number)
  transfer  readout fit on the unperturbed episodes only, applied to the perturbed ones

    python scripts/probe_r2.py --base outputs/interp/nuis/cf1_none \
        --shift bothcam=outputs/interp/nuis/cf1_bothcam --absolute
"""
from __future__ import annotations

import argparse

from readout_repair.data import episode_files
from readout_repair.probes import grasp_rows, scores


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--base', required=True, help='glob of unperturbed rollout dirs')
    p.add_argument('--shift', action='append', default=[], help='name=glob of perturbed rollout dirs')
    p.add_argument('--stride', type=int, default=4, help='use every n-th tick')
    p.add_argument('--absolute', action='store_true',
                   help="the grasp point's table position before it is touched, not relative to the gripper")
    args = p.parse_args()
    base = grasp_rows(episode_files(args.base), args.stride, args.absolute)
    print(f'{"shift":14s} {"joint R² / mm":>15s} {"transfer R² / mm":>17s} {"unshifted R² / mm":>18s}  rows')
    for item in args.shift:
        name, pattern = item.split('=', 1)
        shifted = grasp_rows(episode_files(pattern), args.stride, args.absolute)
        s = scores(base, shifted)
        print(f'{name:14s} {s["joint"]:7.2f} / {s["joint_mm"]:5.1f} {s["transfer"]:9.2f} / '
              f'{s["transfer_mm"]:5.1f} {s["base"]:10.2f} / {s["base_mm"]:5.1f}  {len(shifted[0])}',
              flush=True)


if __name__ == '__main__':
    main()
