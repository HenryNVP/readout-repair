#!/usr/bin/env python3
"""Train Diffusion Policy with its authors' code (external/diffusion_policy/train.py, hydra).

Two shims make the pinned DP commit train under this venv: scripts/dp_shim (a pytorch3d
stand-in for the four rotation functions DP uses, which has no wheel for this torch, and a stub
env runner: evaluation is collect.py's job) and diffusers' lr-scheduler module, which no longer
re-exports typing's Union / Optional. Arguments go to DP's train.py unchanged.

    python scripts/train_dp.py --config-dir=pipeline --config-name=dp_toolhang_84 \
        hydra.run.dir=$PWD/outputs/interp/toolhang/dp/run task.dataset_path=... task.dataset.dataset_path=...
"""
import os
import runpy
import sys
import typing

from readout_repair import REPO

DP = os.path.join(REPO, 'external', 'diffusion_policy')


def main() -> None:
    sys.path[:0] = [os.path.join(REPO, 'scripts', 'dp_shim'), DP]
    os.environ['PYTHONPATH'] = os.pathsep.join([os.path.join(REPO, 'scripts', 'dp_shim'), DP,
                                                os.environ.get('PYTHONPATH', '')])   # dataloader workers
    import diffusers.optimization as o
    o.Union, o.Optional = typing.Union, typing.Optional
    sys.argv = ['train.py'] + sys.argv[1:]
    os.chdir(DP)
    runpy.run_path(os.path.join(DP, 'train.py'), run_name='__main__')


if __name__ == '__main__':
    main()
