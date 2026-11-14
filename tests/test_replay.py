"""Re-running a cell reproduces it bitwise: the claim behind "re-run any cell from the released
checkpoints". Opt-in (GPU, a few minutes): REPLAY=1 python -m pytest tests/test_replay.py

Each case runs the first episodes of a paper cell twice, with the pipeline's own arguments
(pipeline/common.sh), and requires identical commands. DP and SmolVLA need interp_collect's per-scene
torch seed and deterministic kernels for this; ACT draws no noise.

Runs recorded before 2026-11-14 (Docker image, no per-scene seed) are a different sample of the
same distribution, not a bitwise match: the venv's scipy differs in the last LAPACK bits, and DP and
SmolVLA noise was unseeded. They are reproduced in rate, not episode by episode.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
O = os.path.join(REPO, 'outputs/interp')
EPISODES = 2
CASES = {   # paper cell: (task, backend, weights, seed, extra args)
    'eval3_base_bothcam': ('lift', 'act', f'{O}/aug/train/act_lift_aug_base/checkpoints/last/pretrained_model',
                           3, ['--nuisance', 'cam=0,0,0,30;wroll=90']),
    'dp_eval3_rep_faint': ('lift', 'dp', f'{O}/edit/dp_rep_faint.ckpt', 3, ['--nuisance', 'objalpha=0.1']),
    'vla_eval3_base_bothflip': ('lift', 'smolvla', f'{O}/vla/smolvla_lift/checkpoints/last/pretrained_model',
                                3, ['--nuisance', 'roll=180;wroll=180']),
}


@pytest.mark.skipif(not os.environ.get('REPLAY'), reason='opt-in: REPLAY=1 (GPU, minutes)')
@pytest.mark.parametrize('name', sorted(CASES))
def test_a_rerun_is_bitwise_identical(name, tmp_path):
    task, backend, weights, seed, extra = CASES[name]
    if not os.path.exists(weights):
        pytest.skip(f'weights missing: {weights}')
    for run in ('a', 'b'):
        subprocess.run([sys.executable, os.path.join(REPO, 'scripts/interp_collect.py'), '--task', task,
                        '--backend', backend, '--weights', weights, '--seed', str(seed),
                        '--episodes', str(EPISODES), '--deadline-s', '20', '--out-dir', str(tmp_path / run),
                        *extra], check=True, cwd=REPO, env={**os.environ, 'MUJOCO_GL': 'egl'},
                       stdout=subprocess.DEVNULL)
    for k in range(EPISODES):
        a, b = (np.load(tmp_path / run / f'ep{k:04d}.npz') for run in ('a', 'b'))
        assert int(a['ticks']) == int(b['ticks']) and bool(a['success']) == bool(b['success']), f'ep{k}'
        np.testing.assert_array_equal(a['commands'], b['commands'], err_msg=f'ep{k}')
