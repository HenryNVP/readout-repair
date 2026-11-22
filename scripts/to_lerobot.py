#!/usr/bin/env python3
"""Convert a robomimic image dataset (HDF5) into a LeRobotDataset, for ACT training.

LeRobot 0.3.3 and robomimic 0.5 ship no such conversion. The schema is the policy backend's
observation contract:

    observation.state              float32 [9]  = eef_pos(3) + eef_quat(4, xyzw) + gripper_qpos(2)
    observation.images.<camera>    uint8 [H,W,3], scene camera FIRST, wrist LAST
    action                         float32 [7]  delta  [pos, axis-angle, gripper], or
                                   float32 [10] abs    [pos, rot_6d, gripper]
    alt.action                     float32 [7]  = the other action convention, when available

Load-bearing and easy to get wrong: the camera order (ACTBackend maps image_keys[0] <- agentview
and [1] <- wrist positionally); the state layout (a mismatch is padded or truncated, not refused);
and `alt.action`'s prefix (any key starting with `action` becomes a policy output).

Absolute actions, `--actions`:
  abs          read an `actions_abs` key (augment_lift.py writes one)
  abs-derived  compute the targets from the recorded EE pose (data.derive_absolute): exact under
               robosuite's delta OSC, and needs no simulator (robomimic's replay-based converter
               cannot load these robosuite-1.2 demos in the pinned 1.4.1)

Absolute rotation is written as rot_6d (a 10-dim action, the DP abs checkpoints' layout). On these
demos 100% of the absolute orientation targets sit within 0.12 rad of the pi wrap and 48% of
frames flip sign in axis-angle; an ACT trained on axis-angle targets scored 0/3 on Lift.

    python scripts/to_lerobot.py --dataset data/robomimic/square/ph/image.hdf5 \
        --out outputs/interp/square/lerobot/square_ph_abs6 --actions abs-derived
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from readout_repair.data import STATE_KEYS, abs7_to_abs10, derive_absolute, osc_scales, sorted_demos

# Human-readable instruction per robosuite env. ACT is not language-conditioned, so this only
# labels the episodes -- but a dataset that says "Lift" and nothing else is hard to read later.
TASK_INSTRUCTIONS = {
    'Lift': 'lift the cube',
    'PickPlaceCan': 'pick up the can and place it in the bin',
    'NutAssemblySquare': 'place the square nut on the peg',
    'ToolHang': 'assemble the frame and hang the tool',
    'TwoArmTransport': 'transport the payload between the arms',
}

# Written into the dataset's meta/ so the action convention is recoverable after conversion.
ACTION_MODE_FILE = 'evh_action_mode.json'


def camera_keys(obs: dict, wrist_hint: str = 'eye_in_hand') -> list[str]:
    """robomimic `*_image` obs keys, scene camera(s) first and the wrist camera last."""
    cams = [k for k in obs.keys() if k.endswith('_image')]
    return sorted(cams, key=lambda k: (wrist_hint in k, k))


def state_names(keys: tuple[str, ...], dims: list[int]) -> list[str]:
    """Per-element names, e.g. robot0_eef_pos_0 — same convention as the community ports."""
    return [f'{key}_{i}' for key, dim in zip(keys, dims) for i in range(dim)]


def action_names(action_dim: int) -> list[str]:
    """Name the action dims so a reader can tell the 10-dim abs layout from a 7-dim one."""
    if action_dim == 10:
        return ['pos_x', 'pos_y', 'pos_z', *[f'rot6d_{i}' for i in range(6)], 'gripper']
    if action_dim == 7:
        return ['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z', 'gripper']
    return [f'action_{i}' for i in range(action_dim)]


def build_features(cams: list[str], state_dim: int, state_labels: list[str],
                   action_dim: int, image_shape: tuple[int, int, int],
                   use_videos: bool, with_alt: bool, alt_dim: int = 7) -> dict:
    image_dtype = 'video' if use_videos else 'image'
    features = {
        'action': {'dtype': 'float32', 'shape': (action_dim,), 'names': action_names(action_dim)},
        'next.done': {'dtype': 'bool', 'shape': (1,), 'names': ['done']},
        'observation.state': {'dtype': 'float32', 'shape': (state_dim,), 'names': state_labels},
    }
    if with_alt:
        features['alt.action'] = {'dtype': 'float32', 'shape': (alt_dim,), 'names': None}
    for cam in cams:
        features[f'observation.images.{cam[:-len("_image")]}'] = {
            'dtype': image_dtype, 'shape': image_shape, 'names': ['height', 'width', 'channel'],
        }
    return features


def select_demos(f: dict, filter_key: str, limit: int) -> list[str]:
    """Episode list, optionally restricted to one of the file's own train/valid masks."""
    if filter_key:
        if 'mask' not in f or filter_key not in f['mask']:
            available = list(f['mask'].keys()) if 'mask' in f else []
            raise SystemExit(f'no mask/{filter_key} in dataset (available: {available})')
        demos = [name.decode() if isinstance(name, bytes) else str(name)
                 for name in f['mask'][filter_key][:]]
        demos = sorted(demos, key=lambda name: int(name.split('_')[1]))
    else:
        demos = sorted_demos(f['data'])
    return demos[:limit] if limit else demos


def convert(args: argparse.Namespace) -> Path:
    import h5py
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = Path(args.out)
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f'{out} exists (pass --overwrite to replace it)')
        shutil.rmtree(out)

    derived = args.actions == 'abs-derived'
    action_key = 'actions' if args.actions == 'delta' else 'actions_abs'
    alt_key = 'actions_abs' if args.actions == 'delta' else 'actions'

    with h5py.File(args.dataset, 'r') as f:
        data = f['data']
        env_meta = json.loads(data.attrs['env_args'])
        env_name = env_meta['env_name']
        fps = int(env_meta['env_kwargs']['control_freq'])
        demos = select_demos(f, args.filter_key, args.max_episodes)

        probe = data[demos[0]]
        pos_scale, rot_scale = osc_scales(env_meta) if derived else (None, None)
        if derived:
            action_key, alt_key = 'actions', 'actions'   # derived from the delta actions
        if action_key not in probe:
            raise SystemExit(
                f"'{action_key}' missing from {args.dataset} — run robomimic's "
                'conversion/robosuite_add_absolute_actions.py first (see this script\'s docstring)')
        with_alt = derived or alt_key in probe
        cams = camera_keys(probe['obs'])
        if not cams:
            raise SystemExit(f'{args.dataset} has no *_image observations — use an image.hdf5, '
                             'or render one with robomimic/scripts/dataset_states_to_obs.py')
        image_shape = tuple(probe['obs'][cams[0]].shape[1:])
        dims = [probe['obs'][k].shape[1] for k in STATE_KEYS]
        labels = state_names(STATE_KEYS, dims)
        raw_action_dim = probe[action_key].shape[1]
        # abs + rot_6d -> the 10-dim [pos, rot_6d, gripper] layout ACTBackend recognises
        rot6d = args.actions != 'delta' and args.rot_repr == 'rot_6d' and raw_action_dim == 7
        action_dim = 10 if rot6d else raw_action_dim
        task = args.task or TASK_INSTRUCTIONS.get(env_name, env_name)

        note = ''
        if derived:
            note = f'  [derived from deltas, output_max={pos_scale[0]:g}/{rot_scale[0]:g}]'
        if rot6d:
            note += '  [rot_6d]'
        elif not with_alt:
            note = f"  [no '{alt_key}' — alt.action omitted]"
        print(f'{args.dataset}: env={env_name} fps={fps} demos={len(demos)} '
              f'cams={cams} state={sum(dims)}d action={action_dim}d ({args.actions}){note}')

        dataset = LeRobotDataset.create(
            repo_id=args.repo_id or f'local/{out.name}',
            fps=fps,
            root=out,
            robot_type='robomimic',
            features=build_features(cams, sum(dims), labels, action_dim, image_shape,
                                    args.images == 'video', with_alt, alt_dim=raw_action_dim),
            use_videos=args.images == 'video',
            image_writer_threads=args.image_writer_threads,
        )

        for i, demo in enumerate(demos):
            ep = data[demo]
            dones = ep['dones'][:].astype(bool)
            state = np.concatenate([ep['obs'][k][:] for k in STATE_KEYS],
                                   axis=1).astype(np.float32)
            if derived:
                raw = ep['actions'][:].astype(np.float64)
                actions = derive_absolute(raw, ep['obs'][STATE_KEYS[0]][:],
                                          ep['obs'][STATE_KEYS[1]][:],
                                          pos_scale, rot_scale).astype(np.float32)
                alt = raw.astype(np.float32)
            else:
                actions = ep[action_key][:].astype(np.float32)
                alt = ep[alt_key][:].astype(np.float32) if with_alt else None
            if rot6d:
                actions = abs7_to_abs10(actions)
            images = {cam: ep['obs'][cam][:] for cam in cams}

            for t in range(len(actions)):
                frame = {
                    'action': actions[t],
                    'next.done': np.array([dones[t]]),
                    'observation.state': state[t],
                }
                if with_alt:
                    frame['alt.action'] = alt[t]
                for cam in cams:
                    frame[f'observation.images.{cam[:-len("_image")]}'] = images[cam][t]
                dataset.add_frame(frame, task=task)
            dataset.save_episode()
            if (i + 1) % 25 == 0 or i + 1 == len(demos):
                print(f'  {i + 1}/{len(demos)} episodes')

    # The action convention travels with the dataset, so the checkpoint trained on it can be
    # stamped from here rather than by hand (scripts/stamp_act_checkpoint.py).
    mode_path = out / 'meta' / ACTION_MODE_FILE
    mode_path.write_text(json.dumps({'absolute_actions': args.actions != 'delta',
                                     'actions': args.actions,
                                     'rot_repr': args.rot_repr}, indent=2) + '\n')
    print(f'wrote {dataset.meta.total_episodes} episodes / {dataset.meta.total_frames} frames '
          f'-> {out}')
    return out


def _frame_image(sample: dict, key: str) -> np.ndarray:
    """LeRobot returns CHW float in [0,1]; back to the HWC uint8 scale we compare in."""
    return (sample[key].numpy().transpose(1, 2, 0) * 255.0).round()


def self_check(root: Path, args: argparse.Namespace) -> bool:
    """Read the produced dataset back and diff it against the HDF5 it came from."""
    import h5py
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id=f'local/{root.name}', root=root, video_backend='pyav')
    derived = args.actions == 'abs-derived'
    rot6d = args.actions != 'delta' and args.rot_repr == 'rot_6d'
    action_key = 'actions' if args.actions in ('delta', 'abs-derived') else 'actions_abs'
    # `action` holds derived and/or 6D-encoded values with no counterpart in the HDF5; alt.action
    # is the untouched source column, so that is what the alignment check can compare.
    frame_key = 'alt.action' if (derived or rot6d) else 'action'
    episodes = args.check_episodes
    ok = True
    with h5py.File(args.dataset, 'r') as f:
        demos = select_demos(f, args.filter_key, args.max_episodes)   # same order convert() used
        starts, ends = ds.episode_data_index['from'], ds.episode_data_index['to']
        for ep in sorted({0, ds.meta.total_episodes // 2, ds.meta.total_episodes - 1})[:episodes]:
            src = f['data'][demos[ep]]
            i0, i1 = int(starts[ep]), int(ends[ep])
            first, last = ds[i0], ds[i1 - 1]
            n_src, n_out = src[action_key].shape[0], i1 - i0
            a_err = max(float(np.abs(first[frame_key].numpy() - src[action_key][0]).max()),
                        float(np.abs(last[frame_key].numpy() - src[action_key][n_src - 1]).max()))
            s_src = np.concatenate([src['obs'][k][0] for k in STATE_KEYS])
            s_err = float(np.abs(first['observation.state'].numpy() - s_src).max())
            cams = camera_keys(src['obs'])
            img_err = max(
                float(np.abs(_frame_image(first, f'observation.images.{c[:-len("_image")]}')
                             - src['obs'][c][0].astype(np.float32)).max()) for c in cams)
            good = n_src == n_out and a_err < 1e-5 and s_err < 1e-5 and img_err == 0
            ok &= good
            print(f'  {demos[ep]}: len {n_src}=={n_out} action {a_err:.1e} state {s_err:.1e} '
                  f'image maxdiff {img_err:.0f}  {"ok" if good else "MISMATCH"}')
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--dataset', required=True, help='robomimic image.hdf5')
    p.add_argument('--out', required=True, help='output LeRobotDataset root')
    p.add_argument('--repo-id', default='', help='dataset id in the metadata (default local/<out>)')
    p.add_argument('--actions', choices=('delta', 'abs', 'abs-derived'), default='delta',
                   help="abs reads an 'actions_abs' key; abs-derived computes the targets from "
                        'the recorded EE pose (no simulator needed) — see the module docstring')
    p.add_argument('--rot-repr', choices=('rot_6d', 'axis_angle'), default='rot_6d',
                   help='absolute rotation encoding; rot_6d makes the action 10-dim (see the docstring)')
    p.add_argument('--images', choices=('image', 'video'), default='image',
                   help='image = lossless PNG; video is ~40x smaller but visibly lossy at 84x84')
    p.add_argument('--filter-key', default='', help="dataset mask to use, e.g. train / valid")
    p.add_argument('--max-episodes', type=int, default=0)
    p.add_argument('--task', default='', help='instruction string (default: per-env sentence)')
    p.add_argument('--image-writer-threads', type=int, default=8)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--check-episodes', type=int, default=3,
                   help='episodes sampled for the post-conversion self-check (0 to skip)')
    args = p.parse_args()

    out = convert(args)

    ok = True
    if args.check_episodes:
        print('self-check against the source HDF5:')
        ok &= self_check(out, args)
    if not ok:
        sys.exit('verification FAILED')


if __name__ == '__main__':
    main()
