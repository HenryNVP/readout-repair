"""Data the repairs and probes read: recorded rollouts (scripts/collect.py, one ep*.npz per
episode) and robomimic demos (HDF5), with the absolute-action conversion the ACT datasets use."""
from __future__ import annotations

import glob

import numpy as np

# the testbed modules (on sys.path once readout_repair/__init__ has run)
from evh_controller.rotation import abs7_to_abs10
from evh_reactive.transforms import axisangle_to_quat, quat_mul, quat_to_axisangle

STATE_KEYS = ('robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos')
CAMS = ('agentview_image', 'robot0_eye_in_hand_image')     # scene camera first, wrist last

# robosuite's OSC does not control the frame `robot0_eef_quat` reports: its tool frame is that
# one rotated -90 deg about z (measured exactly, and constant: `R_ctrl = R_obs @ Rz(-pi/2)`,
# positions identical). Absolute orientation targets are consumed by the controller, so they must
# be expressed in its frame; deltas are immune. Skipping this is silent: an ACT trained on
# uncorrected targets scored 0/10 on Lift against 7/10 for the same demos in delta form.
EEF_TO_CONTROL_QUAT = np.array([0.0, 0.0, -np.sin(np.pi / 4), np.cos(np.pi / 4)])


# ------------------------------------------------------------------ recorded rollouts
def episode_files(patterns: str) -> list[str]:
    """Every ep*.npz under comma-separated globs of rollout dirs, sorted by path."""
    return sorted(f for g in patterns.split(',') for d in sorted(glob.glob(g))
                  for f in glob.glob(f'{d}/ep*.npz'))


def recorded_frames(patterns: str, H: int, stride: int = 1):
    """replay_frames' tuple from recorded rollouts (collect.py --save-images): what the policy saw
    and the label chunk (abs10) at every stride-th labelled tick; no padding (each label is a
    whole chunk)."""
    img, state, chunk, epi = [[], []], [], [], []
    for e_i, f in enumerate(episode_files(patterns)):
        e = np.load(f)
        idx = np.arange(0, len(e['teacher']), stride)
        img[0].append(e['img_agent'][idx])
        img[1].append(e['img_wrist'][idx])
        state.append(e['proprio'][idx])
        chunk.append(e['teacher'][idx][:, :H].astype(np.float32))
        epi.append(np.full(len(idx), e_i))
    chunk = np.concatenate(chunk)
    return ([np.concatenate(i) for i in img], np.concatenate(state), chunk,
            np.zeros(chunk.shape[:2], bool), np.concatenate(epi))


def subset(data, idx):
    """The rows idx of a replay_frames / recorded_frames tuple."""
    return [i[idx] for i in data[0]], *(x[idx] for x in data[1:])


# ------------------------------------------------------------------ robomimic demos
def sorted_demos(data: dict) -> list[str]:
    """Demo keys in numeric order (h5py yields them alphabetically: demo_10 before demo_2)."""
    return sorted(data.keys(), key=lambda name: int(name.split('_')[1]))


def osc_scales(env_meta: dict) -> tuple[np.ndarray, np.ndarray]:
    """OSC output_max, i.e. how far a unit action moves the goal (position, rotation)."""
    cfg = env_meta['env_kwargs'].get('controller_configs') or {}
    output_max = cfg.get('output_max')
    if output_max is None:                      # robosuite 1.5 nests it per body part
        for part in cfg.get('body_parts', {}).values():
            if isinstance(part, dict) and str(part.get('type', '')).startswith('OSC'):
                output_max = part.get('output_max')
    if output_max is None:
        raise SystemExit('dataset controller config has no output_max — cannot derive absolute '
                         'actions; pass --actions delta, or use a dataset with actions_abs')
    output_max = np.asarray(output_max, dtype=np.float64)
    return output_max[:3], output_max[3:6]


def derive_absolute(actions: np.ndarray, eef_pos: np.ndarray, eef_quat: np.ndarray,
                    pos_scale: np.ndarray, rot_scale: np.ndarray) -> np.ndarray:
    """Delta OSC actions -> absolute EE targets (abs7), using the pose each action was issued from.

    Exact under robosuite's delta OSC, which sets its goal from the current pose every step:
    `p + output_max[:3] * a[:3]` and `R(output_max[3:] * a[3:6]) @ R_current`, the rotation composed
    in the controller's tool frame (EEF_TO_CONTROL_QUAT).
    """
    clipped = np.clip(actions[:, :6], -1.0, 1.0)   # OSC clips to input_min/input_max first
    out = np.empty_like(actions)
    out[:, :3] = eef_pos + clipped[:, :3] * pos_scale
    for t in range(len(actions)):
        current = quat_mul(eef_quat[t], EEF_TO_CONTROL_QUAT)   # into the controller's tool frame
        goal = quat_mul(axisangle_to_quat(clipped[t, 3:6] * rot_scale), current)
        out[t, 3:6] = quat_to_axisangle(goal)
    out[:, 6:] = actions[:, 6:]                    # gripper passes through
    return out


def chunks(actions: np.ndarray, H: int) -> np.ndarray:
    """(T, D) -> (T, H, D): actions t..t+H-1, the last one repeated past the end."""
    T = len(actions)
    return actions[np.minimum(np.arange(T)[:, None] + np.arange(H)[None], T - 1)]


def load_replays(path: str, pick) -> list[dict]:
    """Episodes of a scripts/augment_lift.py HDF5 for which pick(source rank, shift) is true:
    images, state and abs10 actions exactly as to_lerobot.py builds the training set."""
    import h5py
    eps = []
    with h5py.File(path, 'r') as f:
        data = f['data']
        names = sorted_demos(data)
        sources = sorted({str(data[n].attrs['source']) for n in names},
                         key=lambda s: int(s.split('_')[1]))
        rank = {s: i for i, s in enumerate(sources)}
        for n in names:
            ep = data[n]
            shift = np.asarray(ep.attrs['shift'])
            if pick(rank[str(ep.attrs['source'])], shift):
                eps.append(dict(img=[ep['obs'][c][:] for c in CAMS],
                                state=np.concatenate([ep['obs'][k][:] for k in STATE_KEYS], 1)
                                .astype(np.float32),
                                act=abs7_to_abs10(ep['actions_abs'][:].astype(np.float32)),
                                shift=shift, rank=rank[str(ep.attrs['source'])]))
    return eps


def replay_frames(eps: list[dict], H: int):
    """All (images per camera, state, chunk, pad mask, episode) frames; chunks padded with the
    last action."""
    img, state, chunk, pad, epi = [[], []], [], [], [], []
    for e, ep in enumerate(eps):
        T = len(ep['act'])
        for t in range(T):
            epi.append(e)
            j = np.minimum(np.arange(t, t + H), T - 1)
            for c in range(2):
                img[c].append(ep['img'][c][t])
            state.append(ep['state'][t])
            chunk.append(ep['act'][j])
            pad.append(np.arange(t, t + H) >= T)
    return [np.stack(i) for i in img], np.stack(state), np.stack(chunk), np.stack(pad), np.array(epi)
