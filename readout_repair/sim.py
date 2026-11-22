"""Tasks and the robosuite env: one env per task built like the testbed's plant, the task object's
pose and placement, the Square nut's geometry, and bitwise sim snapshots (a counterfactual render
or probe leaves the rollout exactly where it was)."""
from __future__ import annotations

import numpy as np

from .harness import CONTROL_HZ, stack

# Square nut geometry (robosuite NutAssemblySquare)
PEG = np.array([0.23, 0.10, 0.85])           # peg1 (the square peg), world-axis aligned
HANDLE_LOCAL = np.array([0.054, 0.0, 0.0])   # handle site in the nut frame
TABLE_Z = 0.82


def yaw_of(q) -> float:
    """Heading of an (upright) body's x axis about world z; q = [x, y, z, w]."""
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return float(np.arctan2(2 * (x * y + z * w), 1 - 2 * (y * y + z * z)))


def handle_pos(nut_pos, nut_quat) -> np.ndarray:
    a = yaw_of(nut_quat)
    c, s = np.cos(a), np.sin(a)
    return np.asarray(nut_pos, float) + np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ HANDLE_LOCAL


def is_held(eef, grip, nut_pos, nut_quat) -> bool:
    width = float(np.sum(np.abs(np.asarray(grip))))
    return 0.015 < width < 0.06 and float(np.linalg.norm(handle_pos(nut_pos, nut_quat)
                                                          - np.asarray(eef))) < 0.02


def on_peg(nut_pos) -> bool:
    nut_pos = np.asarray(nut_pos)
    return bool(np.linalg.norm(nut_pos[:2] - PEG[:2]) < 0.02 and nut_pos[2] < TABLE_Z + 0.04)


# per task: env, DP checkpoint, object keys and joint, the demos' xy range of the object at
# reset (robosuite placement ranges), and the grasp point
TASKS = {
    'square': dict(env='NutAssemblySquare', ckpt='dp_square_ph_image_cnn.ckpt',
                   pos='SquareNut_pos', quat='SquareNut_quat', joint='SquareNut_joint0',
                   lo=(-0.115, 0.11), hi=(-0.11, 0.225),
                   grasp=lambda o: handle_pos(o['SquareNut_pos'], o['SquareNut_quat'])[:2]),
    'lift': dict(env='Lift', ckpt='dp_lift_ph_image_cnn.ckpt', pos='cube_pos', quat='cube_quat',
                 joint='cube_joint0', lo=(-0.03, -0.03), hi=(0.03, 0.03),
                 grasp=lambda o: np.asarray(o['cube_pos'][:2])),
    # shortcut: lo/hi unset (no position experiments on ToolHang); the tool is the object
    'toolhang': dict(env='ToolHang', ckpt='', pos='tool_pos', quat='tool_quat', joint='tool_joint0',
                     lo=(-np.inf, -np.inf), hi=(np.inf, np.inf),
                     grasp=lambda o: np.asarray(o['tool_pos'][:2])),
}


def make_env(task=TASKS['square'], seed: int = 0, deadline_s: float = 20.0, absolute: bool = True):
    """The task's env at the policy rate, seeded like the plant (scene k is the same everywhere)."""
    from evh_plant.env_factory import EnvSpec, build_env
    built = build_env(EnvSpec(env_name=task['env'], action_hz=CONTROL_HZ,
                              max_episode_s=deadline_s + 1.0, seed=seed, absolute_actions=absolute))
    return built.env, built.obs


def observe(env) -> dict:
    """Fresh observations after the sim state was changed by hand."""
    return env._get_observations(force_update=True)


def set_object(env, joint: str, xy) -> None:
    """Place a free-joint object at table position `xy`, at rest."""
    sim = env.sim
    adr = sim.model.get_joint_qpos_addr(joint)[0]
    sim.data.qpos[adr:adr + 2] = xy
    sim.data.qvel[sim.model.get_joint_qvel_addr(joint)[0]:][:6] = 0.0
    sim.forward()


def place(env, obs, shift, task=TASKS['square']):
    """The scene's object moved by shift = (dx, dy); returns the fresh observation, so the
    policy's first frame already shows it there."""
    if not shift.any():
        return obs
    set_object(env, task['joint'], np.asarray(obs[task['pos']][:2]) + shift[:2])
    return observe(env)


def state(o, task=TASKS['square']) -> dict:
    """What is logged per tick next to what the policy saw: eef, gripper and object poses (on
    Square also held / on_peg)."""
    s = {'eef_pos': o['robot0_eef_pos'], 'eef_quat': o['robot0_eef_quat'],
         'grip': o['robot0_gripper_qpos'], 'nut_pos': o[task['pos']],
         'nut_quat': o[task['quat']]}
    if task is TASKS['square']:
        s['held'] = is_held(o['robot0_eef_pos'], o['robot0_gripper_qpos'],
                            o['SquareNut_pos'], o['SquareNut_quat'])
        s['on_peg'] = on_peg(o['SquareNut_pos'])
    return s


def predict(policy, hist, seed: int) -> np.ndarray:
    """One seeded chunk from a history of harness.frames tuples."""
    import torch
    torch.manual_seed(seed)
    return np.asarray(policy.predict(stack(hist, True)))


def snapshot(d) -> dict:
    """Copies of every writable array on a MjData-like object."""
    out = {}
    for k in dir(d):
        if k.startswith('_'):
            continue
        try:
            v = getattr(d, k)
        except Exception:
            continue
        if isinstance(v, np.ndarray) and v.size and v.flags.writeable:
            out[k] = v.copy()
    return out


def restore(d, saved: dict) -> None:
    """Write a snapshot back; contact-sized (efc_*) arrays that changed size are rebuilt each step."""
    for k, v in saved.items():
        cur = getattr(d, k)
        if cur.shape == v.shape:
            cur[...] = v


class Recorder:
    """Wraps the env: keeps the latest obs, and the command sent on each step."""

    def __init__(self, env, obs) -> None:
        self.env, self.obs, self.commands = env, obs, []

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        self.commands.append(np.asarray(action, np.float32))
        out = self.env.step(action)
        self.obs = out[0]
        return out
