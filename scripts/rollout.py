"""Closed-loop rollouts of a chunk policy on robosuite, and the small sim helpers the experiments share.

The policy runs at 20 Hz, the rate it was trained at, with the OSC stepped once per tick. Each
chunk is executed to its end; the next one is requested from the observation of the tick that
sends the last action and is first usable on the following tick. Tick 0 has no chunk yet, so it
sends the hold action (the current pose in absolute mode, zeros in delta mode).

    run_episode(env, obs, policy, deadline_ticks=..., absolute=..., stack_fn=...) -> Episode
"""
from __future__ import annotations

import collections
import dataclasses
import os

import numpy as np

from env_factory import hold_action

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROL_HZ = 20.0


def frames(obs: dict, needs_wrist: bool = True):
    """robosuite obs -> (agentview, wrist, proprio): images upright, proprio
    [eef_pos(3), eef_quat(4), gripper_qpos(2)]."""
    agent = np.flipud(np.asarray(obs['agentview_image'])).astype(np.uint8)
    wrist = (np.flipud(np.asarray(obs['robot0_eye_in_hand_image'])).astype(np.uint8)
             if needs_wrist else None)
    proprio = np.concatenate([np.asarray(obs['robot0_eef_pos'], np.float32),
                              np.asarray(obs['robot0_eef_quat'], np.float32),
                              np.asarray(obs['robot0_gripper_qpos'], np.float32)])
    return agent, wrist, proprio


def stack(hist, needs_wrist: bool) -> dict:
    """A history of `frames` tuples, oldest first -> the policy's observation dict."""
    out = {'agentview': np.stack([h[0] for h in hist]),
           'proprio': np.stack([h[2] for h in hist])}
    if needs_wrist:
        out['wrist'] = np.stack([h[1] for h in hist])
    return out


@dataclasses.dataclass
class Episode:
    success: bool
    ticks: int            # control ticks elapsed when it ended (success tick + 1, or the deadline)


def run_episode(env, obs, policy, *, deadline_ticks: int, absolute: bool = True,
                stack_fn=stack) -> Episode:
    """One episode from an already-reset env whose first observation is `obs`.

    `stack_fn(hist, needs_wrist)` builds the policy input every tick (callers hook it to record
    what the policy saw). A chunk is requested on the tick its predecessor runs out, even when
    the episode then ends, so a stochastic policy draws the same random numbers either way.
    """
    needs_wrist = bool(getattr(policy, 'needs_wrist', True))
    hist = collections.deque(maxlen=policy.n_obs_steps)
    first = frames(obs, needs_wrist)
    for _ in range(policy.n_obs_steps):     # pad the history with the first frame, as at reset
        hist.append(first)
    chunk, i, ready = None, 0, None
    for t in range(deadline_ticks):
        o = stack_fn(hist, needs_wrist)
        if ready is not None:               # requested last tick
            chunk, i, ready = ready, 0, None
        if chunk is None:                   # tick 0: request, and hold meanwhile
            ready = np.asarray(policy.predict(o))
            command = hold_action(obs, absolute, policy.action_dim)
        else:
            command = np.asarray(chunk[i], dtype=np.float32)
            i += 1
            if i >= len(chunk):             # the last action goes out: request the next chunk
                chunk, ready = None, np.asarray(policy.predict(o))
        obs, _r, _done, _info = env.step(command)
        hist.append(frames(obs, needs_wrist))
        if env._check_success():
            return Episode(True, t + 1)
    return Episode(False, deadline_ticks)


# ------------------------------------------------------------------ sim helpers
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


def predict(policy, hist, seed: int) -> np.ndarray:
    """One seeded chunk from a history of `frames` tuples."""
    import torch
    torch.manual_seed(seed)
    return np.asarray(policy.predict(stack(hist, True)))


def data_arrays(d) -> dict:
    """Copies of every writable array on a MjData-like object (to restore the sim state)."""
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
