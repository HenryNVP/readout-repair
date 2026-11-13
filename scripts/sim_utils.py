"""Small robosuite helpers the experiments share: fresh observations after editing the sim by hand,
moving the task object, one seeded chunk prediction, and a bitwise snapshot of the sim state."""
from __future__ import annotations

import numpy as np
from sil_step import stack


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
    """One seeded chunk from a history of sil_step.frames tuples."""
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
