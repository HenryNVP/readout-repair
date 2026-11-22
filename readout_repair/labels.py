"""Labels for retraining the head on the states the policy visits, without demonstrations.

  cf_label    appearance shift: the unedited policy's own chunk on the same state rendered without
              the shift (exact; the sim supplies it for free)
  self_label  object out of range: the policy's chunk with the object at the nearest in-range
              point, extended to the real object by the policy's own local slope there
  encode      Diffusion Policy's observation features (global_cond), its head's input
"""
from __future__ import annotations

import numpy as np

from .harness import frames, stack
from .sim import observe, restore, set_object, snapshot


def encode(policy, stacked: dict) -> np.ndarray:
    """DP's global_cond for one stacked history, exactly as predict_inpaint builds it."""
    import torch
    dp = policy._policy
    with torch.no_grad():
        nobs = dp.normalizer.normalize(policy._to_policy_obs(stacked))
        To = policy.n_obs_steps
        this = {k: v[:, :To, ...].reshape(-1, *v.shape[2:]) for k, v in nobs.items()}
        return dp.obs_encoder(this).reshape(-1).cpu().numpy().astype(np.float32)


def cf_label(env, policy, nuis, backend: str = 'act') -> np.ndarray:
    """The policy's own chunk (abs10) on the current state rendered without `nuis`. DP: its full
    horizon ('action_pred', both history frames showing the shift-free render)."""
    import torch
    if nuis is not None:
        nuis.off(env)
    stacked = stack([frames(observe(env))] * policy.n_obs_steps, True)
    if nuis is not None:
        nuis.on(env)
    with torch.no_grad():
        if backend == 'dp':
            out = policy._policy.predict_action(policy._to_policy_obs(stacked))
            return out['action_pred'][0].cpu().numpy()
        if backend == 'smolvla':                            # abs10, unnormalised
            return policy._model.predict_action_chunk(policy._obs_to_batch(stacked))[0].cpu().numpy()
        from .act import forward
        return forward(policy._model, policy._obs_to_batch(stacked))[0].cpu().numpy()


def self_label(env, policy, obs, batch, task, delta: float = 0.01, full: bool = False) -> np.ndarray:
    """ACT's label-free target chunk (abs10) for the current state: with the object outside the
    demos' box, the policy's chunk with the object moved to the nearest point of the box (delta
    inside), its xy extended to the real object by the policy's own local slope there (central
    differences, ±delta per axis); inside the box, the policy's own chunk. The sim is restored
    bitwise after each counterfactual. full: rescale each axis's slope so the chunk's last row
    follows that axis fully (gain 1)."""
    import torch

    from .act import forward
    act = policy._model
    xy = np.asarray(obs[task['pos']][:2], float)
    lo, hi = np.asarray(task['lo']), np.asarray(task['hi'])
    if np.all((xy >= lo) & (xy <= hi)):
        with torch.no_grad():
            return forward(act, batch)[0].cpu().numpy()
    c = np.clip(xy, lo + delta, hi - delta)
    d = env.sim.data
    saved = snapshot(d)

    def chunk_at(p):
        set_object(env, task['joint'], p)
        b = policy._obs_to_batch(stack([frames(observe(env))], True))
        restore(d, saved)
        with torch.no_grad():
            return forward(act, b)[0].cpu().numpy()
    label = chunk_at(c)
    for k in (0, 1):
        e = np.eye(2)[k] * delta
        slope = (chunk_at(c + e)[:, :2] - chunk_at(c - e)[:, :2]) / (2 * delta)   # (16, 2)
        if full:
            slope /= np.clip(slope[-1, k], 0.2, 1.0)
        label[:, :2] += slope * (xy[k] - c[k])
    observe(env)
    return label
