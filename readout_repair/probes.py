"""Probes: is the task variable still in the features, and does the action follow it?

  operating_points / probe   freeze a rollout near the grasp point, move the object over a grid
                             and record each chunk with the same noise (common random numbers),
                             restoring the sim bitwise after each
  jacobians, directions      local gains d(response)/d(offset), d(features)/d(offset)
  block_gains                the swap test: in-range head x out-of-range feature directions
  grasp_rows / scores        the grasp-point probe on recorded rollouts (ridge, R²)
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .harness import frames
from .sim import TASKS, handle_pos, observe, predict, restore, set_object, snapshot
from .stats import group_cv, r2, ridge_fit, rmse


def target(chunk: np.ndarray) -> np.ndarray:
    """The last open-gripper target of a chunk (xy); the whole chunk's last entry if it closes
    from the start."""
    open_ = np.flatnonzero(chunk[:, 6] <= 0)
    return chunk[open_[-1] if len(open_) else -1, :2]


def jacobians(resp: np.ndarray, step: float) -> np.ndarray:
    """resp[i, j] = response (2,) at offset (grid[i], grid[j]) = (dx, dy) -> J[i, j] (2x2) on
    the interior, by central differences; columns are d/ddx, d/ddy."""
    jx = (resp[2:, 1:-1] - resp[:-2, 1:-1]) / (2 * step)
    jy = (resp[1:-1, 2:] - resp[1:-1, :-2]) / (2 * step)
    return np.stack([jx, jy], axis=-1)


def operating_points(env, obs, policy, radii, scene: int, task=TASKS['square']):
    """Roll one scene out synchronously with seeded noise and yield (obs, radius, tick) each
    time the eef first comes within a radius of the grasp point (xy), gripper still open.
    The caller may probe the sim at a yield but must leave it as it found it."""
    hist = deque([frames(obs)] * policy.n_obs_steps, maxlen=policy.n_obs_steps)
    radii = sorted(radii, reverse=True)
    chunk, i, ri = None, 0, 0
    for t in range(400):
        gap = np.linalg.norm(task['grasp'](obs) - obs['robot0_eef_pos'][:2])
        while ri < len(radii) and gap < radii[ri]:
            if chunk is not None and chunk[min(i, len(chunk) - 1), 6] > 0:
                return                               # already closing: nothing left to probe
            yield obs, radii[ri], t
            ri += 1
        if ri == len(radii):
            return
        if chunk is None or i >= len(chunk):
            chunk, i = predict(policy, hist, 10_000 + scene * 100 + t), 0
        obs, *_ = env.step(chunk[i])
        i += 1
        hist.append(frames(obs))


def probe(env, policy, o, grid, seed: int, task=TASKS['square'], measure=target) -> np.ndarray:
    """Response (mm) of measure(chunk), by default the grasp target, to each object offset over
    grid x grid; the sim is restored after each, so the rollout carries on unchanged."""
    d = env.sim.data
    saved = snapshot(d)
    xy0 = np.array(o[task['pos']][:2], float)

    def chunk_at(dx, dy):
        set_object(env, task['joint'], xy0 + [dx, dy])
        cur = frames(observe(env))
        c = predict(policy, deque([cur] * policy.n_obs_steps), seed)
        restore(d, saved)
        return measure(c)

    base = chunk_at(0.0, 0.0)
    resp = np.array([[(chunk_at(dx, dy) - base) * 1e3 for dy in grid] for dx in grid])
    observe(env)
    return resp


def inside_demo_range(d: dict, r: float) -> np.ndarray:
    """(points, interior, interior) mask: is the moved object inside the demos' xy range?"""
    task = TASKS[str(d.get('task', 'square'))]
    g = d['grid'][1:-1]
    pos = d['nut_xy'][d['radius'] == r][:, None, None, :] + np.stack(
        np.meshgrid(g, g, indexing='ij'), -1)[None]
    return np.all((pos >= task['lo']) & (pos <= task['hi']), axis=-1)


def directions(F: np.ndarray, step: float) -> np.ndarray:
    """(P, G, G, D) feature changes -> (P, G-2, G-2, D, 2) d features / d mm, interior."""
    vx = (F[:, 2:, 1:-1] - F[:, :-2, 1:-1]) / (2 * step)
    vy = (F[:, 1:-1, 2:] - F[:, 1:-1, :-2]) / (2 * step)
    return np.stack([vx, vy], -1)


def block_gains(J: np.ndarray, V: np.ndarray, inside: np.ndarray) -> dict:
    """Swap test. J (P, n, n, 2, D), V (P, n, n, D, 2), inside (P, n, n) -> {(head, dir): (2,)
    median diagonal gain of J_a @ V_b over all same-point pairs, a from the head set, b from the
    dir set}. Gain lost with out-of-range directions under an in-range head = the directions."""
    out = {}
    for hn, hm in (('in', inside), ('out', ~inside)):
        for dn, dm in (('in', inside), ('out', ~inside)):
            g = [np.einsum('aod,bdo->abo', J[p][hm[p]], V[p][dm[p]]).reshape(-1, 2)
                 for p in range(len(J)) if hm[p].any() and dm[p].any()]
            out[hn, dn] = np.nanmedian(np.concatenate(g), 0)
    return out


def site_change(d: dict, s: str, m: np.ndarray) -> np.ndarray:
    """(P, G, G, D) change of site s against the unmoved middle point; decoder sites at the
    baseline's target row."""
    F = d[s][m].astype(np.float64)
    P, G = F.shape[:2]
    c = G // 2
    if s != 'mem':
        F = F[np.arange(P), :, :, d['idx'][m][:, c, c]]
    F = F.reshape(P, G, G, -1)
    return F - F[:, c:c + 1, c:c + 1]


# ------------------------------------------------------------------ grasp-point probe
def grasp_xy(e, idx) -> np.ndarray:
    """Where the gripper must go: the cube / tool position, or on Square the nut's handle (it
    swings with the nut's yaw)."""
    if str(e['task']) == 'square':
        return np.stack([handle_pos(e['nut_pos'][i], e['nut_quat'][i])[:2] for i in idx])
    return e['nut_pos'][idx, :2]


def grasp_rows(files: list[str], stride: int, absolute: bool = False):
    """(X head inputs, Y grasp xy mm, episode groups) from recorded episodes, every stride-th
    tick. Y is the grasp point minus the gripper, or with `absolute` the grasp point's table
    position on ticks before the object is touched (it has not moved yet), which the gripper's
    motion cannot give away. X is ACT's 'dec' (16 rows) or DP's 'feats'."""
    X, Y, G = [], [], []
    for g, f in enumerate(files):
        e = np.load(f)
        T = len(e['dec'] if 'dec' in e else e['feats'])
        idx = np.arange(0, T, stride)
        if absolute:
            q = e['nut_pos'][:T] - e['nut_pos'][0]   # it settles a little in z at reset
            moved = (np.linalg.norm(q[:, :2], axis=1) > 0.003) | (q[:, 2] > 0.005)
            first = np.argmax(moved) if moved.any() else T
            idx = idx[idx < first]
            if not len(idx):
                continue
        X.append((e['dec'] if 'dec' in e else e['feats'])[idx].reshape(len(idx), -1).astype(np.float32))
        xy = grasp_xy(e, idx)
        Y.append((xy if absolute else xy - e['eef_pos'][idx, :2]) * 1e3)
        G.append(np.full(len(idx), g))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(G)


def indist_score(base) -> tuple[float, float]:
    """R² and xy RMSE (mm) of the probe fit on unperturbed episodes only (5-fold CV by episode)."""
    X, Y, G = base
    pred = group_cv(X, Y, G)
    return r2(pred, Y), rmse(pred, Y)


def scores(base, shifted) -> dict:
    """R² and xy RMSE (mm) of the grasp-point probe. 'joint': one readout fit on unshifted +
    shifted rows (5-fold CV by episode), scored on the shifted rows; 'base': the same on the
    unshifted rows; 'transfer': fit on unshifted, applied to shifted."""
    Xb, Yb, Gb = base
    Xs, Ys, Gs = shifted
    X, Y, G = np.vstack([Xb, Xs]), np.vstack([Yb, Ys]), np.r_[Gb, Gs + Gb.max() + 1]
    pred = group_cv(X, Y, G)
    n = len(Xb)
    tr = ridge_fit(Xb, Yb)(Xs)
    return {'joint': r2(pred[n:], Ys), 'base': r2(pred[:n], Yb), 'transfer': r2(tr, Ys),
            'joint_mm': rmse(pred[n:], Ys), 'base_mm': rmse(pred[:n], Yb), 'transfer_mm': rmse(tr, Ys)}
