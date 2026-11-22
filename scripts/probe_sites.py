#!/usr/bin/env python3
"""Where inside ACT is an out-of-range object offset lost? Probe and swap test on Lift.

Operating points: each scene is rolled out with seeded noise until the arm is within each
--radii of the cube, the sim frozen there, the cube moved over a --grid of offsets and one chunk
computed per offset (with gradients), the sim restored after each. Recorded at three sites along
the action path: mem (encoder output, 20 tokens x 512), xattn (the decoder's cross-attention
output, 16 x 512) and dec (the head input, 16 x 512), with the Jacobian of the grasp target (the
last open-gripper entry's xy) w.r.t. each.

Report per site, inside / outside the demos' range:
  action gain  d(grasp target) / d(offset)
  seen gain    the offset ridge-decoded from the site's change (group CV by scene)
  swap         block_gains: an in-range head applied to out-of-range feature directions; the
               offset decodable at dec but no action gain = the head does not read it

    python scripts/probe_sites.py --weights <act pretrained_model> --out outputs/interp/sites_lift_act_base.npz
"""
from __future__ import annotations

import argparse

import numpy as np

from readout_repair.act import SITES, Sites, forward
from readout_repair.probes import (
    block_gains,
    directions,
    inside_demo_range,
    jacobians,
    operating_points,
    probe,
    site_change,
    target,
)
from readout_repair.sim import TASKS, make_env
from readout_repair.stats import group_cv


def collect(args) -> dict:
    import torch

    from evh_controller.policy import make_policy
    task = TASKS[args.task]
    policy = make_policy('act', args.weights)
    act = policy._model.requires_grad_(True)    # loaded frozen; grads are taken w.r.t. sites only
    sites = Sites(act)
    stash = {}
    to_batch = policy._obs_to_batch

    def grab(obs):
        stash['b'] = to_batch(obs)
        return stash['b']
    policy._obs_to_batch = grab
    env, obs = make_env(task, args.env_seed)
    grid = np.linspace(-args.extent, args.extent, args.grid)
    G = len(grid)
    f16 = lambda t: t.detach().float()[:, 0].cpu().numpy().astype(np.float16)  # noqa: E731
    out = {k: [] for k in ('mem', 'xattn', 'dec', 'J_mem', 'J_xattn', 'J_dec', 'idx', 'replica',
                           'resp', 'scene', 'radius', 'nut_xy')}
    for scene in range(args.scenes):
        for o, r, _t in operating_points(env, obs, policy, [float(x) for x in args.radii.split(',')],
                                         scene, task):
            rows = []

            def measure(chunk7):
                with torch.enable_grad():
                    chunk, i, y = forward(act, stash['b'])
                    outs = [sites.out[s] for s in SITES]
                    J = [torch.autograd.grad(y[k] * 1e3, outs, retain_graph=k == 0)   # mm
                         for k in (0, 1)]
                rows.append(dict(
                    mem=f16(outs[0]), xattn=f16(outs[1]), dec=f16(outs[2]),
                    J_mem=np.stack([f16(j[0]) for j in J]),
                    J_xattn=np.stack([f16(j[1])[i] for j in J]),
                    J_dec=np.stack([f16(j[2])[i] for j in J]), idx=i,
                    # forward() replicates policy.predict: max |xy| gap over the chunk, mm
                    replica=np.abs(chunk.detach().cpu().numpy()[:, :2] - chunk7[:, :2]).max() * 1e3))
                return target(chunk7)

            resp = probe(env, policy, o, grid, 1000 + scene, task, measure)
            for k in rows[0]:                       # rows[0] is the unmoved baseline
                out[k].append(np.stack([np.asarray(x[k]) for x in rows[1:]]).reshape(
                    G, G, *np.shape(rows[1][k])))
            out['resp'].append(resp)
            out['scene'].append(scene)
            out['radius'].append(r)
            out['nut_xy'].append(np.asarray(o[task['pos']][:2]))
        print(f'scene {scene}: {sum(s == scene for s in out["scene"])} points', flush=True)
        obs = env.reset()
    env.close()
    return {**{k: np.array(v) for k, v in out.items()}, 'grid': grid, 'task': args.task}


def report(d: dict) -> None:
    step = (d['grid'][1] - d['grid'][0]) * 1e3
    g = d['grid'] * 1e3
    G = len(g)
    print(f'replica vs policy.predict, max |xy| over the chunk: median '
          f'{np.median(d["replica"]):.3f} mm, max {d["replica"].max():.3f} mm')
    for r in sorted(set(d['radius']), reverse=True):
        m = d['radius'] == r
        P = m.sum()
        inside = inside_demo_range(d, r)
        Ja = np.array([jacobians(x, step) for x in d['resp'][m]])
        print(f'radius {r * 100:.0f} cm, {P} points, {inside.sum()} in / {(~inside).sum()} out')
        print(f'  action gain     in x {np.median(Ja[inside][:, 0, 0]):5.2f} y '
              f'{np.median(Ja[inside][:, 1, 1]):5.2f}   out x {np.median(Ja[~inside][:, 0, 0]):5.2f}'
              f' y {np.median(Ja[~inside][:, 1, 1]):5.2f}')
        Y = np.broadcast_to(np.stack(np.meshgrid(g, g, indexing='ij'), -1), (P, G, G, 2))
        groups = np.repeat(d['scene'][m], G * G)
        idx = d['idx'][m][:, 1:-1, 1:-1]
        same = idx == d['idx'][m][:, G // 2, G // 2][:, None, None]
        for s in SITES:
            dF = site_change(d, s, m)
            pred = group_cv(dF.reshape(P * G * G, -1), Y.reshape(-1, 2), groups).reshape(P, G, G, 2)
            Js = np.array([jacobians(x, step) for x in pred])
            print(f'  {s:6s} seen    in x {np.median(Js[inside][:, 0, 0]):5.2f} y '
                  f'{np.median(Js[inside][:, 1, 1]):5.2f}   out x {np.median(Js[~inside][:, 0, 0]):5.2f}'
                  f' y {np.median(Js[~inside][:, 1, 1]):5.2f}')
            if s == 'xattn':
                continue
            J = d['J_' + s][m][:, 1:-1, 1:-1].astype(np.float64).reshape(P, G - 2, G - 2, 2, -1)
            if s == 'dec':                         # head row must match the direction's row
                J = np.where(same[..., None, None], J, np.nan)
            b = block_gains(J, directions(dF, step), inside)
            print(f'  {s:6s} swap    head in: dir in {b["in", "in"][0]:5.2f}/{b["in", "in"][1]:5.2f}'
                  f'  dir out {b["in", "out"][0]:5.2f}/{b["in", "out"][1]:5.2f}   head out: dir in '
                  f'{b["out", "in"][0]:5.2f}/{b["out", "in"][1]:5.2f}  dir out '
                  f'{b["out", "out"][0]:5.2f}/{b["out", "out"][1]:5.2f}'
                  + (f'  ({same.mean():.0%} same row)' if s == 'dec' else ''))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out', required=True)
    p.add_argument('--weights', required=True, help='ACT pretrained_model dir')
    p.add_argument('--task', default='lift', choices=sorted(TASKS))
    p.add_argument('--scenes', type=int, default=20)
    p.add_argument('--env-seed', type=int, default=0, help='another seed = other scenes')
    p.add_argument('--grid', type=int, default=9)
    p.add_argument('--extent', type=float, default=0.08, help='m, each side')
    p.add_argument('--radii', default='0.06,0.02', help='m, eef to grasp point, xy')
    p.add_argument('--report-only', action='store_true')
    args = p.parse_args()
    if not args.report_only:
        np.savez_compressed(args.out, **collect(args))
    report(dict(np.load(args.out)))


if __name__ == '__main__':
    main()
