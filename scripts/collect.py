#!/usr/bin/env python3
"""Roll a policy out (ACT, Diffusion Policy or SmolVLA) on a task, optionally perturbed, and record
one .npz per episode: per tick the state (eef, gripper, object pose), the executed commands,
success, and on request labels and what the policy saw.

The rollout is the harness's synchronous loop (d = 1 tick). Labels (--teacher, --selflabel) are
computed on the same stacked history the executor gets and do not touch the rollout: the sim is
restored bitwise after each counterfactual.

    python scripts/collect.py --task lift --backend act --weights <pretrained_model> \
        --seed 1 --episodes 30 --selflabel cf --nuisance 'roll=180;wroll=180' --out-dir ...
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from readout_repair import REPO, harness
from readout_repair.labels import align_to_dp, cf_label, encode, self_label
from readout_repair.sim import TASKS, Recorder, make_env, observe, place, state


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out-dir', required=True)
    p.add_argument('--task', default='square', choices=sorted(TASKS))
    p.add_argument('--backend', default='dp', help='make_policy backend (act, dp, smolvla)')
    p.add_argument('--weights', default='', help="default: the task's DP checkpoint")
    p.add_argument('--denoise-steps', type=int, default=16)
    p.add_argument('--episodes', type=int, default=200)
    p.add_argument('--first-scene', type=int, default=0, help='resume: skip scenes below this')
    p.add_argument('--deadline-s', type=float, default=20.0)
    p.add_argument('--seed', type=int, default=0, help='env seed: scene k is the same for every policy')
    p.add_argument('--shift', default='0,0',
                   help='dx,dy: move (m) the object at reset, before the first frame '
                        '(use --shift=-0.02,0: a leading minus reads as an option)')
    p.add_argument('--teacher', default='',
                   help="another ACT (the expert): save its chunk as the label (ACT: at every "
                        'prediction, with the head inputs; DP / SmolVLA: every --label-every ticks)')
    p.add_argument('--selflabel', nargs='?', const='slope', default='', choices=('slope', 'full', 'cf'),
                   help='labels without demos or a teacher: cf = the policy on the --nuisance-free '
                        "render (labels.cf_label); slope / full = the out-of-range pseudo-label "
                        '(labels.self_label; full = slopes rescaled to gain 1)')
    p.add_argument('--nuisance', default='', help='nuisance spec: appearance shift at every reset')
    p.add_argument('--label-weights', default='',
                   help='--selflabel cf: label with this checkpoint (the unedited policy) instead '
                        'of the acting one (DAgger iterations after a repair)')
    p.add_argument('--save-images', action='store_true',
                   help='also save what the policy saw at each labelled tick (agentview, wrist, '
                        'proprio), for repairs deeper than the head')
    p.add_argument('--label-every', type=int, default=1,
                   help='DP / SmolVLA: label every n-th tick (rows say which: label_tick)')
    args = p.parse_args()
    shift = np.array([float(v) for v in args.shift.split(',')])
    task = TASKS[args.task]
    weights = args.weights or os.path.join(REPO, 'checkpoints', task['ckpt'])
    # DP and SmolVLA sample noise from torch's RNG, and default GPU kernels are not bitwise
    # repeatable: deterministic kernels plus a seed per scene make a rerun of scene k reproduce it
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')   # before the first cuBLAS call
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    from evh_controller.policy import make_policy

    os.makedirs(args.out_dir, exist_ok=True)
    policy = make_policy(args.backend, weights, denoise_steps=args.denoise_steps)
    env, first_obs = make_env(task, args.seed, args.deadline_s, policy.absolute_actions)
    act_labels = bool(args.teacher or args.selflabel) and args.backend == 'act'
    if act_labels:
        from readout_repair.act import Sites, forward
        sites = Sites(policy._model)
        teacher = make_policy('act', args.teacher)._model if args.teacher else None
    from readout_repair.nuisance import Nuisance
    nuis = Nuisance(args.nuisance) if args.nuisance else None
    labeller = (make_policy(args.backend, args.label_weights, denoise_steps=args.denoise_steps)
                if args.label_weights else policy)
    # SmolVLA / DP with --teacher: an ACT expert labels their own states (same abs10 action space;
    # for DP the chunk is shifted onto its horizon, align_to_dp)
    teacher_pol = (make_policy('act', args.teacher) if args.teacher and args.backend in ('smolvla', 'dp')
                   else None)
    sampled_labels = args.backend in ('dp', 'smolvla') and bool(args.selflabel or teacher_pol is not None)

    def fresh(o):
        """Reset observation with the nuisance applied (the model is rebuilt each reset)."""
        if nuis is not None:
            nuis.on(env)
            o = observe(env)
        return place(env, o, shift, task)
    obs = fresh(first_obs)
    worker = harness.StepWorker(policy, harness.fixed_delay(1))
    executor = harness.make_strategy('synchronous', worker, policy)
    wins = 0
    for scene in range(args.episodes):
        if scene >= args.first_scene:
            torch.manual_seed(args.seed * 100_000 + scene)
            rec = Recorder(env, obs)
            feats, states, heads, labels, label_ticks, seen = [], [], [], [], [], []

            def see(stacked):
                if args.save_images:
                    seen.append((stacked['agentview'][-1], stacked['wrist'][-1], stacked['proprio'][-1]))

            def stack_fn(hist, needs_wrist):
                stacked = harness.stack(hist, needs_wrist)
                t = len(states)                                       # this tick
                if args.backend == 'dp':
                    feats.append(encode(policy, stacked))
                if sampled_labels:                  # DP / SmolVLA: costly, every --label-every ticks
                    if t % args.label_every == 0:
                        label_ticks.append(t)
                        see(stacked)
                        if args.selflabel == 'cf':
                            labels.append(cf_label(env, labeller, nuis, args.backend))
                        elif teacher_pol is not None:
                            with torch.no_grad():
                                labels.append(teacher_pol._model.predict_action_chunk(
                                    teacher_pol._obs_to_batch(stacked))[0].cpu().numpy())
                            if args.backend == 'dp':
                                labels[-1] = align_to_dp(labels[-1], policy.n_obs_steps)
                        else:
                            labels.append(self_label(env, labeller, rec.obs, None, task,
                                                     full=args.selflabel == 'full', backend=args.backend))
                elif act_labels:
                    see(stacked)
                    b = policy._obs_to_batch(stacked)
                    with torch.no_grad():
                        forward(policy._model, b)
                        heads.append(sites.out['dec'][:, 0].float().cpu().numpy().astype(np.float16))
                        if teacher is not None:
                            labels.append(teacher.predict_action_chunk(b)[0].cpu().numpy())
                    if teacher is None:
                        labels.append(cf_label(env, labeller, nuis) if args.selflabel == 'cf' else
                                      self_label(env, policy, rec.obs, b, task,
                                                 full=args.selflabel == 'full'))
                states.append(state(rec.obs, task))
                return stacked

            ep = harness.run_episode(rec, obs, policy, executor, worker,
                                     deadline_ticks=int(args.deadline_s * harness.CONTROL_HZ),
                                     absolute=policy.absolute_actions, stack_fn=stack_fn)
            states.append(state(rec.obs, task))     # the state after the last step
            cols = {k: np.array([s[k] for s in states]) for k in states[0]}
            cls = 'success' if ep.success else 'fail'
            np.savez_compressed(os.path.join(args.out_dir, f'ep{scene:04d}.npz'),
                                feats=np.array(feats), commands=np.array(rec.commands),
                                success=ep.success, cls=cls, ticks=ep.ticks, shift=shift,
                                task=args.task, nuisance=args.nuisance,
                                **cols, **({'dec': np.array(heads)} if heads else {}),
                                **({'teacher': np.array(labels)} if labels else {}),
                                **({'label_tick': np.array(label_ticks)} if label_ticks else {}),
                                **({'img_agent': np.array([x[0] for x in seen]),
                                    'img_wrist': np.array([x[1] for x in seen]),
                                    'proprio': np.array([x[2] for x in seen], np.float32)}
                                   if seen else {}))
            wins += ep.success
            print(f'scene {scene}: {cls} ticks={ep.ticks} ({wins}/{scene - args.first_scene + 1})', flush=True)
        obs = fresh(env.reset())
    env.close()


if __name__ == '__main__':
    main()
