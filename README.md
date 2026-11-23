# Head or Representation?

*Diagnosing Distribution-Shift Failures of Visuomotor Policies with Last-Layer Retraining*

Code and pipeline to reproduce the paper's results (the paper itself is not in this repo).

Imitation-learned visuomotor policies fail under distribution shift, but a success rate does not
say whether the failure is in the learned representation or in the action head that reads it, and
the two cost very different amounts to fix. This repo carries last-layer retraining, a test of
feature quality in image classifiers, over to closed-loop policies (ACT, Diffusion Policy, SmolVLA)
on robomimic Lift, Square and ToolHang.

Extracted from [edge-vla-hil](https://github.com/HenryNVP/edge-vla-hil/tree/policy-interp) (branch
`policy-interp`), a latency testbed, keeping only what the paper's experiments run.

## Method

1. **Probe.** A ridge probe from the head's input features to the grasp point: is the task
   variable still decodable?
2. **Retrain the head.** Freeze the backbone and retrain only the action head on the states the
   policy visits (DAgger). Simulation supplies exact labels for free: render the same state without
   the perturbation and take the frozen base policy's action on it. No demonstrations, no data
   from the shifted domain.
3. **Decision rule.** A failure is **head-limited** if the retrain beats both the base policy and an
   *invisible-object* negative control (a policy that cannot see the object), and
   **representation-limited** otherwise. Then, if needed, go deeper: LoRA, then full fine-tuning.

## Main results

All numbers are closed-loop success (%) on held-out test seeds: 100 episodes per cell (50 for
SmolVLA and ToolHang), with Wilson 95% intervals of at most ±10 points. They match the paper draft
as of 2026-11-22.

**Lift is mostly head-limited; Square and ToolHang are not.** On Lift, 9 of the 11 perturbations
that cause a failure are head-limited, across all three policies, with in-distribution success
at 98–100% after the retrain. On Square and ToolHang none of the 25 failures is: retrained, all
three policies stay at 0–10%.

Base → head retrained, one round of counterfactual labels (**bold** = head-limited, † = 300
episodes, ° = not a failure, – = no failure on the dev seeds):

| Perturbation (both cameras) | Lift ACT | Lift DP | Lift SmolVLA | Square ACT | Square DP | Square SmolVLA | ToolHang ACT | ToolHang DP |
|---|---|---|---|---|---|---|---|---|
| turn 30° + wrist roll 90° | 4 → 63† | **91 → 100** | **64 → 88** | 0 → 0 | 1 → 2 | 0 → 0 | 0 → 0 | 0 → 0 |
| red table, dim + roll | **49 → 78** | **66 → 98** | 90 → 100° | 2 → 0 | 0 → 1 | 2 → 2 | 0 → 0 | 0 → 0 |
| upside down | **36 → 85** | **54 → 99** | 56 → 72 | 0 → 1 | 0 → 1 | 2 → 0 | 0 → 0 | 0 → 0 |
| zoomed | **37 → 83** | – | – | 0 → 0 | 0 → 1 | 0 → 2 | 0 → 0 | 0 → 0 |
| faint object | **82 → 100** | **62 → 94** | 90 → 98° | 2 → 1 | 0 → 0 | 2 → 10 | 2 → 8 | 0 → 0 |
| *invisible object (control)* | 28 → 55† | 58 → 78 | 26 → 60 | 0 → 0 | 0 → 0 | 0 → 2 | 0 → 0 | 0 → 0 |
| head-limited / failures | 4/5 | 4/4 | 1/2 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |

In distribution the base policies reach 100% on Lift; 84% (ACT), 95% (DP) and 26% (SmolVLA) on
Square; 40% (ACT) and 72% (DP) on ToolHang. The two Lift exceptions (ACT under turn + roll,
SmolVLA upside down) recover partly, but no further than the blind control.

**Out-of-range object: the head alone closes most of the gap.** With the Lift cube 5–7 cm beyond
the demonstrated ±3 cm, its offset stays decodable at ACT's head input (gain 0.85 / 0.78) while
the action barely follows it (0.28 / 0.09). Retraining the head on pseudo-labels, with no
demonstrations, closes over half of the gap to a head retrained on expert labels:

| Lift, cube out of range (x / y / in-dist.) | ACT | SmolVLA | Diffusion Policy |
|---|---|---|---|
| base | 55 / 47 / 100 | 67 / 46 / 100 | 66 / 79 / 100 |
| head retrain, DAgger expert labels (per axis) | 99 / 91 / 100 | 91 / 86 / 100 | 98 / 87 / 100 |
| head retrain, pseudo-labels, no demos | 85 / 72 / 99 | 88 / 88 / 92 | 93 / 92 / 100 |
| head retrain, in-distribution labels (control) | 54 / 49 / 100 | 67 / 50 / 100 | 64 / 74 / 100 |
| weights changed | 5,130 | 23,072 | 11.6M |

For ACT, a full 52M-weight retrain on 400 shifted demos (the expert) reaches 100 / 100.

**Square needs depth.** For ACT on Square (turn + roll), full fine-tuning reaches 0, 1, 18, 43 and
40% over DAgger iterations 1–5. On the same four iterations of labels: head only 0%, LoRA (rank 8,
457K weights) 25%, full fine-tuning 43%, while in-distribution success falls from 84% to 73% (48%
after five). On ToolHang, two iterations of full fine-tuning give 0 of 30.

**The regime follows the task, not the severity.** Weakened perturbations that make Square fail
only partly are still not head-limited: object opacity 0.3 takes ACT from 36% to 51% and Diffusion
Policy from 24% to 26%; a wrist roll alone takes Diffusion Policy from 28% to 41%.

**The probe is consistent with the regimes but does not decide them.** The grasp point is
decodable in distribution on Lift and Square (R² 0.78–0.93), not on ToolHang (0.25). Diffusion
Policy upside down on Lift is fixed at a probe R² of 0.20, while ACT under turn + roll (0.85) and
the partial Square failures (0.18–0.49) are not.

**What `pipeline/` covers.** It reproduces all of the above except the cells added to the paper
after this repo was extracted, which ran from edge-vla-hil's `policy-interp` branch and are not
yet in the pipeline:
- the Square SmolVLA and ToolHang Diffusion Policy columns (both policies trained for them);
- ToolHang ACT beyond upside down and the control, and Square Diffusion Policy zoomed;
- the SmolVLA pseudo-label and control rows and the Diffusion Policy column of the position table;
- the in-distribution probe R² and ACT's probe on the partial Square failures.

## Layout

| Path | What |
|---|---|
| [`readout_repair/`](readout_repair) | the library: step harness, sim and perturbations, ACT internals, labels, probes, statistics, head retraining for ACT / Diffusion Policy / SmolVLA |
| [`scripts/`](scripts) | thin command-line entry points over it; each runs as `python scripts/<name>.py --help` |
| [`pipeline/`](pipeline) | reproduction, one stage per step: fetch, train, then one stage per table / figure |
| [`ros2_ws/src/`](ros2_ws/src) | the testbed modules the experiments run on, unchanged from edge-vla-hil: policy backends, env factory, chunk executors (no ROS nodes; pure Python) |
| [`tests/`](tests) | one test file per module |

Generated figures go to `figures/` and results to `outputs/` (both local, not in git).

`readout_repair/`, by module:

| Module | What |
|---|---|
| `harness` | the testbed's step loop: chunk executors on robosuite, synchronous, 1-tick delay |
| `sim` | tasks, envs, object placement, bitwise sim snapshots |
| `nuisance` | the perturbations: render-only camera, light, colour and opacity changes |
| `labels` | counterfactual labels (the shift-free render) and out-of-range pseudo-labels |
| `act` | ACT internals: differentiable forward, site hooks, low-rank edits |
| `probes` | operating points, gain and swap tests, the grasp-point probe |
| `repair` | head retraining (closed form or SGD), LoRA, full fine-tuning; DP and SmolVLA readouts |
| `data`, `stats` | recorded episodes and robomimic demos; Wilson intervals and ridge probes |

`scripts/`, by role:

| Role | Files |
|---|---|
| rollouts and labels | `collect.py` (any policy, task, perturbation; records what the policy saw and its labels) |
| retraining | `repair_act.py` (head, LoRA, full), `repair_dp.py`, `repair_vla.py` |
| probes | `probe_sites.py` (decodability and gain per ACT site, swap test), `probe_r2.py` (grasp-point R²) |
| data | `to_lerobot.py`, `augment_lift.py` (shifted demo replays), `render_toolhang.py` (ToolHang images), `demos_npz.py`, `stamp_act_checkpoint.py` |
| tables and figures | `tables.py` (Tables 1–3, Figure 3), `fig_shifts.py` (Figure 2) |
| environment | `setup_env.sh` (venv, pins, `pip install -e .`) |

## Setup

A plain Python 3.10 venv; no Docker, no ROS. Pins are the versions the results were run with
([`requirements.txt`](requirements.txt)).

```bash
bash scripts/setup_env.sh          # .venv + ~/evh_env.sh; PYTHON=/path/to/python3.10 if not on PATH
source ~/evh_env.sh                # activates the venv, sets MUJOCO_GL (egl, else osmesa)
python -m pytest -q
```

## Reproduce

Each stage skips work whose output already exists, so it can be rerun after an interruption.
Outputs go to `outputs/interp/` (gitignored).

```bash
bash pipeline/0_fetch.sh      # robomimic demos, the DP authors' checkpoints, pinned DP/robomimic sources (~12 GB, checksummed)
bash pipeline/1_train.sh      # ACT on Lift (base + x/y coverage), Square, ToolHang; SmolVLA on Lift
bash pipeline/2_position.sh   # Table 1, probe and swap numbers
bash pipeline/3_perturb.sh    # Table 2, probe R²
bash pipeline/4_severity.sh   # Table 3
bash pipeline/5_depth.sh      # Figure 3 (LoRA / full fine-tuning, DAgger iterations, forgetting)
bash pipeline/6_figures.sh    # Figure 2, and every table again
```

Compute, roughly, on one 8 GB consumer GPU: training a day or two (SmolVLA wants a 16 GB card);
stages 2–5 a few days in total, most of it Diffusion Policy and SmolVLA rollouts. Seeds are fixed (training scenes 1, 2, …; evaluation
scenes 3, 4), so a rerun evaluates on the paper's scenes.

How close a rerun gets:
- **ACT rollouts, probes and head retrains are deterministic** in this venv (bit-identical run to
  run). On two ACT Lift evals (20 episodes each) a rerun matched the recorded success counts and
  every episode's outcome and length; on Square (3 episodes, 2026-11-14) outcomes matched but two
  episodes ran 2 and 6 ticks longer. Trajectories agree to ~1e-5 on short episodes but drift on
  long failing ones: the original runs used Ubuntu's apt scipy, whose LAPACK (reached through
  robosuite's numba-compiled controller) differs from the pip wheel's in the last bits.
- **Diffusion Policy and SmolVLA rollouts are deterministic too, from 2026-11-14 on**:
  `collect.py` seeds torch per scene and forces deterministic kernels, so a rerun of a cell is
  bit-identical (`REPLAY=1 python -m pytest tests/test_replay.py`). The recorded runs predate this
  (unseeded noise), so for these policies a rerun is a fresh sample: expect the paper's cells within
  their confidence intervals, not the same counts.
- **Full or LoRA fine-tuning, and training from scratch, are not deterministic** (GPU training
  kernels): same expectation.

## License

[MIT](LICENSE)
