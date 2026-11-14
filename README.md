# Head or Representation?

*Diagnosing Distribution-Shift Failures of Visuomotor Policies with Last-Layer Retraining*
(paper source: [`paper/main.tex`](paper/main.tex), draft)

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
SmolVLA and ToolHang), with Wilson 95% intervals of at most ±10 points.

**Lift is mostly head-limited; Square and ToolHang are not.** On Lift, 9 of the 11 perturbations
that cause a failure are head-limited, across all three policies, with in-distribution success
intact. On Square and ToolHang none of the 10 failures is: the retrain leaves ACT and Diffusion
Policy at 0–2%, the control's level.

Base → head retrained, one round of counterfactual labels (**bold** = head-limited, † = 300
episodes, ° = not a failure):

| Perturbation (both cameras) | Lift ACT | Lift Diffusion | Lift SmolVLA | Square ACT | Square Diffusion | ToolHang ACT |
|---|---|---|---|---|---|---|
| turn 30° + wrist roll 90° | 4 → 63† | **91 → 100** | **64 → 88** | 0 → 0 | 1 → 2 | – |
| red table, dim + roll | **49 → 78** | **66 → 98** | 90 → 100° | 2 → 0 | 0 → 1 | – |
| upside down | **36 → 85** | **54 → 99** | 56 → 72 | 0 → 1 | 0 → 1 | 0 → 0 |
| zoomed | **37 → 83** | – | – | 0 → 0 | – | – |
| faint object | **82 → 100** | **62 → 94** | 90 → 98° | 2 → 1 | 0 → 0 | – |
| *invisible object (control)* | 28 → 55† | 58 → 78 | 26 → 60 | 0 → 0 | 0 → 0 | 0 → 0 |
| head-limited / failures | 4/5 | 4/4 | 1/2 | 0/5 | 0/4 | 0/1 |

The two Lift exceptions (ACT under turn + roll, SmolVLA upside down) recover partly, but no further
than the blind control.

**Out-of-range object: the head alone recovers most of the gap.** With the Lift cube 5–7 cm beyond
the demonstrated ±3 cm, its offset stays decodable at the head's input while the action barely
follows it. Retraining ACT's 5,130-weight head on pseudo-labels, with no demonstrations, recovers
most of what a full 52M-weight retrain on 400 shifted demos gets:

| ACT, Lift, cube out of range | x | y | in-dist. | weights changed |
|---|---|---|---|---|
| base | 55 | 47 | 100 | – |
| full retrain, 400 shifted demos | 100 | 100 | 100 | 52M |
| head retrain, DAgger expert labels | 99 | 91 | 100 | 5,130 |
| head retrain, pseudo-labels, no demos | 85 | 72 | 99 | 5,130 |
| head retrain, in-distribution labels (control) | 54 | 49 | 100 | 5,220 |

SmolVLA fails out of range the same way (67 / 46) and is fixed the same way (91 / 86, retraining
23,072 weights).

**Square needs depth.** For ACT on Square (turn + roll), four DAgger iterations on the same labels
give: head only 0%, LoRA (rank 8, 457K weights) 25%, full fine-tuning 43%. The cost is forgetting:
in-distribution success drops from 84% to 73%.

**The probe is consistent with the regimes but does not decide them.** Decodability of the grasp
point does not predict ACT's turn + roll exception, and Diffusion Policy loses it on Lift where the
retrain still works. The retrain, not the probe, is the decisive measurement.

**Still running:** whether partial-strength perturbations on Square are head-limited, i.e. whether
the regime follows the task or the severity of the failure (`tab:severity` in the paper).

## Layout

| Path | What |
|---|---|
| [`paper/`](paper) | `main.tex`, `refs.bib`, `figs/` |
| [`pipeline/`](pipeline) | reproduction, one stage per step: fetch, train, then one script per table / figure |
| [`scripts/`](scripts) | the experiments; each runs as `python scripts/<name>.py --help` |
| [`ros2_ws/src/`](ros2_ws/src) | the testbed modules the experiments run on, unchanged from edge-vla-hil: policy backends, env factory, chunk executors (no ROS nodes; pure Python) |
| [`tests/`](tests) | one test file per module |

The rollouts run through the testbed's own step harness (`scripts/sil_step.py`, synchronous chunk
execution) and modules, exactly as for the paper; only scripts the paper does not use were removed.

`scripts/`, by role:

| Role | Files |
|---|---|
| rollouts and labels | `interp_collect.py` (any policy, task, perturbation; records what the policy saw and counterfactual labels), `interp_nuisance.py` (the perturbations), `sil_step.py` (the step harness), `sim_utils.py` |
| retraining | `interp_edit.py` (ACT: head, LoRA, full), `interp_dp_repair.py`, `interp_vla_repair.py` |
| probes | `interp_act_sites.py` (decodability and gain per ACT site, swap test), `interp_gainmap.py`, `interp_repairable.py` (probe R²), `stats.py` |
| data | `robomimic_to_lerobot.py`, `interp_augment.py` (shifted demo replays), `interp_regen.py` (ToolHang images), `interp_demos_npz.py`, `stamp_act_checkpoint.py` |
| tables and figures | `interp_grid.py` (Tables 1–3, Figure 3), `interp_nuisance.py` (Figure 2) |
| environment | `setup_env.sh`; `run_manifest.py`, `wifi_trace.py` (used by the harness) |

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
  every episode's outcome and length. Trajectories agree to ~1e-5 on short episodes but drift on
  long failing ones: the original runs used Ubuntu's apt scipy, whose LAPACK (reached through
  robosuite's numba-compiled controller) differs from the pip wheel's in the last bits.
- **Diffusion Policy and SmolVLA rollouts, and full or LoRA fine-tuning, are not**: GPU sampling
  and training kernels differ run to run, as they did for the original runs. Neither is training
  from scratch. Expect those cells within their confidence intervals, not the same counts.

Paper:

```bash
cd paper && docker run --rm -u $(id -u):$(id -g) -v $PWD:/p -w /p texlive/texlive:latest-medium latexmk -pdf main.tex
```

## License

[MIT](LICENSE)
