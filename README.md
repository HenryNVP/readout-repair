# Head or Representation?

*Diagnosing Distribution-Shift Failures of Visuomotor Policies with Last-Layer Retraining*
(paper source: [`paper/main.tex`](paper/main.tex), draft)

Imitation-learned visuomotor policies fail under distribution shift, but a success rate does not
say whether the failure is in the learned representation or in the action head that reads it, and
the two cost very different amounts to fix. This repo carries last-layer retraining, a test of
feature quality in image classifiers, over to closed-loop policies (ACT, Diffusion Policy, SmolVLA)
on robomimic Lift, Square and ToolHang.

Moved out of [edge-vla-hil](https://github.com/HenryNVP/edge-vla-hil/tree/policy-interp) (branch
`policy-interp`) with only the code the experiments import; files keep their original paths, so
no imports changed.

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

Figures: [`paper/figs/`](paper/figs) (`fig_grid.pdf`, `fig_depth.pdf`, `fig_shifts.pdf`, ...).

## Layout

| Path | What |
|---|---|
| [`paper/`](paper) | `main.tex`, `refs.bib`, `figs/` |
| [`scripts/interp_*.py`](scripts) | the experiments: collect, probe, gain maps, swap/edit/repair, grid, figures |
| `scripts/oracle_*.py`, `refiner.py`, `square_failures.py`, `dp_mech.py`, `dp_anchor_probe.py`, `fuzzy_gate.py`, `sil_step.py` | scripted demonstrators, rollout harness and helpers the experiments call |
| [`scripts/setup_env.sh`](scripts/setup_env.sh), [`remote_run.sh`](scripts/remote_run.sh) | environment setup; runner for a rented GPU box (SmolVLA runs) |
| [`ros2_ws/src/`](ros2_ws/src) | the pure-Python modules used from the testbed: policy backends, env factory, chunk executors, object filter. No ROS nodes. |
| [`tests/`](tests) | one test file per module |

**Not in this repo:** `outputs/` (results, ~50 GB), `checkpoints/`, `data/` (robomimic datasets)
and `external/{diffusion_policy,robomimic}` (vendored model code, from the edge-vla-hil repo).
They are gitignored; the scripts expect them at those paths, so place them there (or symlink
them) before running. Checkpoints and results are not published yet.

## Setup

A plain Python 3.10 venv; no Docker, no ROS. Pins are the versions the results were run with
([`requirements.txt`](requirements.txt)).

```bash
bash scripts/setup_env.sh          # .venv + ~/evh_env.sh; PYTHON=/path/to/python3.10 if not on PATH
source ~/evh_env.sh                # activates the venv, sets MUJOCO_GL (egl, else osmesa)
```

`scripts/remote_run.sh` runs the same setup on a rented GPU box (with `EVH_APT=1` for the GL libraries).

How close a rerun gets: the venv is deterministic (two runs, bit-identical), and on two ACT Lift
evals (`cam=0,0.06,0,0` and `cam=0,0,0,30;wroll=90`, 20 episodes each) it matched the recorded
success counts and every episode's outcome and length. Trajectories agree to ~1e-5 on short
episodes but drift on long failing ones: the original runs used Ubuntu's apt scipy, whose LAPACK
(reached through robosuite's numba-compiled controller) differs from the pip wheel's in the last
bits of the torques. Expect identical outcomes almost always, not bit-identical trajectories.

## Run

```bash
python3 -m pytest -q
python3 scripts/<script>.py ...
```

Figures:

```bash
python3 scripts/interp_grid.py    # fig_grid, fig_depth
python3 scripts/interp_figs.py    # fig_fix, fig_edit, fig_swap, fig_seen_vs_action, ...
```

Paper:

```bash
cd paper && docker run --rm -u $(id -u):$(id -g) -v $PWD:/p -w /p texlive/texlive:latest-medium latexmk -pdf main.tex
```

## License

[MIT](LICENSE)
