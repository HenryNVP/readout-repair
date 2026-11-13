# Readout or Representation?

A repair test for distribution-shift failures of visuomotor policies: split a policy (ACT,
Diffusion Policy, SmolVLA) into an encoder and a readout, freeze the encoder, refit only the
readout on simulation-labelled data from the shifted condition, and see whether the failure goes
away. Tasks: robosuite Lift, Square, ToolHang.

Moved out of [edge-vla-hil](../edge-vla-hil) (branch `policy-interp`) with only the code the
experiments import; files keep their original paths, so no imports changed.

## Layout

| Path | What |
|---|---|
| `paper/` | `main.tex`, `refs.bib`, `figs/` |
| `scripts/interp_*.py` | the experiments: collect, probe, gain maps, swap/edit/repair, grid, figures |
| `scripts/oracle_*.py`, `refiner.py`, `square_failures.py`, `dp_mech.py`, `dp_anchor_probe.py`, `fuzzy_gate.py`, `sil_step.py` | scripted demonstrators, rollout harness and helpers the experiments call |
| `scripts/setup_env.sh`, `remote_run.sh` | environment setup; runner for a rented GPU box (SmolVLA runs) |
| `ros2_ws/src/evh_*/` | the pure-Python modules used from the testbed: policy backends, env factory, chunk executors, object filter. No ROS nodes. |
| `tests/` | one test file per module |

`outputs`, `checkpoints`, `data` and `external/{diffusion_policy,robomimic}` are gitignored
symlinks into `~/edge-vla-hil` and `~/edge-vla-hil-interp` (results are ~50 GB).

## Setup

A plain Python 3.10 venv; no Docker, no ROS. Pins are the versions the results were run with.

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
