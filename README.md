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
| `scripts/remote_setup.sh`, `remote_run.sh` | rented-GPU box setup and runner (SmolVLA runs) |
| `ros2_ws/src/evh_*/` | the pure-Python modules used from the testbed: policy backends, env factory, chunk executors, object filter. No ROS nodes. |
| `tests/` | one test file per module |

`outputs`, `checkpoints`, `data` and `external/{diffusion_policy,robomimic}` are gitignored
symlinks into `~/edge-vla-hil` and `~/edge-vla-hil-interp` (results are ~50 GB).

## Run

Experiments run in the `edge-vla-hil:host` image (built from edge-vla-hil). Mount the symlink
targets at their own paths so the links resolve:

```bash
docker run --rm --gpus all --entrypoint bash -e MUJOCO_GL=egl \
  -v ~/readout-repair:/ws \
  -v /home/henry/edge-vla-hil:/home/henry/edge-vla-hil:ro \
  -v /home/henry/edge-vla-hil-interp/outputs:/home/henry/edge-vla-hil-interp/outputs \
  edge-vla-hil:host -c "cd /ws && python3 scripts/<script>.py ..."
```

Tests (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is required in the image):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q
```

Figures (needs matplotlib >= 3.7, so run on the host, not in the image):

```bash
python3 scripts/interp_grid.py    # fig_grid, fig_depth
python3 scripts/interp_figs.py    # fig_fix, fig_edit, fig_swap, fig_seen_vs_action, ...
```

Paper:

```bash
cd paper && docker run --rm -u $(id -u):$(id -g) -v $PWD:/p -w /p texlive/texlive:latest-medium latexmk -pdf main.tex
```
