# CLAUDE.md

The paper "Head or Representation?" and only the code its experiments use, moved out of
edge-vla-hil. `pipeline/` reproduces it from scratch. See `README.md` for layout and run commands. The running log and next steps
are in `docs/interp-plan.md` (local only, excluded from git).

## Build & test

No build, no Docker. `bash scripts/setup_env.sh` makes `.venv` (pins in `requirements.txt`);
`source ~/evh_env.sh` activates it. `tests/conftest.py` and the scripts put `ros2_ws/src/<pkg>`
and `scripts/` on `sys.path`; keep files at these paths. Run tests inside the venv: the
system Python has edge-vla-hil pip-installed in editable mode, so a missing module can
silently import from the old repo there.

```bash
python3 -m pytest -q
```

One test file per module, named after it; keep heavy imports (torch, h5py) out of module scope
in scripts so the suite stays fast.

`external/diffusion_policy` is vendored third-party code (via symlink). Do not edit it.

## Invariants that fail silently

1. **Absolute vs delta mode must agree** between the env (`absolute_actions` -> OSC
   `control_delta=False`) and the policy (`ChunkPolicy.absolute_actions`, derived from the
   checkpoint: a 10-dim head is abs; a 7-dim ACT head needs the `evh_policy.json` stamp from
   `scripts/stamp_act_checkpoint.py`). A mismatch yields garbage motion.
2. **In absolute mode a zero action means "go to the world origin".** Any default/hold action must
   be mode-aware: use `env_factory.hold_action`.
3. **Absolute rotation is 6D, never axis-angle** (Lift targets sit at the pi wrap and flip sign
   frame to frame). Conversions live in `evh_controller/rotation.py`.
4. **Absolute orientation targets are in the OSC tool frame**: `R_controller = R_eef_quat @
   Rz(-90 deg)` (`EEF_TO_CONTROL_QUAT` in `env_factory`). Deltas are immune.
5. **robosuite 1.4.1 + mujoco 2.3.7 only** (1.5.x corrupts multi-camera renders).
6. **Touch the robosuite env only from the thread that built it** (thread-bound EGL context).
7. **The control rate must divide MuJoCo's 2 ms timestep** (`env_factory.check_timebase`).

## Conventions

- Dense module docstrings that explain why; terse comments only where code is subtle.
- Commit messages: short subject line only.
- Paper experiments run through `pipeline/` stages; add a step there, not a one-off driver.
- Long runs: launch in the background and get notified; don't poll.
