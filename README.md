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

The **action head** is a module every path from the backbone's features to the action passes
through: for ACT the linear action head (5,130 of 52M weights), for Diffusion Policy the FiLM layers
that condition the denoiser on the observation (11.6M of 300M), for SmolVLA the output projection
of the action expert (23,072 of 450M). The perturbations only change how the simulator renders a
state; the physics is unchanged.

1. **Retrain the head.** Run the base policy under the perturbation and label each state it visits
   with the action the same policy takes at that state rendered *without* the perturbation
   (a counterfactual label: no human, no demonstration). Train only the head with the policy's own
   loss (for ACT's linear head, a closed-form ridge fit in most cases), adding labeled
   in-distribution states to keep normal behaviour. One round: 30–60 perturbed and 20–30
   in-distribution rollouts. For out-of-range objects there is no unperturbed image, so labels
   come either from an expert retrained on shifted demonstrations (queried on the learner's states
   over two DAgger iterations, one for SmolVLA) or from pseudo-labels that extrapolate the policy's
   action from the nearest in-range position.
2. **Decision rule.** With Wilson 95% intervals $[\ell, u]$ for the base policy in distribution
   (in), under the perturbation (base), the retrained policy (ret) and the retrained
   *invisible-object* control (ctl, opacity 0: a retrain that cannot use the object):
   a **failure** if $u_\text{base} < \ell_\text{in}$; **head-limited** if
   $\ell_\text{ret} > \max(u_\text{base}, u_\text{ctl})$; **representation-limited** otherwise.
3. **Fine-tune deeper** when the head retrain does not work: LoRA on every linear layer, then the
   full network, on counterfactual labels over several DAgger iterations.

A ridge **probe** from the head's input features to the grasp point serves as a comparison: does
decodability predict the regime? (It does not.)

## Main results

All numbers are closed-loop success (%) on held-out test seeds, with Wilson 95% intervals (about
±10 points at 100 episodes and ±13 at 50, near 50% success). One training seed per policy–task
pair; each result is scored on 30–300 episodes (counts given per table). They match the paper
draft as of 2026-11-25.

**Lift is mostly head-limited; Square and ToolHang are not.** On Lift, 11 policy–perturbation
pairs fail and 9 of them are head-limited, across all three policies, with in-distribution success
at 98–100% after the retrain. On Square and ToolHang none of the 25 failures is: retrained, all
three policies stay at 0–10% and lose up to 32 points in distribution.

Base → head retrained, one round of counterfactual labels; 100 test episodes per cell, 50 for
SmolVLA and ToolHang (**bold** = head-limited, † = 300 episodes, extended because near the
threshold, ° = not a failure, – = no failure on the dev seeds):

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
Square; 40% (ACT) and 72% (DP) on ToolHang. After the retrain: Lift 98–100; Square 70–90 (ACT),
71–87 (DP), 8–24 (SmolVLA); ToolHang 12–42 (ACT), 40–66 (DP).

In the two Lift exceptions the retrain is not significantly better than the blind control: ACT
under turn + roll goes from 4% to 63% while its control reaches 55%, and SmolVLA upside down from
56% to 72% against a control at 60%. The control behaves like a reach that ignores the cube (for
ACT, 69% of placements near the centre succeed, 40% of those farther away). Two head-limited cases
pass the rule by less than 1.5 points (Diffusion Policy and SmolVLA under turn + roll), and the
first is a failure by only 1.1 points.

**Out-of-range object: the features carry the offset; the head never learned to read it.** With
the Lift cube shifted ±5 or ±7 cm along x or y, on top of the demonstrations' ±3 cm random
placement, a probe on ACT's head input still tracks the offset (gain 0.85 / 0.78) while the action
barely follows it (0.28 / 0.09). A first-order interchange intervention that uses in-range feature
directions at out-of-range states brings the gain back to 0.66 / 0.65. Retraining the head on
pseudo-labels, with no demonstrations, closes over half of the gap between the base policy and a
head retrained on expert labels. With expert labels the head nearly matches the expert, partly by
copying it; the control (in-distribution labels only) does not help, so the gain comes from the
labels. 160 episodes per axis (SmolVLA 120); one retrain per axis:

| Lift, cube out of range (x / y / in-dist.) | ACT | SmolVLA | Diffusion Policy |
|---|---|---|---|
| base | 55 / 47 / 100 | 67 / 46 / 100 | 66 / 79 / 100 |
| head retrain, DAgger expert labels (per axis) | 99 / 91 / 100 | 91 / 86 / 100 | 98 / 87 / 100 |
| head retrain, pseudo-labels, no demos | 85 / 72 / 99 | 88 / 88 / 92 | 93 / 92 / 100 |
| head retrain, in-distribution labels (control) | 54 / 49 / 100 | 67 / 50 / 100 | 64 / 74 / 100 |
| weights changed | 5,130 | 23,072 | 11.6M |

For ACT, a full 52M-weight retrain on 400 shifted demos (the expert) reaches 100 / 100. SmolVLA
takes its expert labels from the same ACT expert; its y retrain lowers its x success from 67% to
55%.

**Square needs depth.** For ACT on Square (turn + roll), full fine-tuning reaches 0, 1, 18, 43 and
40% over DAgger iterations 1–5. With the labels of the first four iterations: head only 0%, LoRA
(rank 8, 457K weights) 25%, full fine-tuning 43%, while in-distribution success falls from 84% to
73% (48% after five). The iterations were collected with the fully fine-tuned policy, which favours
it. On ToolHang, two iterations of full fine-tuning give 0 of 30, and 0 of 30 in distribution.

**The regime follows the task more than the severity.** On Square the same perturbations bring
success down to 0–2%, so task and severity are confounded; weaker perturbations separate them. A
less faint object (opacity 0.3) makes ACT and Diffusion Policy fail only partly, and a wrist roll
alone does it for Diffusion Policy (camera changes still give ACT 0%, even a 10° turn with a 30°
wrist roll). None of these partial failures passes the rule. Two gain 9–10 points, significant by
a two-proportion test (p < 0.03) but far less than on Lift; both were extended to 300 episodes:
opacity 0.3 takes ACT from 38% to 48% (Diffusion Policy 24% to 26%, 100 episodes), and the wrist
roll takes Diffusion Policy from 27% to 36%. At a matched base success, ACT goes from 36% to 85% on
Lift (upside down) but only from 38% to 48% on Square (opacity 0.3). Severity still matters within
Lift: ACT's most severe failure (turn + roll, 4%) is not head-limited.

**The probe does not predict the regime.** A ridge probe on the head's input (frames before
contact, cross-validated R² across episodes) decodes the grasp point in distribution on Lift and
Square (R² 0.78–0.93), not on ToolHang (0.25). Fit on in-distribution and perturbed features
together, ACT's probe still finds it on Lift (0.39–0.90) but loses it on Square (−0.22 and −0.23
under two perturbations). Still, Diffusion Policy upside down on Lift is fixed at a probe R² of
0.20, while ACT under turn + roll (0.85) and the partial Square failures (0.18–0.49) are not.

**Limitations** (as in the paper). Simulation only, and the position expert needed 400 shifted
demonstrations. One seed per policy–task pair. Square and ToolHang have lower in-distribution
success, which limits what the labels can teach. Where each head ends is a choice; for Diffusion
Policy it is large. Still to run: SmolVLA on ToolHang, step 3 on Diffusion Policy, more seeds.

`pipeline/` produces every number above (Reproduce, below).

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
| data | `to_lerobot.py`, `augment_lift.py` (shifted demo replays), `render_toolhang.py` (ToolHang images), `dp_abs_hdf5.py` (DP's ToolHang training set), `demos_npz.py`, `stamp_act_checkpoint.py` |
| training | `train_dp.py` (Diffusion Policy on ToolHang with the authors' code; `dp_shim/`, config `pipeline/dp_toolhang_84.yaml`); ACT and SmolVLA train with `lerobot-train` |
| tables and figures | `tables.py` (Tables 1–2, the severity and depth numbers, `fig_depth.pdf`), `fig_shifts.py` (Figure 2) |
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
bash pipeline/1_train.sh      # ACT on Lift (base + x/y coverage), Square, ToolHang; SmolVLA on Lift and Square; DP on ToolHang
bash pipeline/2_position.sh   # Table 2 (object position), probe and swap numbers
bash pipeline/3_perturb.sh    # Table 1 (visual perturbations), probe R²
bash pipeline/4_severity.sh   # task or severity (Square, weaker perturbations)
bash pipeline/5_depth.sh      # fine-tuning deeper on Square (LoRA / full, DAgger iterations, forgetting)
bash pipeline/6_figures.sh    # Figure 2, and every table again
```

Compute, roughly: the four ACTs train in about a GPU-day on one 8 GB consumer GPU; each SmolVLA
(~10 h) and Diffusion Policy on ToolHang (~12 h) want a 16 GB card (the paper's ran on a rented
one). Stages 2–5 take a few days in total, most of it Diffusion Policy
and SmolVLA rollouts. Seeds are fixed (training scenes 1, 2, …; evaluation scenes 3, 4), so a rerun
evaluates on the paper's scenes.

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
