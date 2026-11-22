"""Last-layer retraining as a test of distribution-shift failures: the code behind scripts/.

  harness   the testbed's step loop (chunk executors on robosuite, delay in ticks)
  sim       tasks, envs, object placement and sim snapshots
  nuisance  render-only perturbations (camera, light, colour, object opacity)
  act       ACT internals: a differentiable forward, site hooks, low-rank edits
  labels    counterfactual and pseudo-labels; DP's observation features
  data      recorded episodes and robomimic demos
  probes    operating points, gain probes, the swap test, the grasp-point probe
  stats     Wilson intervals, ridge probes, grouped cross-validation
  repair    head (readout) retraining for ACT, Diffusion Policy and SmolVLA

The testbed modules in ros2_ws/src are pure Python; importing this package puts them on sys.path.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _pkg in ('evh_controller', 'evh_plant', 'evh_reactive'):
    _path = os.path.join(REPO, 'ros2_ws', 'src', _pkg)
    if _path not in sys.path:
        sys.path.insert(0, _path)
