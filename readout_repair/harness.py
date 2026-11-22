"""Step harness: the testbed's chunk executors on robosuite, delay in ticks, no ROS.

Every rollout in the paper runs through `run_episode` with `make_strategy('synchronous', ...)` and
`fixed_delay(1)`: a chunk requested at tick t is computed immediately and first usable at t + 1.
The pieces are the testbed's own code, unchanged:

  * the strategies are `evh_controller.chunk_executor` (make_executor);
  * the policy is any `ChunkPolicy` from `make_policy` (DP, ACT, SmolVLA);
  * the env is `evh_plant.env_factory.build_env` with the plant's seeding, so scene k is the same
    scene everywhere; the idle-tick hold is the plant's `hold_action`.

The OSC runs at the policy's 20 Hz (the rate the checkpoints were trained at), with no reactive
layer. WORLD-SPEED sets what happens on a tick the executor has no action for: 1 = the world keeps
stepping (the arm holds its last target), 0 = it is frozen. Every tick counts toward the deadline.

The latency-testbed parts (sampled and replayed-WiFi delays, the sweep CLI) are in edge-vla-hil.
"""
from __future__ import annotations

import collections
import dataclasses
import time
from collections.abc import Callable

import numpy as np

# the testbed modules (on sys.path once readout_repair/__init__ has run); numpy only
from evh_controller.inference_worker import Arrival
from evh_plant.env_factory import hold_action

CONTROL_HZ = 20.0


# ------------------------------------------------------------------ delay
def fixed_delay(d: int) -> Callable[[], int]:
    if d < 1:
        raise ValueError(f'delay must be >= 1 tick (a chunk is first usable the tick after it is '
                         f'requested), got {d}')
    return lambda: d


def make_strategy(strategy: str, worker, policy, s_wait: int = 0, s_min: int = 0):
    """make_executor, plus the coverage-aware family's knobs when given: the waiting-mode chunk
    length `s_wait`, and `s_min`, the fewest ticks between time-aligned requests (on a large model
    replanning every tick costs a guided inference per tick; LIBERO uses 10, like its RTC)."""
    from evh_controller.chunk_executor import _REGISTRY, CoverageAwareExecutor, make_executor
    if strategy == 'continuation_inpaint':
        # the queue as a hard inpainting prefix, no state model: oracle_lift's `_inpaint` arm
        from evh_controller.chunk_executor.continuation import ContinuationExecutor
        return ContinuationExecutor(worker, policy, roll_fn=None, inpaint=True)
    cls = _REGISTRY.get(strategy)
    # the whole family, subclasses included (they take the knobs through **kw)
    if cls is not None and issubclass(cls, CoverageAwareExecutor) and (s_wait or s_min):
        kw = {}
        if s_wait:
            kw['s_wait'] = s_wait
        if s_min:
            kw['exec_horizon_min'] = s_min
        return cls(worker, policy, **kw)
    return make_executor(strategy, worker, policy)


class StepWorker:
    """The `InferenceWorker` interface, in ticks: compute on request, deliver `delay()` polls later.

    One request in flight, as in the testbed (invariant 5). `compute_s` is the real inference
    time, reported but not charged: in this harness only `delay` spends ticks.
    """

    def __init__(self, policy, delay: Callable[[], int]) -> None:
        self.policy = policy
        self.delay = delay
        self._pending: list | None = None     # [polls remaining, Arrival]
        self.requests = 0

    @property
    def busy(self) -> bool:
        return self._pending is not None

    def try_request(self, obs, t_issue, epoch, prefix=None, weights=None) -> bool:
        if self._pending is not None:
            return False
        t0 = time.perf_counter()
        if prefix is not None and len(prefix) and weights is not None:
            chunk = self.policy.predict_inpaint(obs, prefix, weights)
        else:
            chunk = self.policy.predict(obs)
        self.requests += 1
        delay = self.delay(t_issue) if getattr(self.delay, 'needs_tick', False) else self.delay()
        self._pending = [delay, Arrival(np.asarray(chunk), t_issue, epoch,
                                        time.perf_counter() - t0)]
        return True

    def poll(self):
        if self._pending is None:
            return None
        self._pending[0] -= 1
        if self._pending[0] > 0:
            return None
        arrival, self._pending = self._pending[1], None
        return arrival

    def shutdown(self) -> None:
        self._pending = None


# ------------------------------------------------------------------ episode
def frames(obs: dict, needs_wrist: bool = True):
    """robosuite obs -> (agentview, wrist, proprio), the plant's observation contract."""
    agent = np.flipud(np.asarray(obs['agentview_image'])).astype(np.uint8)
    wrist = (np.flipud(np.asarray(obs['robot0_eye_in_hand_image'])).astype(np.uint8)
             if needs_wrist else None)
    proprio = np.concatenate([np.asarray(obs['robot0_eef_pos'], np.float32),
                              np.asarray(obs['robot0_eef_quat'], np.float32),
                              np.asarray(obs['robot0_gripper_qpos'], np.float32)])
    return agent, wrist, proprio


def stack(hist, needs_wrist: bool) -> dict:
    out = {'agentview': np.stack([h[0] for h in hist]),
           'proprio': np.stack([h[2] for h in hist])}
    if needs_wrist:
        out['wrist'] = np.stack([h[1] for h in hist])
    return out


@dataclasses.dataclass
class Episode:
    success: bool
    ticks: int            # control ticks elapsed when it ended (success tick + 1, or the deadline)
    env_steps: int        # ticks on which the world advanced
    idle: int             # ticks with no action from the executor
    path_m: float         # commanded end-effector path (sum of |dpos| between executed actions)
    requests: int
    wall_s: float

    @property
    def sim_s(self) -> float:
        return self.ticks / CONTROL_HZ


def run_episode(env, obs, policy, executor, worker, **kw) -> Episode:
    """One episode from an already-reset env whose first observation is `obs` (see `episode`)."""
    gen = episode(env, obs, policy, executor, worker, **kw)
    while True:
        try:
            next(gen)
        except StopIteration as stop:
            return stop.value


def episode(env, obs, policy, executor, worker, *, deadline_ticks: int,
            world_speed: float = 1.0, absolute: bool = True,
            is_success=None, frame_fn=frames, stack_fn=stack, hold_fn=None):
    """One episode as a generator: yields once per control tick, returns the Episode. Driving
    several of these in lockstep is how sil_fleet.py shares one server between robots.

    `frame_fn`/`stack_fn` turn env observations into the policy's input, and `hold_fn(obs)` is
    the command for an idle tick before any action (default: the plant's `hold_action`), so an
    env with a different observation or action convention (LIBERO through LeRobot) plugs in
    without changing the loop.

    An idle tick after the first action repeats the last command in ABSOLUTE mode (it is a
    target: repeating it holds the pose) but sends the hold in DELTA mode, where repeating the
    last delta would keep the arm moving (invariant 2).
    """
    is_success = is_success or (lambda e: bool(e._check_success()))
    hold_fn = hold_fn or (lambda o: hold_action(o, absolute, policy.action_dim))
    needs_wrist = bool(getattr(policy, 'needs_wrist', True))
    executor.reset()
    requests0 = worker.requests
    hist = collections.deque(maxlen=policy.n_obs_steps)
    first = frame_fn(obs, needs_wrist)
    for _ in range(policy.n_obs_steps):     # pad the history with the first frame, as at reset
        hist.append(first)
    last_action = None
    env_steps = idle = 0
    path = 0.0
    acc = 0.0
    t0 = time.perf_counter()
    for t in range(deadline_ticks):
        action = executor.step(stack_fn(hist, needs_wrist), t)
        if action is None:
            idle += 1
            acc += world_speed
            if acc < 1.0 - 1e-9:
                yield                      # frozen world: the tick passes, nothing moves
                continue
            acc -= 1.0
            # absolute: the arm keeps its last target; delta, or before any action: the hold
            command = last_action if (absolute and last_action is not None) else hold_fn(obs)
        else:
            command = np.asarray(action, dtype=np.float32)
            if absolute and last_action is not None:   # a path only exists between targets
                path += float(np.linalg.norm(command[:3] - last_action[:3]))
            last_action = command
        obs, _r, _done, _info = env.step(command)
        env_steps += 1
        hist.append(frame_fn(obs, needs_wrist))
        if is_success(env):
            return Episode(True, t + 1, env_steps, idle, path, worker.requests - requests0,
                           time.perf_counter() - t0)
        yield
    return Episode(False, deadline_ticks, env_steps, idle, path, worker.requests - requests0,
                   time.perf_counter() - t0)
