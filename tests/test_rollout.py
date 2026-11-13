"""The rollout's chunk timing, on a fake env and policy: hold on tick 0, each chunk executed to
its end, the next requested from the observation of the tick that sends the last action."""
import numpy as np

from rollout import run_episode


def _obs(t):
    return {'agentview_image': np.full((4, 4, 3), t, np.uint8),
            'robot0_eye_in_hand_image': np.zeros((4, 4, 3), np.uint8),
            'robot0_eef_pos': np.array([t, 0.0, 0.0]), 'robot0_eef_quat': np.array([0, 0, 0, 1.0]),
            'robot0_gripper_qpos': np.zeros(2)}


class Env:
    def __init__(self, succeed_at=None):
        self.t, self.sent, self.succeed_at = 0, [], succeed_at

    def step(self, a):
        self.sent.append(np.asarray(a))
        self.t += 1
        return _obs(self.t), 0.0, False, {}

    def _check_success(self):
        return self.t == self.succeed_at


class Policy:
    n_obs_steps, needs_wrist, action_dim = 1, True, 7

    def __init__(self):
        self.seen = []

    def predict(self, obs):
        t = int(obs['agentview'][-1, 0, 0, 0])            # the tick the observation is from
        self.seen.append(t)
        return np.full((3, 7), 100 + t, np.float32)       # chunks of 3, tagged by their obs


def test_chunk_timing():
    env, pol = Env(), Policy()
    ep = run_episode(env, _obs(0), pol, deadline_ticks=8, absolute=False)
    assert (ep.success, ep.ticks) == (False, 8)
    assert pol.seen == [0, 3, 6]                           # tick 0, then each chunk's last tick
    sent = [int(a[0]) for a in env.sent]
    assert sent == [0, 100, 100, 100, 103, 103, 103, 106]  # hold (delta: zeros) on tick 0


def test_absolute_hold_is_the_current_pose_and_success_stops_early():
    env, pol = Env(succeed_at=5), Policy()
    ep = run_episode(env, _obs(0), pol, deadline_ticks=20, absolute=True)
    assert (ep.success, ep.ticks) == (True, 5)
    assert env.sent[0][0] == 0.0 and env.sent[0][3:6].any()   # pose hold, not the origin
