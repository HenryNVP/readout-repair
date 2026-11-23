import numpy as np

from readout_repair.labels import align_to_dp


def test_align_to_dp_puts_now_at_the_last_observation_row():
    chunk = np.arange(16.0)[:, None] * np.ones((1, 10))        # row t = t steps ahead
    out = align_to_dp(chunk, n_obs_steps=2)
    assert out.shape == chunk.shape
    assert out[0, 0] == 0 and out[1, 0] == 0 and out[2, 0] == 1 and out[-1, 0] == 14


def test_align_to_dp_is_identity_for_one_observation_step():
    chunk = np.random.default_rng(0).normal(size=(16, 10))
    assert align_to_dp(chunk, 1) is chunk
