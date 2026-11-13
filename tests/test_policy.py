"""Tests for the policy backends that need no checkpoint: make_policy dispatch, the chunk contract
of an unloaded backend, and the action-convention stamp."""
import numpy as np
import pytest

from policy import (
    POLICY_SIDECAR,
    ACTBackend,
    SmolVLABackend,
    make_policy,
    resolve_absolute,
    sidecar_absolute,
)


def _dummy_obs():
    return {
        'agentview': np.zeros((2, 84, 84, 3), dtype=np.uint8),   # stacked history of 2
        'wrist': np.zeros((2, 84, 84, 3), dtype=np.uint8),
        'proprio': np.zeros((2, 9), dtype=np.float32),
    }


@pytest.mark.parametrize('backend,cls', [('act', ACTBackend), ('act_lerobot', ACTBackend),
                                         ('smolvla', SmolVLABackend)])
def test_make_policy_returns_backend(backend, cls):
    assert isinstance(make_policy(backend, weights_path=''), cls)


def test_predict_chunk_shape_and_dtype():
    policy = make_policy('act', '')
    chunk = policy.predict(_dummy_obs())
    assert chunk.shape == (policy.chunk_size, policy.action_dim)
    assert chunk.dtype == np.float32


def test_make_policy_rejects_unknown():
    with pytest.raises(ValueError):
        make_policy('jax', '')


def test_newest_unwraps_history():
    from policy import newest
    imgs = np.arange(2 * 4 * 4 * 3, dtype=np.uint8).reshape(2, 4, 4, 3)
    assert np.array_equal(newest(imgs), imgs[1])
    vecs = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.array_equal(newest(vecs), [3.0, 4.0])
    single = np.zeros((4, 4, 3))
    assert newest(single).shape == (4, 4, 3)   # no history axis -> unchanged


# --- action convention (invariant 1) -------------------------------------------------------
# An ACT checkpoint is 7-dim whether it was trained on delta or absolute actions, so unlike the
# DP backend it cannot derive the mode from its own shapes. It reads a stamp written beside the
# weights; a wrong answer here is silent garbage motion, so it is logged loudly.

def _stamp(tmp_path, payload):
    (tmp_path / POLICY_SIDECAR).write_text(payload)
    return str(tmp_path)


def test_sidecar_absolute_reads_the_stamp(tmp_path):
    assert sidecar_absolute(_stamp(tmp_path, '{"absolute_actions": true}')) is True


def test_sidecar_absolute_reads_a_delta_stamp(tmp_path):
    assert sidecar_absolute(_stamp(tmp_path, '{"absolute_actions": false}')) is False


def test_sidecar_absolute_is_none_when_unstamped(tmp_path):
    assert sidecar_absolute(str(tmp_path)) is None
    assert sidecar_absolute(str(tmp_path / 'nope')) is None
    assert sidecar_absolute('') is None


def test_sidecar_absolute_is_none_when_the_key_is_missing(tmp_path):
    assert sidecar_absolute(_stamp(tmp_path, '{"chunk_size": 16}')) is None


def test_sidecar_absolute_raises_rather_than_defaulting_on_a_broken_stamp(tmp_path):
    """A truncated stamp must not read as 'delta' — that is the failure it exists to prevent."""
    with pytest.raises(ValueError):
        sidecar_absolute(_stamp(tmp_path, '{"absolute_actions": tru'))


@pytest.mark.parametrize('backend', ['act', 'smolvla'])
def test_make_policy_forwards_the_absolute_override(backend):
    assert make_policy(backend, '', absolute=True).absolute_actions is True
    assert make_policy(backend, '', absolute=False).absolute_actions is False


@pytest.mark.parametrize('backend', ['act', 'smolvla'])
def test_backends_default_to_delta_without_a_stamp(backend):
    assert make_policy(backend, '').absolute_actions is False


def test_resolve_absolute_prefers_the_override_over_the_stamp():
    assert resolve_absolute(False, True, 'ckpt') is False
    assert resolve_absolute(True, False, 'ckpt') is True


def test_resolve_absolute_falls_back_to_the_stamp_then_to_delta():
    assert resolve_absolute(None, True, 'ckpt') is True
    assert resolve_absolute(None, False, 'ckpt') is False
    assert resolve_absolute(None, None, 'ckpt') is False
