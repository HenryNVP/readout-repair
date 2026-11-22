import types

import numpy as np

from readout_repair.sim import restore, snapshot


def test_snapshot_copies_only_writable_arrays():
    ro = np.zeros(3)
    ro.flags.writeable = False
    d = types.SimpleNamespace(qpos=np.arange(4.0), empty=np.zeros(0), ro=ro, n=3, _hidden=np.ones(2))
    snap = snapshot(d)
    assert set(snap) == {'qpos'}
    d.qpos[:] = -1
    assert np.array_equal(snap['qpos'], np.arange(4.0))      # a copy, not a view


def test_restore_writes_back_and_skips_resized_arrays():
    d = types.SimpleNamespace(qpos=np.arange(4.0), efc=np.zeros(2))
    snap = snapshot(d)
    d.qpos[:] = -1
    d.efc = np.zeros(5)                                      # contact arrays change size
    restore(d, snap)
    assert np.array_equal(d.qpos, np.arange(4.0)) and d.efc.shape == (5,)
