import types

import numpy as np
from sim_utils import data_arrays


def test_data_arrays_copies_only_writable_arrays():
    ro = np.zeros(3)
    ro.flags.writeable = False
    d = types.SimpleNamespace(qpos=np.arange(4.0), empty=np.zeros(0), ro=ro, n=3, _hidden=np.ones(2))
    snap = data_arrays(d)
    assert set(snap) == {'qpos'}
    d.qpos[:] = -1
    assert np.array_equal(snap['qpos'], np.arange(4.0))      # a copy, not a view
