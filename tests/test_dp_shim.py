"""The pytorch3d stand-in DP trains with (scripts/dp_shim) agrees with pytorch3d's conventions as
evh_controller.rotation implements them (6D = the first two rows of R)."""
import os
import sys

import numpy as np
import pytest

from evh_controller import rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts', 'dp_shim'))


def test_shim_matches_the_project_rotation_6d():
    torch = pytest.importorskip('torch')
    pytest.importorskip('scipy')
    from pytorch3d import transforms as T
    aa = np.random.default_rng(0).uniform(-3, 3, (200, 3))
    m = T.axis_angle_to_matrix(torch.as_tensor(aa))
    d6 = T.matrix_to_rotation_6d(m)
    assert np.allclose(d6.numpy(), m.numpy()[:, :2, :].reshape(-1, 6))
    assert np.allclose(T.rotation_6d_to_matrix(d6).numpy(), m.numpy(), atol=1e-6)
    assert np.allclose(rotation.rotation_6d_to_matrix(d6.numpy()), m.numpy(), atol=1e-6)
    back = T.matrix_to_axis_angle(m).numpy()
    assert np.allclose(T.axis_angle_to_matrix(torch.as_tensor(back)).numpy(), m.numpy(), atol=1e-6)
