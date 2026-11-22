import numpy as np

from readout_repair.nuisance import parse, roll_quat, yaw_quat


def test_nuisance_parse_and_yaw():
    items = parse('light=0.3; cam=0,0.1,0,90')
    assert items[0][0] == 'light' and np.allclose(items[1][1], [0, 0.1, 0, 90])
    q = yaw_quat(np.array([1.0, 0, 0, 0]), 90)
    assert np.allclose(q, [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])


def test_roll_is_right_multiplied():
    q = yaw_quat(np.array([1.0, 0, 0, 0]), 90)              # some orientation
    assert np.allclose(roll_quat(np.array([1.0, 0, 0, 0]), 90), q)   # from identity: same turn
    r = roll_quat(q, 180)
    assert np.isclose(np.linalg.norm(r), 1) and not np.allclose(r, q)
