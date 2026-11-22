import numpy as np

from readout_repair.stats import group_cv, ridge_fit, wilson


def test_wilson_brackets_the_rate_and_stays_in_unit_interval():
    lo, hi = wilson(36, 100)
    assert lo < 0.36 < hi
    assert wilson(0, 50)[0] == 0.0 and wilson(50, 50)[1] == 1.0
    assert wilson(0, 0) == (0.0, 1.0)


def test_ridge_recovers_linear_map():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 6)) * [1, 10, 0.1, 1, 1, 1]
    Y = X @ rng.normal(size=(6, 2)) + 3.0
    assert np.abs(ridge_fit(X, Y)(X) - Y).max() < 0.05


def test_group_cv_holds_out_whole_groups():
    # each group's label is a constant X cannot explain: a leak would predict it exactly
    groups = np.repeat(np.arange(10), 20)
    X = np.random.default_rng(1).normal(size=(200, 3))
    Y = groups[:, None].astype(float) * 100
    assert np.abs(group_cv(X, Y, groups) - Y).mean() > 100
