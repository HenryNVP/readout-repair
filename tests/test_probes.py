import numpy as np

from readout_repair.probes import block_gains, directions, jacobians, scores, site_change, target


def test_jacobians_recover_a_linear_response():
    grid = np.linspace(-60, 60, 7)
    A = np.array([[0.3, 0.1], [-0.2, 0.9]])               # response = A @ (dx, dy)
    resp = np.einsum('ab,ijb->ija', A, np.stack(np.meshgrid(grid, grid, indexing='ij'), -1))
    J = jacobians(resp, grid[1] - grid[0])
    assert J.shape == (5, 5, 2, 2) and np.allclose(J, A)


def test_target_is_last_open_entry():
    c = np.zeros((5, 7))
    c[:, 0] = np.arange(5)
    c[3:, 6] = 1.0                                        # closes from entry 3
    assert target(c)[0] == 2


def test_directions_linear_field():
    g = np.arange(5.0)
    F = (g[:, None, None] * np.array([1.0, 0.0]) + g[None, :, None] * np.array([0.0, 2.0]))
    V = directions(F[None], 1.0)                       # (1, 3, 3, 2, 2)
    assert np.allclose(V[0, 1, 1], [[1, 0], [0, 2]])


def test_block_gains_head_vs_direction():
    D = 6
    inside = np.zeros((1, 2, 2), bool)
    inside[0, 0] = True
    e = np.eye(D)
    # in-range directions use features 0/1, out-of-range ones 2/3; every head reads only 0/1
    V = np.zeros((1, 2, 2, D, 2))
    V[0][inside[0]] = np.stack([e[0], e[1]], -1)
    V[0][~inside[0]] = np.stack([e[2], e[3]], -1)
    J = np.broadcast_to(np.stack([e[0], e[1]]), (1, 2, 2, 2, D))
    b = block_gains(J, V, inside)
    assert np.allclose(b['in', 'in'], 1) and np.allclose(b['out', 'in'], 1)
    assert np.allclose(b['in', 'out'], 0) and np.allclose(b['out', 'out'], 0)


def test_site_change_uses_baseline_row():
    G, P = 3, 1
    d = {'dec': np.zeros((P, G, G, 4, 2)), 'idx': np.full((P, G, G), 2), 'radius': np.zeros(P)}
    d['dec'][0, 0, 0, 2] = [1, 1]                    # only row 2 moves
    d['dec'][0, 0, 0, 1] = [9, 9]
    dF = site_change(d, 'dec', np.ones(P, bool))
    assert dF.shape == (P, G, G, 2) and np.allclose(dF[0, 0, 0], [1, 1]) and np.allclose(dF[0, 1, 1], 0)


def fake(n_eps, rng, code):
    """Episodes whose features carry y = code(y) plus noise."""
    X, Y, G = [], [], []
    for g in range(n_eps):
        y = rng.normal(0, 50, (30, 2))
        X.append(np.c_[code(y), rng.normal(0, 1, (30, 6))] + rng.normal(0, 0.5, (30, 8)))
        Y.append(y)
        G.append(np.full(30, g))
    return np.vstack(X), np.vstack(Y), np.concatenate(G)


def test_scores_separate_moved_from_lost_information():
    rng = np.random.default_rng(0)
    base = fake(10, rng, lambda y: y / 10)
    moved = fake(10, rng, lambda y: -y / 10)                 # same info, different direction
    lost = fake(10, rng, lambda y: 0 * y)                    # info gone
    s_moved, s_lost = scores(base, moved), scores(base, lost)
    assert s_moved['joint'] < 0.9 or s_moved['transfer'] < 0      # one map can't serve both signs
    assert s_moved['transfer'] < 0 and s_lost['joint'] < 0.1 and s_lost['transfer'] < 0.1
    same = fake(10, rng, lambda y: y / 10)
    assert scores(base, same)['joint'] > 0.9 and scores(base, same)['transfer'] > 0.9
