"""Statistics shared by the experiments: Wilson intervals for success rates, and the ridge probe
with episode-grouped cross-validation."""
from __future__ import annotations

import math

import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — behaves at small n and near 0/1, unlike the normal approximation."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def ridge_fit(X, Y, lam: float = 1.0):
    """Standardise X on the training rows, then closed-form ridge with an unpenalised bias."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    W = np.linalg.solve(Z.T @ Z + lam * len(Z) * 1e-3 * np.eye(Z.shape[1]), Z.T @ (Y - Y.mean(0)))
    b = Y.mean(0)
    return lambda Xn: ((Xn - mu) / sd) @ W + b


def group_cv(X, Y, groups, k: int = 5, seed: int = 0):
    """Out-of-fold ridge predictions, folds made of whole groups (episodes)."""
    ids = np.unique(groups)
    np.random.default_rng(seed).shuffle(ids)
    pred = np.zeros_like(Y, dtype=float)
    for fold in np.array_split(ids, k):
        test = np.isin(groups, fold)
        pred[test] = ridge_fit(X[~test], Y[~test])(X[test])
    return pred
