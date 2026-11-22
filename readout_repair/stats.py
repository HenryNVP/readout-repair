"""Statistics: Wilson intervals for success rates, ridge probes with episode-grouped
cross-validation, and the weighted ridge the closed-form head retrain solves."""
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


def weighted_ridge(X, Y, w, groups, lams=(1e-4, 1e-3, 1e-2, 1e-1, 1.0)):
    """Weighted ridge with intercept, Y ~ X dW^T + db; lambda (x weighted mean squared row norm /
    D) by 5-fold CV over groups. Returns (dW, db, lambda)."""
    w = w / w.sum()

    def fit(m, lam):
        ww = w[m] / w[m].sum()
        mx, my = ww @ X[m], ww @ Y[m]
        Xc = X[m] - mx
        scale = ww @ (Xc ** 2).sum(1) / X.shape[1]
        dW = np.linalg.solve((Xc * ww[:, None]).T @ Xc + lam * scale * np.eye(X.shape[1]),
                             (Xc * ww[:, None]).T @ (Y[m] - my)).T
        return dW, my - dW @ mx
    folds = np.array_split(np.random.default_rng(0).permutation(np.unique(groups)), 5)
    err = []
    for lam in lams:
        e = 0.0
        for fold in folds:
            te = np.isin(groups, fold)
            dW, db = fit(~te, lam)
            e += w[te] @ ((X[te] @ dW.T + db - Y[te]) ** 2).sum(1)
        err.append(e)
    lam = lams[int(np.argmin(err))]
    return (*fit(np.ones(len(X), bool), lam), lam)


def r2(pred: np.ndarray, y: np.ndarray) -> float:
    """1 - SSE / SST per output, averaged."""
    return float(np.mean(1 - ((pred - y) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)))


def rmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(((pred - y) ** 2).sum(1).mean()))
