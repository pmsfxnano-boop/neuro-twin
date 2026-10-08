"""Numerically stable covariance/precision utilities."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve


@dataclass(frozen=True)
class CholeskySolver:
    covariance: np.ndarray
    factor: tuple[np.ndarray, bool]

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        return cho_solve(self.factor, np.asarray(rhs, dtype=float))

    def whiten(self, residual: np.ndarray) -> np.ndarray:
        r = np.asarray(residual, dtype=float)
        # Solve L z = r for lower Cholesky. scipy's cho_solve solves R^T R z=r,
        # so perform the triangular solve explicitly for whitening.
        L = self.factor[0] if self.factor[1] else self.factor[0].T
        from scipy.linalg import solve_triangular
        return solve_triangular(L, r.T, lower=True).T


def make_cholesky(covariance: np.ndarray, *, jitter: float = 0.0, max_jitter: float = 1e-6) -> CholeskySolver:
    R = np.asarray(covariance, dtype=float)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("covariance must be square")
    R = 0.5 * (R + R.T)
    if not np.all(np.isfinite(R)):
        raise ValueError("covariance contains non-finite values")
    eye = np.eye(R.shape[0])
    trial = float(jitter)
    while True:
        try:
            factor = cho_factor(R + trial * eye, lower=True, check_finite=True)
            return CholeskySolver(R + trial * eye, factor)
        except np.linalg.LinAlgError:
            if trial == 0.0:
                trial = np.finfo(float).eps
            else:
                trial *= 10.0
            if trial > max_jitter:
                raise ValueError("covariance is not positive definite within jitter budget")
