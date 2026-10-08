"""Local Gaussian/Laplace uncertainty around an optimized estimate."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LaplaceResult:
    covariance: np.ndarray
    standard_errors: np.ndarray
    rank: int
    condition_number: float
    residual_scale: float
    degrees_of_freedom: int


def laplace_covariance(
    jacobian_whitened: np.ndarray,
    objective: float,
    n_residuals: int,
    *,
    prior_precision: np.ndarray | None = None,
    estimate_residual_scale: bool = False,
    rtol: float = 1e-10,
) -> LaplaceResult:
    J = np.asarray(jacobian_whitened, dtype=float)
    if J.ndim != 2:
        raise ValueError("jacobian must be 2-D")
    n, p = J.shape
    if n != n_residuals:
        raise ValueError("n_residuals does not match jacobian")
    if prior_precision is None:
        precision = J.T @ J
    else:
        P = np.asarray(prior_precision, dtype=float)
        if P.shape != (p, p):
            raise ValueError("prior_precision dimension mismatch")
        precision = J.T @ J + 0.5 * (P + P.T)
    u, s, vt = np.linalg.svd(precision, full_matrices=False)
    if len(s) == 0:
        raise ValueError("empty jacobian")
    threshold = max(s[0] * rtol, np.finfo(float).eps)
    keep = s > threshold
    rank = int(np.sum(keep))
    inv_s = np.where(keep, 1.0 / s, 0.0)
    covariance = (vt.T * inv_s) @ vt
    dof = max(n - rank, 1)
    scale = float(2.0 * objective / dof) if estimate_residual_scale else 1.0
    covariance = covariance * scale
    se = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
    nonzero = s[s > threshold]
    condition = float(s[0] / nonzero[-1]) if len(nonzero) else float("inf")
    return LaplaceResult(covariance, se, rank, condition, scale, dof)


def predictive_covariance(jacobian_prediction: np.ndarray, parameter_covariance: np.ndarray, observation_covariance: np.ndarray | None = None) -> np.ndarray:
    G = np.asarray(jacobian_prediction, dtype=float)
    C = np.asarray(parameter_covariance, dtype=float)
    if G.ndim != 2 or C.shape != (G.shape[1], G.shape[1]):
        raise ValueError("dimension mismatch in predictive covariance")
    out = G @ C @ G.T
    if observation_covariance is not None:
        R = np.asarray(observation_covariance, dtype=float)
        if R.shape != out.shape:
            raise ValueError("observation covariance dimension mismatch")
        out = out + R
    return 0.5 * (out + out.T)
