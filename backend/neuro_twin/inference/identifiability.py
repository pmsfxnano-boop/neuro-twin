"""Fisher/SVD identifiability diagnostics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IdentifiabilityReport:
    rank: int
    singular_values: np.ndarray
    condition_number: float
    weak_modes: np.ndarray
    identifiable_parameters: tuple[int, ...]
    non_identifiable_parameters: tuple[int, ...]


def fisher_svd(S: np.ndarray, W: np.ndarray | None = None, *, rtol: float = 1e-8) -> IdentifiabilityReport:
    """Compute F=SᵀWS and SVD-based effective rank.

    S may be shaped (n_observations, n_parameters). W is an optional precision
    matrix. Singular-value thresholding is scale-relative, preventing a fake
    claim of identifiability when columns differ by many orders of magnitude.
    """
    S = np.asarray(S, dtype=float)
    if S.ndim != 2:
        raise ValueError("S must be a 2-D sensitivity matrix")
    if W is None:
        weighted = S
    else:
        W = np.asarray(W, dtype=float)
        if W.shape != (S.shape[0], S.shape[0]):
            raise ValueError("W must be square with dimension n_observations")
        weighted = np.linalg.cholesky(W) @ S
    _, singular_values, vt = np.linalg.svd(weighted, full_matrices=False)
    threshold = (singular_values[0] * rtol) if len(singular_values) else 0.0
    rank = int(np.sum(singular_values > threshold))
    condition = float(singular_values[0] / singular_values[-1]) if len(singular_values) and singular_values[-1] > max(threshold, np.finfo(float).eps) else float("inf")
    weak_modes = np.where(singular_values <= max(threshold, np.finfo(float).eps))[0]
    if rank == S.shape[1]:
        identifiable = tuple(range(S.shape[1]))
        non_identifiable = ()
    else:
        # Parameters participating most strongly in weak right-singular modes.
        if len(weak_modes):
            weak_load = np.max(np.abs(vt[weak_modes]), axis=0)
            non_identifiable = tuple(int(i) for i in np.where(weak_load >= np.median(weak_load))[0])
        else:
            non_identifiable = tuple(range(rank, S.shape[1]))
        identifiable = tuple(i for i in range(S.shape[1]) if i not in non_identifiable)
    return IdentifiabilityReport(rank, singular_values, condition, weak_modes, identifiable, non_identifiable)
