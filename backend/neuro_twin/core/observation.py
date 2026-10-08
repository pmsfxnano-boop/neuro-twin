"""Configurable observation operator h(x, theta)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinearObservationModel:
    H: np.ndarray
    bias: np.ndarray | None = None
    R: np.ndarray | None = None
    feature_names: tuple[str, ...] | None = None
    model_name: str | None = None
    model_version: str | None = None

    def __post_init__(self):
        H = np.asarray(self.H, dtype=float)
        if H.ndim != 2 or H.shape[1] != 4:
            raise ValueError("H must have shape (n_observed, 4)")
        if not np.all(np.isfinite(H)):
            raise ValueError("H must be finite")
        object.__setattr__(self, "H", H)
        if self.bias is not None:
            b = np.asarray(self.bias, dtype=float)
            if b.shape != (H.shape[0],):
                raise ValueError("bias must match number of observations")
            if not np.all(np.isfinite(b)):
                raise ValueError("bias must be finite")
            object.__setattr__(self, "bias", b)
        if self.R is not None:
            R = np.asarray(self.R, dtype=float)
            if R.shape != (H.shape[0], H.shape[0]):
                raise ValueError("R must be square and match observations")
            if not np.all(np.isfinite(R)):
                raise ValueError("R must be finite")
            if not np.allclose(R, R.T, atol=1e-10):
                raise ValueError("R must be symmetric")
            eig = np.linalg.eigvalsh(R)
            if np.min(eig) <= 0.0:
                raise ValueError("R must be positive definite")
            object.__setattr__(self, "R", R)
        if self.feature_names is not None:
            if len(self.feature_names) != H.shape[0]:
                raise ValueError("feature_names must match number of observations")
            if len(set(self.feature_names)) != len(self.feature_names):
                raise ValueError("feature_names must be unique")

    @property
    def n_observed(self) -> int:
        return self.H.shape[0]

    def predict(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=float)
        if state.shape[-1] != 4:
            raise ValueError("state must end with dimension 4")
        y = state @ self.H.T
        if self.bias is not None:
            y = y + self.bias
        return y

    def jacobian_state(self, state: np.ndarray | None = None) -> np.ndarray:
        """Return dh/dx; identical at every state for the affine-linear operator."""
        return self.H.copy()
