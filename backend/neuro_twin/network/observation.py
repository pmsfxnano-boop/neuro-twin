"""Explicit observation operator for the six-domain latent network."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np


@dataclass(frozen=True)
class LinearNetworkObservationModel:
    """Versioned linear observation model y = H x + bias + error.

    H must be supplied explicitly by a research pipeline. Feature names,
    modality labels, or biomarker names are never used to infer H.
    """

    H: np.ndarray
    R: np.ndarray
    feature_names: tuple[str, ...]
    model_name: str
    model_version: str
    bias: np.ndarray | None = None

    def __post_init__(self) -> None:
        H = np.array(self.H, dtype=float, copy=True)
        if H.ndim != 2 or H.shape[1] != 6 or H.shape[0] < 1:
            raise ValueError("H must have shape (n_observed, 6)")
        if not np.all(np.isfinite(H)):
            raise ValueError("H must contain only finite values")

        R = np.array(self.R, dtype=float, copy=True)
        if R.shape != (H.shape[0], H.shape[0]) or not np.all(np.isfinite(R)):
            raise ValueError("R must be finite and match the observation dimension")
        if not np.allclose(R, R.T, atol=1e-10):
            raise ValueError("R must be symmetric")
        if float(np.min(np.linalg.eigvalsh(R))) <= 0.0:
            raise ValueError("R must be positive definite")

        if len(self.feature_names) != H.shape[0]:
            raise ValueError("feature_names must match the number of observation rows")
        if any(not isinstance(name, str) or not name.strip() for name in self.feature_names):
            raise ValueError("feature_names must be non-empty strings")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be unique")
        if not self.model_name.strip() or not self.model_version.strip():
            raise ValueError("model_name and model_version must be non-empty")

        if self.bias is None:
            bias = np.zeros(H.shape[0], dtype=float)
        else:
            bias = np.array(self.bias, dtype=float, copy=True)
        if bias.shape != (H.shape[0],) or not np.all(np.isfinite(bias)):
            raise ValueError("bias must be finite and match the observation dimension")

        H.setflags(write=False)
        R.setflags(write=False)
        bias.setflags(write=False)
        object.__setattr__(self, "H", H)
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "bias", bias)
        object.__setattr__(self, "feature_names", tuple(self.feature_names))

    @property
    def n_observed(self) -> int:
        return self.H.shape[0]

    @property
    def config_hash(self) -> str:
        payload = {
            "H": self.H.tolist(),
            "R": self.R.tolist(),
            "bias": self.bias.tolist(),
            "feature_names": list(self.feature_names),
            "model_name": self.model_name,
            "model_version": self.model_version,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def predict(self, state: np.ndarray) -> np.ndarray:
        x = np.asarray(state, dtype=float)
        if x.shape[-1] != 6 or not np.all(np.isfinite(x)):
            raise ValueError("state must be finite and end with dimension 6")
        return x @ self.H.T + self.bias
