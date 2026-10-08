"""Parameter vector for the canonical P-I-N-Q model."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PARAMETER_NAMES = (
    "aP", "bPI", "dP", "bIP", "dI", "bNP", "bNI", "dN", "bQN", "bQI", "dQ"
)


@dataclass(frozen=True)
class PINQParameters:
    aP: float
    bPI: float
    dP: float
    bIP: float
    dI: float
    bNP: float
    bNI: float
    dN: float
    bQN: float
    bQI: float
    dQ: float

    def as_array(self) -> np.ndarray:
        return np.array([getattr(self, name) for name in PARAMETER_NAMES], dtype=float)

    @classmethod
    def from_array(cls, theta: np.ndarray) -> "PINQParameters":
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (len(PARAMETER_NAMES),):
            raise ValueError(f"theta must have shape {(len(PARAMETER_NAMES),)}")
        return cls(**dict(zip(PARAMETER_NAMES, theta)))

    def matrix_A(self) -> np.ndarray:
        return np.array([
            [-self.dP, self.bPI, 0.0, 0.0],
            [self.bIP, -self.dI, 0.0, 0.0],
            [self.bNP, self.bNI, -self.dN, 0.0],
            [0.0, self.bQI, self.bQN, -self.dQ],
        ], dtype=float)

    def vector_c(self) -> np.ndarray:
        return np.array([self.aP, 0.0, 0.0, 0.0], dtype=float)
