"""Gaussian random-effects design for persistent center/assay/scanner effects."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RandomEffectSpec:
    name: str
    n_latent: int
    prior_covariance: np.ndarray
    design: np.ndarray

    def __post_init__(self) -> None:
        if not self.name or self.n_latent <= 0:
            raise ValueError("invalid random-effect specification")
        cov = np.asarray(self.prior_covariance, dtype=float)
        Z = np.asarray(self.design, dtype=float)
        if cov.shape != (self.n_latent, self.n_latent):
            raise ValueError("prior_covariance shape mismatch")
        if Z.ndim != 2 or Z.shape[1] != self.n_latent:
            raise ValueError("design must have shape (n_observations, n_latent)")
        if not np.all(np.isfinite(cov)) or not np.all(np.isfinite(Z)):
            raise ValueError("random effects must be finite")
        cov = 0.5 * (cov + cov.T)
        if np.min(np.linalg.eigvalsh(cov)) <= 0:
            raise ValueError("prior_covariance must be positive definite")
        object.__setattr__(self, "prior_covariance", cov)
        object.__setattr__(self, "design", Z)


@dataclass(frozen=True)
class RandomEffectsModel:
    specs: tuple[RandomEffectSpec, ...]

    @property
    def n_latent(self) -> int:
        return sum(s.n_latent for s in self.specs)

    @property
    def design(self) -> np.ndarray:
        if not self.specs:
            return np.zeros((0, 0))
        Z = np.concatenate([s.design for s in self.specs], axis=1)
        return Z

    @property
    def covariance(self) -> np.ndarray:
        if not self.specs:
            return np.zeros((0, 0))
        out = np.zeros((self.n_latent, self.n_latent), dtype=float)
        start = 0
        for s in self.specs:
            stop = start + s.n_latent
            out[start:stop, start:stop] = s.prior_covariance
            start = stop
        return out

    def augment_state(self, state: np.ndarray, random_effect: np.ndarray) -> np.ndarray:
        x = np.asarray(state, dtype=float)
        u = np.asarray(random_effect, dtype=float)
        if x.shape != (4,) or u.shape != (self.n_latent,):
            raise ValueError("invalid augmented state dimensions")
        return np.concatenate([x, u])

    def expanded_initial_covariance(self, state_covariance: np.ndarray) -> np.ndarray:
        P = np.asarray(state_covariance, dtype=float)
        if P.shape != (4, 4):
            raise ValueError("state_covariance must be 4x4")
        out = np.zeros((4 + self.n_latent, 4 + self.n_latent), dtype=float)
        out[:4, :4] = P
        out[4:, 4:] = self.covariance
        return out
