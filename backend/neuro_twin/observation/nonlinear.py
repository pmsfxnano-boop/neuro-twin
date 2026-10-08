"""Configurable nonlinear observation operators.

No biological mapping is hard-coded. Each observation channel explicitly
specifies a latent-state loading, optional parameter loading, bias, and link.
The resulting operator is

    y = g(W_x x + W_theta theta + b) + Z u + eps

where u denotes optional Gaussian random effects in the observation scale.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from neuro_twin.core.parameters import PARAMETER_NAMES


@dataclass(frozen=True)
class LinkFunction:
    name: str
    fn: Callable[[np.ndarray], np.ndarray]
    deriv: Callable[[np.ndarray], np.ndarray]


def identity_link() -> LinkFunction:
    return LinkFunction("identity", lambda x: x, lambda x: np.ones_like(x))


def log_link() -> LinkFunction:
    return LinkFunction("log", lambda x: np.log(np.maximum(x, 1e-12)), lambda x: 1.0 / np.maximum(x, 1e-12))


def exp_link() -> LinkFunction:
    return LinkFunction("exp", np.exp, np.exp)


def softplus_link() -> LinkFunction:
    def fn(x: np.ndarray) -> np.ndarray:
        return np.logaddexp(0.0, x)

    def deriv(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        out = np.empty_like(x, dtype=float)
        pos = x >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        ex = np.exp(x[~pos])
        out[~pos] = ex / (1.0 + ex)
        return out

    return LinkFunction("softplus", fn, deriv)


def sigmoid_link() -> LinkFunction:
    def fn(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        out = np.empty_like(x, dtype=float)
        pos = x >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        ex = np.exp(x[~pos])
        out[~pos] = ex / (1.0 + ex)
        return out

    def deriv(x: np.ndarray) -> np.ndarray:
        s = fn(x)
        return s * (1.0 - s)

    return LinkFunction("sigmoid", fn, deriv)


_LINKS = {
    "identity": identity_link,
    "log": log_link,
    "exp": exp_link,
    "softplus": softplus_link,
    "sigmoid": sigmoid_link,
}


@dataclass(frozen=True)
class NonlinearObservationSpec:
    name: str
    state_weights: tuple[float, float, float, float]
    parameter_weights: tuple[float, ...] = (0.0,) * len(PARAMETER_NAMES)
    bias: float = 0.0
    link: str = "identity"
    sigma: float = 1.0
    group: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if len(self.state_weights) != 4:
            raise ValueError("state_weights must have length 4")
        if len(self.parameter_weights) != len(PARAMETER_NAMES):
            raise ValueError("parameter_weights must have length 11")
        if self.link not in _LINKS:
            raise ValueError(f"unsupported link: {self.link}")
        if not np.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("sigma must be finite and > 0")

    @property
    def link_function(self) -> LinkFunction:
        return _LINKS[self.link]()


@dataclass(frozen=True)
class NonlinearObservationModel:
    specs: tuple[NonlinearObservationSpec, ...]
    correlation: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not self.specs:
            raise ValueError("at least one observation specification is required")
        names = [s.name for s in self.specs]
        if len(set(names)) != len(names):
            raise ValueError("observation names must be unique")
        n = len(self.specs)
        if self.correlation is not None:
            C = np.asarray(self.correlation, dtype=float)
            if C.shape != (n, n) or not np.allclose(C, C.T, atol=1e-10):
                raise ValueError("correlation must be symmetric with one row per observation")
            if np.min(np.linalg.eigvalsh(C)) <= 0:
                raise ValueError("correlation must be positive definite")
            object.__setattr__(self, "correlation", C)

    @property
    def n_observed(self) -> int:
        return len(self.specs)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.specs)

    @property
    def R(self) -> np.ndarray:
        std = np.asarray([s.sigma for s in self.specs], dtype=float)
        if self.correlation is None:
            return np.diag(std * std)
        return np.diag(std) @ self.correlation @ np.diag(std)

    def _eta(self, state: np.ndarray, theta: np.ndarray) -> np.ndarray:
        x = np.asarray(state, dtype=float)
        t = np.asarray(theta, dtype=float)
        if x.shape[-1] != 4 or t.shape != (len(PARAMETER_NAMES),):
            raise ValueError("invalid state/theta shapes")
        W_x = np.asarray([s.state_weights for s in self.specs], dtype=float)
        W_t = np.asarray([s.parameter_weights for s in self.specs], dtype=float)
        b = np.asarray([s.bias for s in self.specs], dtype=float)
        return x @ W_x.T + W_t @ t + b

    def predict(self, state: np.ndarray, theta: np.ndarray) -> np.ndarray:
        eta = self._eta(state, theta)
        single = eta.ndim == 1
        E = eta if single else eta
        out = np.empty_like(E, dtype=float)
        for i, s in enumerate(self.specs):
            out[..., i] = s.link_function.fn(E[..., i])
        return out

    def jacobian_state(self, state: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Return dh/dx with shape (..., n_obs, 4)."""
        eta = self._eta(state, theta)
        W_x = np.asarray([s.state_weights for s in self.specs], dtype=float)
        if eta.ndim == 1:
            out = np.empty((self.n_observed, 4), dtype=float)
            for i, s in enumerate(self.specs):
                out[i] = s.link_function.deriv(np.asarray([eta[i]]))[0] * W_x[i]
            return out
        out = np.empty(eta.shape + (4,), dtype=float)
        for i, s in enumerate(self.specs):
            out[..., i, :] = s.link_function.deriv(eta[..., i])[..., None] * W_x[i]
        return out

    def jacobian_theta(self, state: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Return dh/dtheta with shape (..., n_obs, 11)."""
        eta = self._eta(state, theta)
        W_t = np.asarray([s.parameter_weights for s in self.specs], dtype=float)
        if eta.ndim == 1:
            out = np.empty((self.n_observed, len(PARAMETER_NAMES)), dtype=float)
            for i, s in enumerate(self.specs):
                out[i] = s.link_function.deriv(np.asarray([eta[i]]))[0] * W_t[i]
            return out
        out = np.empty(eta.shape + (len(PARAMETER_NAMES),), dtype=float)
        for i, s in enumerate(self.specs):
            out[..., i, :] = s.link_function.deriv(eta[..., i])[..., None] * W_t[i]
        return out
