"""Explicit observation-operator registry.

The registry separates a scientific model configuration from the canonical data
layer. A feature is observed through an explicit mapping
    y_i = h_i(x) + eps_i
and this mapping is versioned. The registry deliberately contains no
hard-coded biological claim about which biomarker corresponds to which latent
state.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np

from neuro_twin.core.observation import LinearObservationModel


@dataclass(frozen=True)
class ObservationSpec:
    name: str
    weights: tuple[float, float, float, float]
    bias: float = 0.0
    sigma: float = 1.0
    group: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("observation name must be non-empty")
        if len(self.weights) != 4:
            raise ValueError("weights must have length 4")
        if self.sigma <= 0.0 or not np.isfinite(self.sigma):
            raise ValueError("sigma must be finite and > 0")
        if not all(np.isfinite(float(v)) for v in self.weights + (self.bias,)):
            raise ValueError("observation mapping must be finite")


@dataclass(frozen=True)
class ObservationModelConfig:
    name: str
    version: str
    specs: tuple[ObservationSpec, ...]
    correlation: tuple[tuple[float, ...], ...] | None = None
    metadata: Mapping[str, str] | None = None

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.specs)

    def canonical_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "specs": [
                {
                    "name": s.name,
                    "weights": list(s.weights),
                    "bias": s.bias,
                    "sigma": s.sigma,
                    "group": s.group,
                }
                for s in self.specs
            ],
            "correlation": [list(row) for row in self.correlation] if self.correlation is not None else None,
            "metadata": dict(self.metadata or {}),
        }

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    def build(self) -> LinearObservationModel:
        H = np.asarray([s.weights for s in self.specs], dtype=float)
        bias = np.asarray([s.bias for s in self.specs], dtype=float)
        if self.correlation is None:
            R = np.diag([s.sigma**2 for s in self.specs])
        else:
            C = np.asarray(self.correlation, dtype=float)
            if C.shape != (len(self.specs), len(self.specs)):
                raise ValueError("correlation matrix dimension does not match specs")
            if not np.allclose(C, C.T, atol=1e-10):
                raise ValueError("correlation matrix must be symmetric")
            if np.any(np.diag(C) <= 0):
                raise ValueError("correlation diagonal must be positive")
            std = np.asarray([s.sigma for s in self.specs])
            R = np.diag(std) @ C @ np.diag(std)
        return LinearObservationModel(H=H, bias=bias, R=R, feature_names=self.feature_names, model_name=self.name, model_version=self.version)


class ObservationRegistry:
    def __init__(self) -> None:
        self._configs: dict[tuple[str, str], ObservationModelConfig] = {}

    def register(self, config: ObservationModelConfig) -> str:
        key = (config.name, config.version)
        if key in self._configs:
            raise ValueError(f"model already registered: {key}")
        if len(set(config.feature_names)) != len(config.feature_names):
            raise ValueError("observation feature names must be unique")
        # Trigger covariance validation at registration time, not during a later fit.
        config.build()
        self._configs[key] = config
        return config.config_hash

    def get(self, name: str, version: str) -> ObservationModelConfig:
        try:
            return self._configs[(name, version)]
        except KeyError as exc:
            raise KeyError(f"unknown observation model: {name}@{version}") from exc

    def list(self) -> Sequence[tuple[str, str]]:
        return tuple(sorted(self._configs))
