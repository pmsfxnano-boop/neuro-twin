"""Ground-truth synthetic benchmark for parameter/state recovery."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neuro_twin.core.dynamics import integrate
from neuro_twin.core.observation import LinearObservationModel
from neuro_twin.core.parameters import PINQParameters


@dataclass(frozen=True)
class SyntheticBenchmark:
    time: np.ndarray
    x0_true: np.ndarray
    theta_true: PINQParameters
    latent_state: np.ndarray
    observations: np.ndarray
    observation_model: LinearObservationModel


def generate_synthetic(
    *,
    time: np.ndarray,
    x0_true: np.ndarray,
    theta_true: PINQParameters,
    observation_model: LinearObservationModel,
    seed: int = 0,
) -> SyntheticBenchmark:
    time = np.asarray(time, dtype=float)
    latent = integrate(x0_true, time, theta_true).state
    clean = observation_model.predict(latent)
    rng = np.random.default_rng(seed)
    if observation_model.R is None:
        noise = np.zeros_like(clean)
    else:
        cov = np.asarray(observation_model.R, dtype=float)
        noise = rng.multivariate_normal(np.zeros(cov.shape[0]), cov, size=len(time))
    return SyntheticBenchmark(time, np.asarray(x0_true, dtype=float), theta_true, latent, clean + noise, observation_model)
