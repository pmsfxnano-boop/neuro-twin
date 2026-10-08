"""Uncertainty utilities tied to an inference result."""
from __future__ import annotations

import numpy as np

from neuro_twin.uncertainty.laplace import LaplaceResult, laplace_covariance, predictive_covariance


def joint_laplace_from_whitened_jacobian(jacobian_whitened: np.ndarray, objective: float, *, estimate_residual_scale: bool = False) -> LaplaceResult:
    return laplace_covariance(
        jacobian_whitened,
        objective,
        int(np.asarray(jacobian_whitened).shape[0]),
        estimate_residual_scale=estimate_residual_scale,
    )


def delta_predictive_covariance(
    observation_jacobian_joint: np.ndarray,
    joint_covariance: np.ndarray,
    measurement_covariance: np.ndarray | None = None,
) -> np.ndarray:
    return predictive_covariance(observation_jacobian_joint, joint_covariance, measurement_covariance)
