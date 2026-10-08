"""Profile-likelihood identifiability for P-I-N-Q parameters."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from neuro_twin.core.observation import LinearObservationModel
from neuro_twin.core.parameters import PARAMETER_NAMES, PINQParameters
from neuro_twin.inference.marginal import MarginalFitResult, fit_marginal_likelihood


@dataclass(frozen=True)
class ProfilePoint:
    value: float
    negative_log_likelihood: float
    delta_objective: float
    converged: bool
    theta: PINQParameters
    x0: np.ndarray


@dataclass(frozen=True)
class ProfileResult:
    parameter: str
    mle_value: float
    mle_negative_log_likelihood: float
    points: tuple[ProfilePoint, ...]
    confidence_level: float
    threshold_delta_nll: float
    interval: tuple[float, float] | None


def profile_parameter(
    time: np.ndarray,
    observations: np.ndarray,
    observation_model: LinearObservationModel,
    initial_x0: np.ndarray,
    initial_theta: PINQParameters,
    initial_covariance: np.ndarray,
    parameter: str,
    grid: np.ndarray,
    *,
    theta_lower: np.ndarray | None = None,
    theta_upper: np.ndarray | None = None,
    x0_lower: np.ndarray | None = None,
    x0_upper: np.ndarray | None = None,
    process_noise_spectral_density: np.ndarray | None = None,
    confidence_level: float = 0.95,
    backend: str = "auto",
) -> ProfileResult:
    if parameter not in PARAMETER_NAMES:
        raise ValueError(f"unknown parameter: {parameter}")
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or grid.size == 0 or not np.all(np.isfinite(grid)):
        raise ValueError("grid must be a non-empty finite vector")
    idx = PARAMETER_NAMES.index(parameter)

    baseline = fit_marginal_likelihood(
        time,
        observations,
        observation_model,
        initial_x0,
        initial_theta,
        initial_covariance,
        theta_lower=theta_lower,
        theta_upper=theta_upper,
        x0_lower=x0_lower,
        x0_upper=x0_upper,
        process_noise_spectral_density=process_noise_spectral_density,
        backend=backend,
    )
    mle_nll = baseline.negative_log_likelihood

    # Warm-start profile points from the previous optimum for numerical
    # stability. Each point fixes exactly one theta component.
    points: list[ProfilePoint] = []
    theta_seed = baseline.theta.as_array()
    x0_seed = baseline.x0.copy()
    for value in grid:
        lo = None if theta_lower is None else np.asarray(theta_lower, dtype=float).copy()
        hi = None if theta_upper is None else np.asarray(theta_upper, dtype=float).copy()
        if lo is None:
            lo = np.full(len(PARAMETER_NAMES), -np.inf)
        if hi is None:
            hi = np.full(len(PARAMETER_NAMES), np.inf)
        lo[idx] = value
        hi[idx] = value
        fit = fit_marginal_likelihood(
            time,
            observations,
            observation_model,
            x0_seed,
            PINQParameters.from_array(theta_seed),
            initial_covariance,
            theta_lower=lo,
            theta_upper=hi,
            x0_lower=x0_lower,
            x0_upper=x0_upper,
            process_noise_spectral_density=process_noise_spectral_density,
            backend=backend,
        )
        points.append(
            ProfilePoint(
                value=float(value),
                negative_log_likelihood=float(fit.negative_log_likelihood),
                delta_objective=float(fit.negative_log_likelihood - mle_nll),
                converged=fit.optimizer_status >= 0,
                theta=fit.theta,
                x0=fit.x0,
            )
        )
        theta_seed = fit.theta.as_array()
        x0_seed = fit.x0

    threshold = 0.5 * float(chi2.ppf(confidence_level, df=1))
    acceptable = [p.value for p in points if p.delta_objective <= threshold]
    interval = (min(acceptable), max(acceptable)) if acceptable else None
    return ProfileResult(
        parameter=parameter,
        mle_value=float(baseline.theta.as_array()[idx]),
        mle_negative_log_likelihood=mle_nll,
        points=tuple(points),
        confidence_level=float(confidence_level),
        threshold_delta_nll=threshold,
        interval=interval,
    )
