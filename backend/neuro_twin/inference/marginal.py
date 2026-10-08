"""Marginal-likelihood state/parameter fitting.

For the canonical affine-linear P-I-N-Q model, the hidden state can be
integrated exactly with a continuous-discrete Kalman filter. This module
optimises the resulting likelihood over x0 and theta, using JAX derivatives
when available and SciPy numerical gradients otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from neuro_twin.core.observation import LinearObservationModel
from neuro_twin.core.parameters import PARAMETER_NAMES, PINQParameters
from neuro_twin.inference.jax_likelihood import JAX_AVAILABLE, negative_loglik_and_grad
from neuro_twin.longitudinal.state_space import kalman_filter


@dataclass(frozen=True)
class MarginalFitResult:
    x0: np.ndarray
    theta: PINQParameters
    negative_log_likelihood: float
    log_likelihood: float
    optimizer_status: int
    optimizer_message: str
    n_iterations: int
    backend: str
    jacobian_rank: int | None = None
    hessian_approximation: np.ndarray | None = None


def _objective_scipy(
    z: np.ndarray,
    time: np.ndarray,
    observations: np.ndarray,
    model: LinearObservationModel,
    initial_covariance: np.ndarray,
    process_noise_spectral_density: np.ndarray | None,
) -> float:
    x0 = z[:4]
    theta = PINQParameters.from_array(z[4:])
    result = kalman_filter(
        time,
        observations,
        model,
        x0,
        initial_covariance,
        theta,
        process_noise_spectral_density=process_noise_spectral_density,
    )
    return float(-result.log_likelihood)


def fit_marginal_likelihood(
    time: np.ndarray,
    observations: np.ndarray,
    observation_model: LinearObservationModel,
    initial_x0: np.ndarray,
    initial_theta: PINQParameters,
    initial_covariance: np.ndarray,
    *,
    theta_lower: np.ndarray | None = None,
    theta_upper: np.ndarray | None = None,
    x0_lower: np.ndarray | None = None,
    x0_upper: np.ndarray | None = None,
    process_noise_spectral_density: np.ndarray | None = None,
    backend: str = "auto",
    options: dict | None = None,
) -> MarginalFitResult:
    """Fit x0 and theta by exact state-space marginal likelihood.

    ``backend='jax'`` requires the optional JAX dependency and supplies exact
    reverse-mode derivatives through the matrix-exponential transition and
    Kalman recursion. ``backend='scipy'`` uses the same reference likelihood
    with numerical differentiation.
    """
    time = np.asarray(time, dtype=float)
    observations = np.asarray(observations, dtype=float)
    initial_x0 = np.asarray(initial_x0, dtype=float)
    initial_theta_arr = initial_theta.as_array()
    z0 = np.concatenate([initial_x0, initial_theta_arr])
    lo = np.full_like(z0, -np.inf, dtype=float)
    hi = np.full_like(z0, np.inf, dtype=float)
    if x0_lower is not None:
        lo[:4] = np.asarray(x0_lower, dtype=float)
    if x0_upper is not None:
        hi[:4] = np.asarray(x0_upper, dtype=float)
    if theta_lower is not None:
        lo[4:] = np.asarray(theta_lower, dtype=float)
    if theta_upper is not None:
        hi[4:] = np.asarray(theta_upper, dtype=float)

    eps = 1e-12
    z0 = np.maximum(z0, np.where(np.isfinite(lo), lo + eps, -np.inf))
    z0 = np.minimum(z0, np.where(np.isfinite(hi), hi - eps, np.inf))
    bounds = list(zip(lo, hi))

    selected_backend = backend
    jac = None
    if backend == "auto":
        selected_backend = "jax" if JAX_AVAILABLE else "scipy"
    if selected_backend == "jax":
        if not JAX_AVAILABLE:
            raise ImportError("JAX backend requested but JAX is not installed")
        fun_with_grad = negative_loglik_and_grad(
            time,
            observations,
            observation_model.H,
            observation_model.R,
            np.asarray(initial_x0, dtype=float),
            np.asarray(initial_covariance, dtype=float),
            process_noise_spectral_density=process_noise_spectral_density,
        )

        def fun(z):
            return fun_with_grad(z)[0]

        def jac(z):
            return fun_with_grad(z)[1]

    elif selected_backend == "scipy":
        fun = lambda z: _objective_scipy(
            z,
            time,
            observations,
            observation_model,
            initial_covariance,
            process_noise_spectral_density,
        )
        jac = None
    else:
        raise ValueError("backend must be 'auto', 'jax', or 'scipy'")

    result = minimize(
        fun,
        z0,
        jac=jac,
        bounds=bounds,
        method="L-BFGS-B",
        options=options or {"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
    )
    z = np.asarray(result.x, dtype=float)
    nll = float(result.fun)
    return MarginalFitResult(
        x0=z[:4],
        theta=PINQParameters.from_array(z[4:]),
        negative_log_likelihood=nll,
        log_likelihood=-nll,
        optimizer_status=int(result.status),
        optimizer_message=str(result.message),
        n_iterations=int(getattr(result, "nit", 0)),
        backend=selected_backend,
    )
