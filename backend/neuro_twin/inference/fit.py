"""Constrained weighted least-squares parameter/state inference.

This is the deterministic baseline estimator. It is designed to be the
reference against which Bayesian samplers and online filters are benchmarked.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from neuro_twin.core.dynamics import integrate_with_sensitivities, integrate_with_state_transition
from neuro_twin.core.observation import LinearObservationModel
from neuro_twin.core.parameters import PARAMETER_NAMES, PINQParameters
from neuro_twin.uncertainty.laplace import laplace_covariance


@dataclass(frozen=True)
class FitResult:
    x0: np.ndarray
    theta: PINQParameters
    predicted_observations: np.ndarray
    residuals: np.ndarray
    objective: float
    jacobian_rank: int
    optimizer_status: int
    optimizer_message: str
    joint_covariance: np.ndarray | None = None
    parameter_standard_errors: np.ndarray | None = None
    covariance_rank: int | None = None
    covariance_condition_number: float | None = None


def _whitener(R: np.ndarray, n: int) -> np.ndarray:
    if R is None:
        return np.eye(n)
    R = np.asarray(R, dtype=float)
    if R.shape != (n, n):
        raise ValueError("R dimension does not match observations")
    # Lower-triangular Cholesky gives L z = residual for Mahalanobis norm.
    return np.linalg.inv(np.linalg.cholesky(R))


def fit_pinq(
    time: np.ndarray,
    observed: np.ndarray,
    observation_model: LinearObservationModel,
    initial_x0: np.ndarray,
    initial_theta: PINQParameters,
    *,
    theta_lower: np.ndarray | None = None,
    theta_upper: np.ndarray | None = None,
    x0_lower: np.ndarray | None = None,
    x0_upper: np.ndarray | None = None,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    estimate_residual_scale: bool = False,
) -> FitResult:
    time = np.asarray(time, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if observed.ndim != 2 or observed.shape[0] != len(time):
        raise ValueError("observed must have shape (n_time, n_observed)")
    if observed.shape[1] != observation_model.H.shape[0]:
        raise ValueError("observed dimension does not match observation model")

    ntheta = len(PARAMETER_NAMES)
    initial_x0 = np.asarray(initial_x0, dtype=float)
    initial_theta = initial_theta.as_array()
    z0 = np.concatenate([initial_x0, initial_theta])

    lb = np.full(4 + ntheta, -np.inf)
    ub = np.full(4 + ntheta, np.inf)
    if x0_lower is not None:
        lb[:4] = np.asarray(x0_lower, dtype=float)
    if x0_upper is not None:
        ub[:4] = np.asarray(x0_upper, dtype=float)
    if theta_lower is not None:
        lb[4:] = np.asarray(theta_lower, dtype=float)
    if theta_upper is not None:
        ub[4:] = np.asarray(theta_upper, dtype=float)

    # Avoid exactly-on-bound starting points causing scipy's strict interior
    # requirement to fail.
    eps = 1e-12
    z0 = np.maximum(z0, lb + eps)
    z0 = np.minimum(z0, ub - eps)
    Wsqrt = _whitener(observation_model.R, observed.shape[1])

    def evaluate(z: np.ndarray):
        x0 = z[:4]
        theta = PINQParameters.from_array(z[4:])
        s_theta = integrate_with_sensitivities(x0, time, theta, rtol=rtol, atol=atol)
        # Exact transition sensitivity wrt x0: integrate once with S0=I.
        state_x0, s_x0 = integrate_with_state_transition(x0, time, theta, rtol=rtol, atol=atol)
        pred = observation_model.predict(s_theta.state)
        raw = pred - observed
        white = (Wsqrt @ raw.T).T
        J_theta = np.einsum("ij,tjk->tik", observation_model.H, s_theta.sensitivity)
        J_x0 = np.einsum("ij,tjk->tik", observation_model.H, s_x0)
        J = np.concatenate([J_x0, J_theta], axis=2)
        J_white = np.einsum("ij,tjk->tik", Wsqrt, J)
        return white.ravel(), J_white.reshape(-1, 4 + ntheta), pred

    def fun(z):
        return evaluate(z)[0]

    def jac(z):
        return evaluate(z)[1]

    result = least_squares(fun, z0, jac=jac, bounds=(lb, ub), method="trf", x_scale="jac", loss="linear")
    residual, J, pred = evaluate(result.x)
    rank = int(np.linalg.matrix_rank(J))
    objective = float(0.5 * np.dot(residual, residual))
    lap = laplace_covariance(
        J,
        objective=objective,
        n_residuals=J.shape[0],
        estimate_residual_scale=estimate_residual_scale,
    )
    return FitResult(
        x0=result.x[:4],
        theta=PINQParameters.from_array(result.x[4:]),
        predicted_observations=pred,
        residuals=residual,
        objective=objective,
        jacobian_rank=rank,
        optimizer_status=int(result.status),
        optimizer_message=str(result.message),
        joint_covariance=lap.covariance,
        parameter_standard_errors=lap.standard_errors[4:],
        covariance_rank=lap.rank,
        covariance_condition_number=lap.condition_number,
    )
