"""Continuous-discrete linear-Gaussian inference for the six-domain network.

Filtering is forward-only and suitable as a building block for prospective
PIT pipelines. RTS smoothing is retrospective: it uses later observations and
must not be used to populate a prospective live posterior.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from neuro_twin.network.model import SixDomainNetwork
from neuro_twin.network.observation import LinearNetworkObservationModel


_STATE_DIM = 6


@dataclass(frozen=True)
class NetworkTransition:
    phi: np.ndarray
    offset: np.ndarray
    q: np.ndarray


@dataclass(frozen=True)
class NetworkFilterResult:
    time: np.ndarray
    predicted_state: np.ndarray
    predicted_covariance: np.ndarray
    filtered_state: np.ndarray
    filtered_covariance: np.ndarray
    transition_matrices: np.ndarray
    process_covariances: np.ndarray
    innovations: tuple[np.ndarray | None, ...]
    innovation_covariances: tuple[np.ndarray | None, ...]
    observed_channel_indices: tuple[np.ndarray, ...]
    log_likelihood: float


@dataclass(frozen=True)
class NetworkSmootherResult:
    filtered: NetworkFilterResult
    smoothed_state: np.ndarray
    smoothed_covariance: np.ndarray
    smoothing_gains: np.ndarray


def _validate_time(time: np.ndarray) -> np.ndarray:
    t = np.asarray(time, dtype=float)
    if t.ndim != 1 or len(t) < 1 or not np.all(np.isfinite(t)):
        raise ValueError("time must be a finite 1-D array with at least one point")
    if len(t) > 1 and np.any(np.diff(t) <= 0.0):
        raise ValueError("time must be strictly increasing")
    return t


def _validate_covariance(value: np.ndarray, *, name: str) -> np.ndarray:
    cov = np.asarray(value, dtype=float)
    if cov.shape != (_STATE_DIM, _STATE_DIM) or not np.all(np.isfinite(cov)):
        raise ValueError(f"{name} must be a finite 6x6 matrix")
    if not np.allclose(cov, cov.T, atol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    cov = 0.5 * (cov + cov.T)
    if float(np.min(np.linalg.eigvalsh(cov))) < -1e-10:
        raise ValueError(f"{name} must be positive semidefinite")
    return cov


def _symmetrize_psd(value: np.ndarray) -> np.ndarray:
    cov = 0.5 * (np.asarray(value, dtype=float) + np.asarray(value, dtype=float).T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    if float(np.min(eigenvalues)) < -1e-8:
        raise RuntimeError("covariance lost positive-semidefinite property")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return (eigenvectors * eigenvalues) @ eigenvectors.T


def _stable_psd_solve(matrix: np.ndarray, rhs: np.ndarray, *, rtol: float = 1e-12) -> np.ndarray:
    matrix = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(np.min(eigenvalues)) < -1e-8:
        raise RuntimeError("matrix expected to be positive semidefinite")
    inverse_values = np.where(eigenvalues > scale * rtol, 1.0 / np.maximum(eigenvalues, scale * rtol), 0.0)
    return (eigenvectors * inverse_values) @ (eigenvectors.T @ np.asarray(rhs, dtype=float))


def _discrete_process_covariance(A: np.ndarray, Qc: np.ndarray, dt: float, phi: np.ndarray) -> np.ndarray:
    """Van Loan discretization of continuous-time process noise."""
    block = np.zeros((12, 12), dtype=float)
    block[:6, :6] = A
    block[:6, 6:] = Qc
    block[6:, 6:] = -A.T
    exponential = expm(block * float(dt))
    qd = exponential[:6, 6:] @ phi.T
    return _symmetrize_psd(qd)


def exact_network_transition(
    network: SixDomainNetwork,
    dt: float,
    process_noise_spectral_density: np.ndarray | None = None,
) -> NetworkTransition:
    """Create exact irregular-interval transition and discrete process noise."""
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and > 0")
    augmented = np.zeros((7, 7), dtype=float)
    augmented[:6, :6] = network.coupling_matrix
    augmented[:6, 6] = network.drift
    exponential = expm(augmented * float(dt))
    phi = exponential[:6, :6]
    offset = exponential[:6, 6]
    if process_noise_spectral_density is None:
        q = np.zeros((6, 6), dtype=float)
    else:
        qc = _validate_covariance(
            process_noise_spectral_density,
            name="process_noise_spectral_density",
        )
        q = _discrete_process_covariance(network.coupling_matrix, qc, float(dt), phi)
    return NetworkTransition(phi=phi, offset=offset, q=q)


def kalman_filter_network(
    time: np.ndarray,
    observations: np.ndarray,
    observation_model: LinearNetworkObservationModel,
    initial_mean: np.ndarray,
    initial_covariance: np.ndarray,
    network: SixDomainNetwork,
    *,
    process_noise_spectral_density: np.ndarray | None = None,
) -> NetworkFilterResult:
    """Run a forward continuous-discrete Kalman filter with missing channels.

    NaN means unavailable/missing. Infinite observation values are rejected.
    The filter never uses observations later than the current time index.
    """
    t = _validate_time(time)
    values = np.asarray(observations, dtype=float)
    if values.shape != (len(t), observation_model.n_observed):
        raise ValueError("observations must have shape (n_time, n_observed)")
    if np.any(np.isinf(values)):
        raise ValueError("observations may contain NaN for missing values, not infinity")

    mean = np.asarray(initial_mean, dtype=float)
    if mean.shape != (_STATE_DIM,) or not np.all(np.isfinite(mean)):
        raise ValueError("initial_mean must be a finite vector with shape (6,)")
    covariance = _validate_covariance(initial_covariance, name="initial_covariance")
    qc = None
    if process_noise_spectral_density is not None:
        qc = _validate_covariance(
            process_noise_spectral_density,
            name="process_noise_spectral_density",
        )

    n_time = len(t)
    predicted_state = np.empty((n_time, _STATE_DIM), dtype=float)
    predicted_covariance = np.empty((n_time, _STATE_DIM, _STATE_DIM), dtype=float)
    filtered_state = np.empty_like(predicted_state)
    filtered_covariance = np.empty_like(predicted_covariance)
    transitions = np.empty((max(n_time - 1, 0), _STATE_DIM, _STATE_DIM), dtype=float)
    process_covariances = np.empty_like(transitions)
    innovations: list[np.ndarray | None] = []
    innovation_covariances: list[np.ndarray | None] = []
    observed_indices: list[np.ndarray] = []
    log_likelihood = 0.0

    x = mean.copy()
    p = covariance.copy()
    identity = np.eye(_STATE_DIM)

    for index in range(n_time):
        if index == 0:
            x_pred = x.copy()
            p_pred = p.copy()
        else:
            transition = exact_network_transition(
                network,
                float(t[index] - t[index - 1]),
                qc,
            )
            transitions[index - 1] = transition.phi
            process_covariances[index - 1] = transition.q
            x_pred = transition.phi @ x + transition.offset
            p_pred = _symmetrize_psd(
                transition.phi @ p @ transition.phi.T + transition.q
            )

        predicted_state[index] = x_pred
        predicted_covariance[index] = p_pred

        available = np.flatnonzero(np.isfinite(values[index]))
        observed_indices.append(available)
        if len(available) == 0:
            x, p = x_pred, p_pred
            innovations.append(None)
            innovation_covariances.append(None)
            filtered_state[index] = x
            filtered_covariance[index] = p
            continue

        H = observation_model.H[available]
        R = observation_model.R[np.ix_(available, available)]
        bias = observation_model.bias[available]
        innovation = values[index, available] - (H @ x_pred + bias)
        innovation_cov = _symmetrize_psd(H @ p_pred @ H.T + R)
        inverse_innovation_cov = _stable_psd_solve(
            innovation_cov,
            np.eye(len(available)),
        )
        gain = p_pred @ H.T @ inverse_innovation_cov
        x = x_pred + gain @ innovation

        # Joseph form limits loss of covariance positive semidefiniteness.
        residual_operator = identity - gain @ H
        p = _symmetrize_psd(
            residual_operator @ p_pred @ residual_operator.T
            + gain @ R @ gain.T
        )

        sign, logdet = np.linalg.slogdet(innovation_cov)
        if sign <= 0:
            raise RuntimeError("innovation covariance is not positive definite")
        mahalanobis = float(innovation @ inverse_innovation_cov @ innovation)
        log_likelihood += -0.5 * (
            len(available) * np.log(2.0 * np.pi) + logdet + mahalanobis
        )

        innovations.append(innovation.copy())
        innovation_covariances.append(innovation_cov.copy())
        filtered_state[index] = x
        filtered_covariance[index] = p

    return NetworkFilterResult(
        time=t.copy(),
        predicted_state=predicted_state,
        predicted_covariance=predicted_covariance,
        filtered_state=filtered_state,
        filtered_covariance=filtered_covariance,
        transition_matrices=transitions,
        process_covariances=process_covariances,
        innovations=tuple(innovations),
        innovation_covariances=tuple(innovation_covariances),
        observed_channel_indices=tuple(observed_indices),
        log_likelihood=float(log_likelihood),
    )


def rts_smoother_network(filtered: NetworkFilterResult) -> NetworkSmootherResult:
    """Retrospective Rauch-Tung-Striebel smoother for a completed sequence."""
    n_time = len(filtered.time)
    smoothed_state = filtered.filtered_state.copy()
    smoothed_covariance = filtered.filtered_covariance.copy()
    gains = np.empty((max(n_time - 1, 0), _STATE_DIM, _STATE_DIM), dtype=float)

    for index in range(n_time - 2, -1, -1):
        phi = filtered.transition_matrices[index]
        p_filtered = filtered.filtered_covariance[index]
        p_pred_next = filtered.predicted_covariance[index + 1]
        gain = p_filtered @ phi.T @ _stable_psd_solve(
            p_pred_next,
            np.eye(_STATE_DIM),
        )
        gains[index] = gain
        smoothed_state[index] = (
            filtered.filtered_state[index]
            + gain @ (
                smoothed_state[index + 1]
                - filtered.predicted_state[index + 1]
            )
        )
        smoothed_covariance[index] = _symmetrize_psd(
            p_filtered
            + gain
            @ (smoothed_covariance[index + 1] - p_pred_next)
            @ gain.T
        )

    return NetworkSmootherResult(
        filtered=filtered,
        smoothed_state=smoothed_state,
        smoothed_covariance=smoothed_covariance,
        smoothing_gains=gains,
    )


def filter_and_smooth_network(
    time: np.ndarray,
    observations: np.ndarray,
    observation_model: LinearNetworkObservationModel,
    initial_mean: np.ndarray,
    initial_covariance: np.ndarray,
    network: SixDomainNetwork,
    *,
    process_noise_spectral_density: np.ndarray | None = None,
) -> NetworkSmootherResult:
    """Convenience wrapper; use only for retrospective analysis."""
    filtered = kalman_filter_network(
        time,
        observations,
        observation_model,
        initial_mean,
        initial_covariance,
        network,
        process_noise_spectral_density=process_noise_spectral_density,
    )
    return rts_smoother_network(filtered)
