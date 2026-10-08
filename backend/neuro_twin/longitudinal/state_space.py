"""Exact linear-Gaussian longitudinal state-space inference for P-I-N-Q.

The canonical P-I-N-Q dynamics are affine-linear,

    dx/dt = A(theta) x + c(theta),

so the continuous-time model admits an exact discrete transition for irregular
observation intervals. This module adds an *inference layer* on top of the
canonical ODE; it does not change the biological model itself.

Continuous process noise is represented by a spectral-density matrix Qc:

    dx = (A x + c) dt + L dW,     L L^T = Qc.

For each irregular interval dt we compute (Phi, q) exactly with matrix
exponentials and then run a Kalman filter followed by a Rauch-Tung-Striebel
(RTS) smoother. Missing observations are handled by selecting only finite
channels at each time point. If all channels are missing, the time point is a
prediction-only step and remains in the trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm, solve_triangular

from neuro_twin.core.observation import LinearObservationModel
from neuro_twin.core.parameters import PINQParameters


_EPS = np.finfo(float).eps


@dataclass(frozen=True)
class Transition:
    phi: np.ndarray  # state transition matrix
    offset: np.ndarray  # affine offset
    q: np.ndarray  # discrete process covariance


@dataclass(frozen=True)
class KalmanFilterResult:
    time: np.ndarray
    predicted_state: np.ndarray
    predicted_covariance: np.ndarray
    filtered_state: np.ndarray
    filtered_covariance: np.ndarray
    transition_matrices: np.ndarray  # (T-1, 4, 4)
    process_covariances: np.ndarray  # (T-1, 4, 4)
    innovations: tuple[np.ndarray | None, ...]
    innovation_covariances: tuple[np.ndarray | None, ...]
    observed_channel_indices: tuple[np.ndarray, ...]
    log_likelihood: float


@dataclass(frozen=True)
class KalmanSmootherResult:
    filtered: KalmanFilterResult
    smoothed_state: np.ndarray
    smoothed_covariance: np.ndarray
    smoothing_gains: np.ndarray  # (T-1, 4, 4)


def _validate_time(time: np.ndarray) -> np.ndarray:
    t = np.asarray(time, dtype=float)
    if t.ndim != 1 or len(t) < 1 or not np.all(np.isfinite(t)):
        raise ValueError("time must be a finite 1-D array")
    if len(t) > 1 and np.any(np.diff(t) <= 0.0):
        raise ValueError("time must be strictly increasing")
    return t


def _validate_covariance(cov: np.ndarray, *, name: str) -> np.ndarray:
    C = np.asarray(cov, dtype=float)
    if C.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4)")
    if not np.all(np.isfinite(C)):
        raise ValueError(f"{name} must be finite")
    C = 0.5 * (C + C.T)
    eig = np.linalg.eigvalsh(C)
    if np.min(eig) < -1e-12:
        raise ValueError(f"{name} must be positive semidefinite")
    return C


def _symmetrize_psd(C: np.ndarray, floor: float = 0.0) -> np.ndarray:
    C = 0.5 * (np.asarray(C, dtype=float) + np.asarray(C, dtype=float).T)
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, floor)
    return (V * w) @ V.T


def _stable_psd_solve(A: np.ndarray, B: np.ndarray, *, rtol: float = 1e-11) -> np.ndarray:
    """Solve A X = B for symmetric PSD A using an eigenvalue floor."""
    A = 0.5 * (np.asarray(A, dtype=float) + np.asarray(A, dtype=float).T)
    w, V = np.linalg.eigh(A)
    scale = max(float(np.max(np.abs(w))), 1.0)
    floor = scale * rtol
    inv = np.where(w > floor, 1.0 / w, 0.0)
    return (V * inv) @ (V.T @ np.asarray(B, dtype=float))


def _affine_transition(A: np.ndarray, c: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact Phi and affine offset for dx/dt = A x + c."""
    M = np.zeros((5, 5), dtype=float)
    M[:4, :4] = A
    M[:4, 4] = c
    E = expm(M * float(dt))
    return E[:4, :4], E[:4, 4]


def _discrete_process_covariance(A: np.ndarray, Qc: np.ndarray, dt: float) -> np.ndarray:
    """Exact Qd = integral exp(A s) Qc exp(A^T s) ds via Van Loan."""
    M = np.zeros((8, 8), dtype=float)
    M[:4, :4] = A
    M[:4, 4:] = Qc
    M[4:, 4:] = -A.T
    E = expm(M * float(dt))
    phi = E[:4, :4]
    upper = E[:4, 4:]
    Qd = upper @ phi.T
    return _symmetrize_psd(Qd)


def exact_transition(theta: PINQParameters, dt: float, process_noise_spectral_density: np.ndarray | None = None) -> Transition:
    """Create the exact discrete transition for an irregular interval."""
    if dt <= 0.0 or not np.isfinite(dt):
        raise ValueError("dt must be finite and > 0")
    A = theta.matrix_A()
    c = theta.vector_c()
    phi, offset = _affine_transition(A, c, dt)
    if process_noise_spectral_density is None:
        q = np.zeros((4, 4), dtype=float)
    else:
        Qc = _validate_covariance(process_noise_spectral_density, name="process_noise_spectral_density")
        q = _discrete_process_covariance(A, Qc, dt)
    return Transition(phi=phi, offset=offset, q=q)


def _select_observations(
    y: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(y)
    idx = np.flatnonzero(finite)
    if len(idx) == 0:
        return idx, H[:0], R[:0, :0]
    return idx, H[idx], R[np.ix_(idx, idx)]


def kalman_filter(
    time: np.ndarray,
    observations: np.ndarray,
    observation_model: LinearObservationModel,
    initial_mean: np.ndarray,
    initial_covariance: np.ndarray,
    theta: PINQParameters,
    *,
    process_noise_spectral_density: np.ndarray | None = None,
) -> KalmanFilterResult:
    """Run an exact continuous-discrete Kalman filter with missing channels."""
    t = _validate_time(time)
    Y = np.asarray(observations, dtype=float)
    if Y.ndim != 2 or Y.shape != (len(t), observation_model.n_observed):
        raise ValueError("observations must have shape (n_time, n_observed)")
    if observation_model.R is None:
        raise ValueError("observation_model.R is required for probabilistic filtering")
    x0 = np.asarray(initial_mean, dtype=float)
    P0 = _validate_covariance(initial_covariance, name="initial_covariance")
    if x0.shape != (4,) or not np.all(np.isfinite(x0)):
        raise ValueError("initial_mean must be finite with shape (4,)")
    if process_noise_spectral_density is not None:
        Qc = _validate_covariance(process_noise_spectral_density, name="process_noise_spectral_density")
    else:
        Qc = None

    T = len(t)
    pred_x = np.empty((T, 4), dtype=float)
    pred_P = np.empty((T, 4, 4), dtype=float)
    filt_x = np.empty((T, 4), dtype=float)
    filt_P = np.empty((T, 4, 4), dtype=float)
    phis = np.empty((max(T - 1, 0), 4, 4), dtype=float)
    qs = np.empty((max(T - 1, 0), 4, 4), dtype=float)
    innovations: list[np.ndarray | None] = []
    innovation_covariances: list[np.ndarray | None] = []
    channel_indices: list[np.ndarray] = []
    loglik = 0.0

    x = x0.copy()
    P = P0.copy()

    for k in range(T):
        if k == 0:
            x_pred, P_pred = x.copy(), P.copy()
        else:
            tr = exact_transition(theta, float(t[k] - t[k - 1]), Qc)
            phis[k - 1] = tr.phi
            qs[k - 1] = tr.q
            x_pred = tr.phi @ x + tr.offset
            P_pred = tr.phi @ P @ tr.phi.T + tr.q
            P_pred = _symmetrize_psd(P_pred)
        pred_x[k] = x_pred
        pred_P[k] = P_pred

        y = Y[k]
        idx, H, R = _select_observations(y, observation_model.H, observation_model.R)
        channel_indices.append(idx)
        if len(idx) == 0:
            x, P = x_pred, P_pred
            innovations.append(None)
            innovation_covariances.append(None)
            filt_x[k], filt_P[k] = x, P
            continue

        b = observation_model.bias[idx] if observation_model.bias is not None else np.zeros(len(idx))
        innov = y[idx] - (H @ x_pred + b)
        S = H @ P_pred @ H.T + R
        S = _symmetrize_psd(S, floor=0.0)
        Sinv = _stable_psd_solve(S, np.eye(len(idx)))
        K = P_pred @ H.T @ Sinv
        x = x_pred + K @ innov

        # Joseph form for numerical PSD preservation.
        I_KH = np.eye(4) - K @ H
        P = I_KH @ P_pred @ I_KH.T + K @ R @ K.T
        P = _symmetrize_psd(P)

        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            raise RuntimeError("innovation covariance is not positive definite")
        maha = float(innov @ Sinv @ innov)
        loglik += -0.5 * (len(idx) * np.log(2.0 * np.pi) + logdet + maha)
        innovations.append(innov.copy())
        innovation_covariances.append(S.copy())
        filt_x[k], filt_P[k] = x, P

    return KalmanFilterResult(
        time=t,
        predicted_state=pred_x,
        predicted_covariance=pred_P,
        filtered_state=filt_x,
        filtered_covariance=filt_P,
        transition_matrices=phis,
        process_covariances=qs,
        innovations=tuple(innovations),
        innovation_covariances=tuple(innovation_covariances),
        observed_channel_indices=tuple(channel_indices),
        log_likelihood=float(loglik),
    )


def rts_smoother(filtered: KalmanFilterResult) -> KalmanSmootherResult:
    """Run a Rauch-Tung-Striebel smoother over a filter result."""
    T = len(filtered.time)
    x_s = filtered.filtered_state.copy()
    P_s = filtered.filtered_covariance.copy()
    gains = np.empty((max(T - 1, 0), 4, 4), dtype=float)

    for k in range(T - 2, -1, -1):
        F = filtered.transition_matrices[k]
        P_f = filtered.filtered_covariance[k]
        P_pred_next = filtered.predicted_covariance[k + 1]
        G = P_f @ F.T @ _stable_psd_solve(P_pred_next, np.eye(4))
        x_s[k] = filtered.filtered_state[k] + G @ (x_s[k + 1] - filtered.predicted_state[k + 1])
        P_s[k] = P_f + G @ (P_s[k + 1] - P_pred_next) @ G.T
        P_s[k] = _symmetrize_psd(P_s[k])
        gains[k] = G
    return KalmanSmootherResult(filtered=filtered, smoothed_state=x_s, smoothed_covariance=P_s, smoothing_gains=gains)


def filter_and_smooth(
    time: np.ndarray,
    observations: np.ndarray,
    observation_model: LinearObservationModel,
    initial_mean: np.ndarray,
    initial_covariance: np.ndarray,
    theta: PINQParameters,
    *,
    process_noise_spectral_density: np.ndarray | None = None,
) -> KalmanSmootherResult:
    """Convenience entry point for longitudinal state reconstruction."""
    filtered = kalman_filter(
        time,
        observations,
        observation_model,
        initial_mean,
        initial_covariance,
        theta,
        process_noise_spectral_density=process_noise_spectral_density,
    )
    return rts_smoother(filtered)
