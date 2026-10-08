"""Unscented Kalman filter for nonlinear Neuro-Twin observations.

The P-I-N-Q state transition remains exact affine propagation. The UKF is used
for the nonlinear observation transform (and optional persistent random
 effects), providing a higher-order alternative to the EKF without finite
 differences. It is still an approximation and must be benchmarked.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from neuro_twin.core.parameters import PINQParameters
from neuro_twin.hierarchical.random_effects import RandomEffectsModel
from neuro_twin.observation.nonlinear import NonlinearObservationModel


@dataclass(frozen=True)
class UKFResult:
    time: np.ndarray
    filtered_state: np.ndarray
    filtered_covariance: np.ndarray
    predicted_state: np.ndarray
    predicted_covariance: np.ndarray
    log_likelihood: float


def _affine_transition(theta: PINQParameters, dt: float) -> tuple[np.ndarray, np.ndarray]:
    A, c = theta.matrix_A(), theta.vector_c()
    M = np.zeros((5, 5), float)
    M[:4, :4] = A
    M[:4, 4] = c
    E = expm(M * dt)
    return E[:4, :4], E[:4, 4]


def _chol_psd(P: np.ndarray, jitter: float = 1e-10) -> np.ndarray:
    P = 0.5 * (P + P.T)
    try:
        return np.linalg.cholesky(P)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(P)
        vals = np.maximum(vals, jitter)
        return vecs @ np.diag(np.sqrt(vals))


def _sigma_points(m: np.ndarray, P: np.ndarray, alpha: float, beta: float, kappa: float):
    n = m.size
    lam = alpha * alpha * (n + kappa) - n
    c = n + lam
    if c <= 0:
        raise ValueError("UKF scaling requires n + lambda > 0")
    S = _chol_psd(c * P)
    X = np.empty((2 * n + 1, n), float)
    X[0] = m
    X[1:n + 1] = m + S.T
    X[n + 1:] = m - S.T
    Wm = np.full(2 * n + 1, 1.0 / (2.0 * c), float)
    Wc = Wm.copy()
    Wm[0] = lam / c
    Wc[0] = lam / c + (1.0 - alpha * alpha + beta)
    return X, Wm, Wc


def ukf_filter(
    time: np.ndarray,
    observations: np.ndarray,
    observation_model: NonlinearObservationModel,
    initial_state_mean: np.ndarray,
    initial_state_covariance: np.ndarray,
    theta: PINQParameters,
    *,
    process_noise_spectral_density: np.ndarray | None = None,
    random_effects: RandomEffectsModel | None = None,
    alpha: float = 0.35,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> UKFResult:
    t = np.asarray(time, float)
    Y = np.asarray(observations, float)
    if Y.shape != (len(t), observation_model.n_observed):
        raise ValueError("observations shape mismatch")
    if np.any(np.diff(t) <= 0):
        raise ValueError("time must be strictly increasing")
    n_u = 0 if random_effects is None else random_effects.n_latent
    n = 4 + n_u
    m = np.asarray(initial_state_mean, float)
    P = 0.5 * (np.asarray(initial_state_covariance, float) + np.asarray(initial_state_covariance, float).T)
    if m.shape != (n,) or P.shape != (n, n):
        raise ValueError("initial state dimensions mismatch")
    Qc = np.zeros((4, 4), float) if process_noise_spectral_density is None else np.asarray(process_noise_spectral_density, float)
    if Qc.shape != (4, 4):
        raise ValueError("process_noise_spectral_density must be 4x4")

    theta_arr = theta.as_array()
    pred_m = np.empty((len(t), n))
    pred_P = np.empty((len(t), n, n))
    filt_m = np.empty_like(pred_m)
    filt_P = np.empty_like(pred_P)
    ll = 0.0

    def transition(X: np.ndarray, dt: float) -> np.ndarray:
        Phi, off = _affine_transition(theta, dt)
        out = X.copy()
        out[:, :4] = X[:, :4] @ Phi.T + off
        return out

    def observe(X: np.ndarray) -> np.ndarray:
        vals = np.vstack([observation_model.predict(row[:4], theta_arr) for row in X])
        if n_u:
            vals = vals + X[:, 4:] @ random_effects.design.T
        return vals

    for k in range(len(t)):
        if k == 0:
            mp, Pp = m, P
        else:
            X, Wm, Wc = _sigma_points(m, P, alpha, beta, kappa)
            Xp = transition(X, float(t[k] - t[k - 1]))
            mp = np.sum(Wm[:, None] * Xp, axis=0)
            dX = Xp - mp
            Pp = np.einsum("i,ij,ik->jk", Wc, dX, dX)
            q = np.zeros((n, n), float)
            # Van Loan process-noise discretisation re-used through an
            # equivalent sigma-point prediction covariance baseline.
            # Qc is treated as a continuous spectral density over dt.
            q[:4, :4] = Qc * float(t[k] - t[k - 1])
            Pp = 0.5 * (Pp + Pp.T) + q
        pred_m[k], pred_P[k] = mp, Pp

        y = Y[k]
        idx = np.flatnonzero(np.isfinite(y))
        if len(idx) == 0:
            m, P = mp, Pp
            filt_m[k], filt_P[k] = m, P
            continue

        X, Wm, Wc = _sigma_points(mp, Pp, alpha, beta, kappa)
        Ysig = observe(X)[:, idx]
        ybar = np.sum(Wm[:, None] * Ysig, axis=0)
        dY = Ysig - ybar
        dX = X - mp
        S = np.einsum("i,ij,ik->jk", Wc, dY, dY)
        R = observation_model.R[np.ix_(idx, idx)]
        S = 0.5 * (S + S.T) + R
        Cxy = np.einsum("i,ij,ik->jk", Wc, dX, dY)
        S = 0.5 * (S + S.T)
        L = _chol_psd(S)
        innov = y[idx] - ybar
        alpha_innov = np.linalg.solve(L, innov)
        Sinv_innov = np.linalg.solve(L.T, alpha_innov)
        Sinv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(len(idx))))
        K = Cxy @ Sinv
        m = mp + K @ innov
        P = Pp - K @ S @ K.T
        P = 0.5 * (P + P.T)
        # Cholesky is the definitive SPD check; small numerical negatives are
        # clipped only through the symmetric eigendecomposition when needed.
        vals, vecs = np.linalg.eigh(P)
        if vals.min() < -1e-8:
            raise RuntimeError("UKF produced a non-PSD covariance")
        if vals.min() < 0:
            P = vecs @ np.diag(np.maximum(vals, 1e-12)) @ vecs.T
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        ll += -0.5 * (len(idx) * np.log(2.0 * np.pi) + logdet + innov @ Sinv_innov)
        filt_m[k], filt_P[k] = m, P

    return UKFResult(t, filt_m, filt_P, pred_m, pred_P, float(ll))
