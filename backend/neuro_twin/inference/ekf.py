"""Reference extended Kalman filter for nonlinear multimodal observations.

The P-I-N-Q transition remains exact. The EKF is used only for the nonlinear
observation update and is therefore explicitly an approximation whose quality
must be benchmarked against synthetic ground truth or a higher-order method.
Persistent Gaussian random effects are represented as static augmented state
components with identity transition dynamics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from neuro_twin.core.parameters import PINQParameters
from neuro_twin.observation.nonlinear import NonlinearObservationModel
from neuro_twin.hierarchical.random_effects import RandomEffectsModel


@dataclass(frozen=True)
class EKFResult:
    time: np.ndarray
    filtered_state: np.ndarray
    filtered_covariance: np.ndarray
    predicted_state: np.ndarray
    predicted_covariance: np.ndarray
    log_likelihood: float


def _affine_transition(theta: PINQParameters, dt: float):
    A, c = theta.matrix_A(), theta.vector_c()
    M = np.zeros((5, 5), float)
    M[:4, :4] = A
    M[:4, 4] = c
    E = expm(M * dt)
    return E[:4, :4], E[:4, 4]


def _inv_spd(S):
    return np.linalg.inv(0.5 * (S + S.T))


def ekf_filter(
    time: np.ndarray,
    observations: np.ndarray,
    observation_model: NonlinearObservationModel,
    initial_state_mean: np.ndarray,
    initial_state_covariance: np.ndarray,
    theta: PINQParameters,
    *,
    process_noise_spectral_density: np.ndarray | None = None,
    random_effects: RandomEffectsModel | None = None,
) -> EKFResult:
    t = np.asarray(time, float)
    Y = np.asarray(observations, float)
    if Y.shape != (len(t), observation_model.n_observed):
        raise ValueError("observations shape mismatch")
    if np.any(np.diff(t) <= 0):
        raise ValueError("time must be strictly increasing")
    if observation_model.n_observed == 0:
        raise ValueError("no observation channels")
    theta_arr = theta.as_array()
    n_u = 0 if random_effects is None else random_effects.n_latent
    n = 4 + n_u
    m0 = np.asarray(initial_state_mean, float)
    if m0.shape != (n,):
        raise ValueError("initial_state_mean dimension mismatch")
    P = np.asarray(initial_state_covariance, float)
    if P.shape != (n, n):
        raise ValueError("initial_state_covariance dimension mismatch")
    if process_noise_spectral_density is None:
        Qc = np.zeros((4, 4))
    else:
        Qc = np.asarray(process_noise_spectral_density, float)
    if Qc.shape != (4, 4):
        raise ValueError("process_noise_spectral_density must be 4x4")

    pred_m = np.empty((len(t), n))
    pred_P = np.empty((len(t), n, n))
    filt_m = np.empty_like(pred_m)
    filt_P = np.empty_like(pred_P)
    m, Pcur = m0.copy(), 0.5 * (P + P.T)
    ll = 0.0

    for k in range(len(t)):
        if k == 0:
            mp, Pp = m, Pcur
        else:
            Phi, offset = _affine_transition(theta, float(t[k] - t[k - 1]))
            F = np.eye(n)
            F[:4, :4] = Phi
            q = np.zeros((n, n))
            q[:4, :4] = Qc
            mp = F @ m
            mp[:4] += offset
            Pp = F @ Pcur @ F.T + q
            Pp = 0.5 * (Pp + Pp.T)
        pred_m[k], pred_P[k] = mp, Pp

        y = Y[k]
        idx = np.flatnonzero(np.isfinite(y))
        if len(idx) == 0:
            m, Pcur = mp, Pp
            filt_m[k], filt_P[k] = m, Pcur
            continue

        x = mp[:4]
        h = observation_model.predict(x, theta_arr)
        Hx = observation_model.jacobian_state(x, theta_arr)
        if n_u:
            Z = random_effects.design
            Haug = np.concatenate([Hx, Z], axis=1)
        else:
            Haug = Hx
        bidx = observation_model.R[np.ix_(idx, idx)]
        innov = y[idx] - h[idx]
        Hobs = Haug[idx]
        S = Hobs @ Pp @ Hobs.T + bidx
        S = 0.5 * (S + S.T)
        Sinv = _inv_spd(S)
        K = Pp @ Hobs.T @ Sinv
        m = mp + K @ innov
        I_KH = np.eye(n) - K @ Hobs
        Pcur = I_KH @ Pp @ I_KH.T + K @ bidx @ K.T
        Pcur = 0.5 * (Pcur + Pcur.T)
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            raise RuntimeError("innovation covariance not positive definite")
        ll += -0.5 * (len(idx) * np.log(2*np.pi) + logdet + innov @ Sinv @ innov)
        filt_m[k], filt_P[k] = m, Pcur

    return EKFResult(t, filt_m, filt_P, pred_m, pred_P, float(ll))
