"""Differentiable exact continuous-discrete likelihood for P-I-N-Q.

This module is optional and uses JAX.  It does not define a new biological
model: it differentiates the exact affine transition and the exact
continuous-time process-noise discretisation already used by the reference
SciPy state-space engine.

The implementation deliberately keeps the observation operator linear for the
current P-I-N-Q contract.  Missing channels are represented through a mask and
are integrated out rather than imputed.
"""
from __future__ import annotations

from typing import Any

import numpy as np

try:  # Optional dependency.
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import expm

    jax.config.update("jax_enable_x64", True)
    JAX_AVAILABLE = True
except Exception:  # pragma: no cover - exercised when optional deps absent.
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    expm = None  # type: ignore[assignment]
    JAX_AVAILABLE = False


PARAMETER_COUNT = 11
STATE_DIM = 4
_MISSING_VARIANCE = 1.0e12


def require_jax() -> None:
    if not JAX_AVAILABLE:
        raise ImportError(
            "JAX is required for the differentiable backend. Install the "
            "'science' extras of neuro-twin."
        )


def theta_to_A_c(theta: Any):
    """Return JAX A(theta), c(theta) for the canonical P-I-N-Q model."""
    require_jax()
    t = jnp.asarray(theta)
    if t.shape != (PARAMETER_COUNT,):
        raise ValueError("theta must have shape (11,)")
    aP, bPI, dP, bIP, dI, bNP, bNI, dN, bQN, bQI, dQ = t
    A = jnp.array(
        [
            [-dP, bPI, 0.0, 0.0],
            [bIP, -dI, 0.0, 0.0],
            [bNP, bNI, -dN, 0.0],
            [0.0, bQI, bQN, -dQ],
        ],
        dtype=t.dtype,
    )
    c = jnp.array([aP, 0.0, 0.0, 0.0], dtype=t.dtype)
    return A, c


def _affine_transition(A, c, dt):
    M = jnp.zeros((5, 5), dtype=A.dtype)
    M = M.at[:4, :4].set(A)
    M = M.at[:4, 4].set(c)
    E = expm(M * dt)
    return E[:4, :4], E[:4, 4]


def _process_covariance(A, Qc, dt):
    M = jnp.zeros((8, 8), dtype=A.dtype)
    M = M.at[:4, :4].set(A)
    M = M.at[:4, 4:].set(Qc)
    M = M.at[4:, 4:].set(-A.T)
    E = expm(M * dt)
    phi = E[:4, :4]
    Qd = E[:4, 4:] @ phi.T
    return 0.5 * (Qd + Qd.T)


def _sym_psd(P):
    return 0.5 * (P + P.T)


def _masked_measurement(y, mask, H, R):
    """Build a full-size measurement system whose missing block is inert.

    The returned log-likelihood later removes the constant contribution from
    the artificial missing variances. The observed block is exactly H_oo/R_oo.
    """
    D = jnp.diag(mask)
    Hm = D @ H
    Rm = D @ R @ D + jnp.diag((1.0 - mask) * _MISSING_VARIANCE)
    ym = jnp.where(mask > 0.5, y, 0.0)
    return ym, Hm, Rm


def kalman_loglik_jax(
    time,
    observations,
    H,
    R,
    initial_mean,
    initial_covariance,
    theta,
    *,
    process_noise_spectral_density=None,
):
    """Differentiable exact Kalman marginal log-likelihood.

    Missing channels are integrated out by using the corresponding observed
    principal submatrix of R. No missing value is imputed.
    """
    require_jax()
    t = jnp.asarray(time, dtype=jnp.float64)
    Y = jnp.asarray(observations, dtype=jnp.float64)
    finite = jnp.isfinite(Y)
    Y = jnp.where(finite, Y, 0.0)
    mask = finite.astype(jnp.float64)
    H = jnp.asarray(H, dtype=jnp.float64)
    R = jnp.asarray(R, dtype=jnp.float64)
    x0 = jnp.asarray(initial_mean, dtype=jnp.float64)
    P0 = jnp.asarray(initial_covariance, dtype=jnp.float64)
    theta = jnp.asarray(theta, dtype=jnp.float64)
    Qc = jnp.zeros((4, 4), dtype=jnp.float64) if process_noise_spectral_density is None else jnp.asarray(process_noise_spectral_density, dtype=jnp.float64)

    if Y.ndim != 2 or H.shape[0] != Y.shape[1]:
        raise ValueError("observations and H dimensions are inconsistent")
    if Y.shape[0] != t.shape[0]:
        raise ValueError("time and observations lengths differ")

    A, c = theta_to_A_c(theta)

    def step(carry, data):
        x_prev, P_prev, loglik = carry
        k, t_k, y_k, m_k = data

        def propagate(_):
            dt = t_k - t[k - 1]
            phi, offset = _affine_transition(A, c, dt)
            q = _process_covariance(A, Qc, dt)
            xp = phi @ x_prev + offset
            Pp = _sym_psd(phi @ P_prev @ phi.T + q)
            return xp, Pp

        xp, Pp = jax.lax.cond(
            k == 0,
            lambda _: (x_prev, P_prev),
            propagate,
            operand=None,
        )

        ym, Hm, Rm = _masked_measurement(y_k, m_k, H, R)
        innov = ym - Hm @ xp
        S = _sym_psd(Hm @ Pp @ Hm.T + Rm)
        L = jnp.linalg.cholesky(S)
        alpha = jax.scipy.linalg.solve_triangular(L, innov, lower=True)
        Sinv_innov = jax.scipy.linalg.solve_triangular(L.T, alpha, lower=False)
        K = Pp @ Hm.T @ jax.scipy.linalg.solve_triangular(L.T, jax.scipy.linalg.solve_triangular(L, jnp.eye(H.shape[0]), lower=True), lower=False)
        x_new = xp + K @ innov
        IKH = jnp.eye(4) - K @ Hm
        P_new = _sym_psd(IKH @ Pp @ IKH.T + K @ Rm @ K.T)

        logdet_full = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        maha = innov @ Sinv_innov
        n_miss = H.shape[0] - jnp.sum(m_k)
        # Remove the artificial missing-dimensional contribution. This leaves
        # exactly the likelihood of the observed principal subvector.
        constant_missing = n_miss * (jnp.log(2.0 * jnp.pi) + jnp.log(_MISSING_VARIANCE))
        increment = -0.5 * (H.shape[0] * jnp.log(2.0 * jnp.pi) + logdet_full + maha - constant_missing)
        return (x_new, P_new, loglik + increment), increment

    ks = jnp.arange(t.shape[0], dtype=jnp.int32)
    (x_last, P_last, loglik), increments = jax.lax.scan(step, (x0, P0, 0.0), (ks, t, Y, mask))
    return loglik


def negative_loglik_and_grad(
    time,
    observations,
    H,
    R,
    initial_mean,
    initial_covariance,
    *,
    process_noise_spectral_density=None,
):
    """Return a JIT-compatible scalar objective and gradient function."""
    require_jax()

    def objective(z):
        z = jnp.asarray(z, dtype=jnp.float64)
        x0 = z[:4]
        theta = z[4:]
        ll = kalman_loglik_jax(
            time,
            observations,
            H,
            R,
            x0,
            initial_covariance,
            theta,
            process_noise_spectral_density=process_noise_spectral_density,
        )
        return -ll

    value_and_grad = jax.jit(jax.value_and_grad(objective))

    def wrapped(z_np: np.ndarray) -> tuple[float, np.ndarray]:
        value, grad = value_and_grad(jnp.asarray(z_np, dtype=jnp.float64))
        return float(value), np.asarray(grad, dtype=float)

    return wrapped
