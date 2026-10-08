"""Joint hierarchical MAP over raw longitudinal observations.

Scientific boundary
-------------------
This backend is a *joint MAP approximation built on the nonlinear EKF
marginal likelihood*.  It is not an exact Bayesian posterior for the nonlinear
state-space model.  The hierarchy is expressed directly on subject-specific
ODE parameters and initial latent states:

    y_s | x0_s, theta_s ~ EKF-marginal likelihood
    theta_s | mu_c      ~ N(mu_c, Sigma_within)
    mu_c | mu0          ~ N(mu0, Sigma_cohort)
    mu0                  ~ N(mu_prior, Sigma_global)
    x0_s                 ~ N(x0_prior, Sigma_x0)

The module uses JAX autodiff for gradients and Hessians and SciPy L-BFGS-B for
optimization.  This gives a deterministic reference backend before optional
NUTS/variational implementations are introduced.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

from neuro_twin.inference.jax_ekf import JAX_AVAILABLE, ekf_loglik_jax


PARAMETER_COUNT = 11
STATE_DIM = 4


@dataclass(frozen=True)
class SubjectSeries:
    subject_id: str
    cohort: str
    time: np.ndarray
    observations: np.ndarray
    observation_specs: tuple[dict[str, Any], ...]
    R: np.ndarray
    initial_state_prior: np.ndarray
    initial_state_covariance: np.ndarray
    initial_state_mean_for_filter: np.ndarray
    process_noise_spectral_density: np.ndarray | None = None
    random_effect_design: np.ndarray | None = None
    random_effect_covariance: np.ndarray | None = None

    def __post_init__(self) -> None:
        t = np.asarray(self.time, dtype=float)
        y = np.asarray(self.observations, dtype=float)
        R = np.asarray(self.R, dtype=float)
        m = np.asarray(self.initial_state_prior, dtype=float)
        C = np.asarray(self.initial_state_covariance, dtype=float)
        xf = np.asarray(self.initial_state_mean_for_filter, dtype=float)
        if t.ndim != 1 or len(t) < 1 or np.any(np.diff(t) < 0):
            raise ValueError("time must be a non-empty nondecreasing 1-D array")
        if y.ndim != 2 or y.shape[0] != len(t):
            raise ValueError("observations must have shape (n_time, n_obs)")
        if R.shape != (y.shape[1], y.shape[1]):
            raise ValueError("R shape mismatch")
        if m.shape != (STATE_DIM,) or xf.shape != (STATE_DIM,):
            raise ValueError("initial state means must have shape (4,)")
        if C.shape != (STATE_DIM, STATE_DIM):
            raise ValueError("initial_state_covariance must be 4x4")
        if np.min(np.linalg.eigvalsh(0.5 * (C + C.T))) <= 0:
            raise ValueError("initial_state_covariance must be positive definite")
        if self.random_effect_design is not None:
            Z = np.asarray(self.random_effect_design, dtype=float)
            if Z.ndim != 2 or Z.shape[0] != y.shape[1]:
                raise ValueError("random_effect_design must be (n_obs, n_random_effects)")
            if self.random_effect_covariance is None:
                raise ValueError("random_effect_covariance is required when random_effect_design is provided")
            U = np.asarray(self.random_effect_covariance, dtype=float)
            if U.shape != (Z.shape[1], Z.shape[1]):
                raise ValueError("random_effect_covariance shape mismatch")
            if np.min(np.linalg.eigvalsh(0.5*(U+U.T))) <= 0:
                raise ValueError("random_effect_covariance must be positive definite")
            object.__setattr__(self, "random_effect_design", Z)
            object.__setattr__(self, "random_effect_covariance", 0.5*(U+U.T))
        object.__setattr__(self, "time", t)
        object.__setattr__(self, "observations", y)
        object.__setattr__(self, "R", 0.5 * (R + R.T))
        object.__setattr__(self, "initial_state_prior", m)
        object.__setattr__(self, "initial_state_covariance", 0.5 * (C + C.T))
        object.__setattr__(self, "initial_state_mean_for_filter", xf)


@dataclass(frozen=True)
class HierarchicalMAPResult:
    success: bool
    message: str
    objective: float
    global_mean: np.ndarray
    cohort_means: dict[str, np.ndarray]
    subject_theta: dict[str, np.ndarray]
    subject_x0: dict[str, np.ndarray]
    map_vector: np.ndarray
    laplace_covariance: np.ndarray | None
    gradient_norm: float
    n_iter: int


def _log_gaussian(value: Any, mean: Any, cov_inv: Any, logdet_cov: Any):
    import jax.numpy as jnp

    d = value.shape[-1]
    diff = value - mean
    quad = diff @ cov_inv @ diff
    return 0.5 * (d * jnp.log(2.0 * jnp.pi) + logdet_cov + quad)


def fit_joint_hierarchical_map(
    subjects: list[SubjectSeries],
    *,
    theta_mean_prior: np.ndarray,
    theta_global_covariance: np.ndarray,
    theta_cohort_covariance: np.ndarray,
    theta_within_covariance: np.ndarray,
    x0_prior_covariance: np.ndarray,
    initial_theta: np.ndarray | None = None,
    initial_x0: np.ndarray | None = None,
    maxiter: int = 300,
    gtol: float = 1e-6,
) -> HierarchicalMAPResult:
    """Fit a joint subject/cohort/global MAP using the nonlinear EKF likelihood."""
    if not JAX_AVAILABLE:
        raise ImportError("JAX is required for hierarchical joint MAP")
    if not subjects:
        raise ValueError("at least one subject is required")

    import jax
    import jax.numpy as jnp

    mu_prior = np.asarray(theta_mean_prior, dtype=float)
    G = np.asarray(theta_global_covariance, dtype=float)
    C = np.asarray(theta_cohort_covariance, dtype=float)
    W = np.asarray(theta_within_covariance, dtype=float)
    X = np.asarray(x0_prior_covariance, dtype=float)
    for name, A, shape in (
        ("theta_mean_prior", mu_prior, (PARAMETER_COUNT,)),
        ("theta_global_covariance", G, (PARAMETER_COUNT, PARAMETER_COUNT)),
        ("theta_cohort_covariance", C, (PARAMETER_COUNT, PARAMETER_COUNT)),
        ("theta_within_covariance", W, (PARAMETER_COUNT, PARAMETER_COUNT)),
        ("x0_prior_covariance", X, (STATE_DIM, STATE_DIM)),
    ):
        if A.shape != shape:
            raise ValueError(f"{name} has invalid shape")
        if not np.all(np.isfinite(A)):
            raise ValueError(f"{name} contains non-finite values")
    for name, A in (("G", G), ("C", C), ("W", W), ("X", X)):
        if np.min(np.linalg.eigvalsh(0.5 * (A + A.T))) <= 0:
            raise ValueError(f"{name} must be positive definite")

    cohort_names = tuple(sorted({s.cohort for s in subjects}))
    cohort_index = {g: i for i, g in enumerate(cohort_names)}
    ns = len(subjects)
    ng = len(cohort_names)

    theta0 = mu_prior if initial_theta is None else np.asarray(initial_theta, dtype=float)
    x00 = np.asarray(subjects[0].initial_state_prior if initial_x0 is None else initial_x0, dtype=float)
    if theta0.shape != (PARAMETER_COUNT,) or x00.shape != (STATE_DIM,):
        raise ValueError("initial_theta/initial_x0 dimensions are invalid")

    # Layout: global mean, cohort means, [subject x0, subject theta] * ns.
    pieces = [theta0, *([theta0] * ng)]
    for s in subjects:
        pieces.extend([x00, theta0])
    z0 = np.concatenate(pieces).astype(float)

    G_inv = np.linalg.inv(G)
    C_inv = np.linalg.inv(C)
    W_inv = np.linalg.inv(W)
    X_inv = np.linalg.inv(X)
    logdet_G = float(np.linalg.slogdet(G)[1])
    logdet_C = float(np.linalg.slogdet(C)[1])
    logdet_W = float(np.linalg.slogdet(W)[1])
    logdet_X = float(np.linalg.slogdet(X)[1])

    # Freeze per-subject numpy arrays in closure; these are static across calls.
    static = []
    for s in subjects:
        Z = None if s.random_effect_design is None else s.random_effect_design
        static.append(
            (
                jnp.asarray(s.time, dtype=jnp.float64),
                jnp.asarray(s.observations, dtype=jnp.float64),
                s.observation_specs,
                jnp.asarray(s.R, dtype=jnp.float64),
                jnp.asarray(s.initial_state_mean_for_filter, dtype=jnp.float64),
                jnp.asarray(s.initial_state_covariance, dtype=jnp.float64),
                None if s.process_noise_spectral_density is None else jnp.asarray(s.process_noise_spectral_density, dtype=jnp.float64),
                None if Z is None else jnp.asarray(Z, dtype=jnp.float64),
                None if s.random_effect_covariance is None else jnp.asarray(s.random_effect_covariance, dtype=jnp.float64),
            )
        )

    def objective_fixed(z):
        pos = 0
        mu0 = z[pos:pos + PARAMETER_COUNT]; pos += PARAMETER_COUNT
        cohort_mu = z[pos:pos + ng * PARAMETER_COUNT].reshape((ng, PARAMETER_COUNT)); pos += ng * PARAMETER_COUNT
        total = _log_gaussian(mu0, jnp.asarray(mu_prior), jnp.asarray(G_inv), logdet_G)
        for g in range(ng):
            total = total + _log_gaussian(cohort_mu[g], mu0, jnp.asarray(C_inv), logdet_C)
        for idx, s in enumerate(subjects):
            x0_s = z[pos:pos + STATE_DIM]; pos += STATE_DIM
            th_s = z[pos:pos + PARAMETER_COUNT]; pos += PARAMETER_COUNT
            t, y, specs, R, _x_filter, P_filter, Qc, Z, U0 = static[idx]
            ll = ekf_loglik_jax(
                t, y, specs, R, x0_s, P_filter, th_s, Qc=Qc, Z=Z,
                random_effect_covariance=U0,
            )
            total = total - ll
            c_idx = cohort_index[s.cohort]
            total = total + _log_gaussian(th_s, cohort_mu[c_idx], jnp.asarray(W_inv), logdet_W)
            total = total + _log_gaussian(x0_s, jnp.asarray(s.initial_state_prior), jnp.asarray(X_inv), logdet_X)
        return total

    vg = jax.jit(jax.value_and_grad(objective_fixed))

    def fg(z_np: np.ndarray):
        value, grad = vg(jnp.asarray(z_np, dtype=jnp.float64))
        return float(value), np.asarray(grad, dtype=float)

    result = minimize(
        lambda z: fg(z)[0],
        z0,
        jac=lambda z: fg(z)[1],
        method="L-BFGS-B",
        options={"maxiter": int(maxiter), "gtol": float(gtol), "maxls": 40},
    )

    zhat = np.asarray(result.x, dtype=float)
    pos = 0
    mu0_hat = zhat[pos:pos + PARAMETER_COUNT]; pos += PARAMETER_COUNT
    cohort_hat = {
        cohort_names[g]: zhat[pos + g * PARAMETER_COUNT: pos + (g + 1) * PARAMETER_COUNT].copy()
        for g in range(ng)
    }
    pos += ng * PARAMETER_COUNT
    subject_theta: dict[str, np.ndarray] = {}
    subject_x0: dict[str, np.ndarray] = {}
    for s in subjects:
        subject_x0[s.subject_id] = zhat[pos:pos + STATE_DIM].copy(); pos += STATE_DIM
        subject_theta[s.subject_id] = zhat[pos:pos + PARAMETER_COUNT].copy(); pos += PARAMETER_COUNT

    laplace = None
    try:
        H = np.asarray(jax.hessian(objective_fixed)(jnp.asarray(zhat, dtype=jnp.float64)), dtype=float)
        H = 0.5 * (H + H.T)
        vals, vecs = np.linalg.eigh(H)
        floor = max(float(np.max(np.abs(vals))) * 1e-10, 1e-8)
        vals = np.maximum(vals, floor)
        laplace = vecs @ np.diag(1.0 / vals) @ vecs.T
    except Exception:
        laplace = None

    value, grad = fg(zhat)
    return HierarchicalMAPResult(
        success=bool(result.success),
        message=str(result.message),
        objective=float(value),
        global_mean=mu0_hat,
        cohort_means=cohort_hat,
        subject_theta=subject_theta,
        subject_x0=subject_x0,
        map_vector=zhat,
        laplace_covariance=laplace,
        gradient_norm=float(np.linalg.norm(grad)),
        n_iter=int(getattr(result, "nit", 0)),
    )
