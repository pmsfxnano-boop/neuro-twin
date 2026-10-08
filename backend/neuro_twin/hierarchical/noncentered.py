"""Non-centered Bayesian parameterization for hierarchical P-I-N-Q inference.

The actual NumPyro graph is built only when NumPyro is installed.  The transform
itself is backend-agnostic and unit-testable: correlated Gaussian hierarchy is
represented by standard-normal latent variables and Cholesky factors.
"""
from __future__ import annotations

from typing import Any
import numpy as np


def cholesky_spd(cov: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    C = 0.5 * (np.asarray(cov, dtype=float) + np.asarray(cov, dtype=float).T)
    try:
        return np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(C)
        vals = np.maximum(vals, float(jitter))
        return np.linalg.cholesky((vecs * vals) @ vecs.T)


def noncentered_gaussian(z: Any, mean: Any, covariance: Any):
    """Return mean + L z while preserving full covariance structure."""
    import jax.numpy as jnp
    L = jnp.linalg.cholesky(0.5 * (jnp.asarray(covariance) + jnp.asarray(covariance).T))
    return jnp.asarray(mean) + L @ jnp.asarray(z)


def parameterization_contract() -> dict[str, Any]:
    return {
        "name": "noncentered",
        "global": "mu_global = mu_prior + L_global @ z_global",
        "cohort": "mu_cohort = mu_global + L_cohort @ z_cohort",
        "subject": "theta_subject = mu_cohort + L_within @ z_subject",
        "benefit": "reduces funnel pathologies when group variance is weakly identified",
        "claim_boundary": "parameterization choice; it does not guarantee better posterior geometry",
    }


def build_hierarchical_model_noncentered(
    subjects,
    *,
    theta_mean_prior,
    theta_global_covariance,
    theta_cohort_covariance,
    theta_within_covariance,
):
    """Build the full NumPyro hierarchical graph in non-centered form.

    This intentionally imports NumPyro lazily. The graph is mathematically
    equivalent to the centered Batch 10 hierarchy, but samples standard-normal
    latent variables and applies Cholesky transforms, which can improve HMC
    geometry when variance components are weakly identified.
    """
    try:
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from neuro_twin.hierarchical.bayesian_hierarchical import (
            _affine_transition,
            _process_covariance,
            _safe_name,
            _spd,
            observation_mean,
        )
        from neuro_twin.hierarchical.joint_map import PARAMETER_COUNT, STATE_DIM
    except Exception as exc:  # pragma: no cover
        raise ImportError("NumPyro/JAX are required for the non-centered model") from exc

    mu_prior = np.asarray(theta_mean_prior, dtype=float)
    G = _spd(theta_global_covariance)
    C = _spd(theta_cohort_covariance)
    W = _spd(theta_within_covariance)
    cohort_names = tuple(sorted({s.cohort for s in subjects}))
    cohort_index = {name: i for i, name in enumerate(cohort_names)}
    Lg, Lc, Lw = (jnp.asarray(cholesky_spd(M)) for M in (G, C, W))

    def model():
        zg = numpyro.sample("z_global", dist.Normal(0.0, 1.0).expand((PARAMETER_COUNT,)).to_event(1))
        global_theta = jnp.asarray(mu_prior) + Lg @ zg
        numpyro.deterministic("global_theta", global_theta)
        cohort_theta = {}
        for cohort in cohort_names:
            zc = numpyro.sample(f"z_cohort__{_safe_name(cohort)}", dist.Normal(0.0, 1.0).expand((PARAMETER_COUNT,)).to_event(1))
            cohort_theta[cohort] = global_theta + Lc @ zc
            numpyro.deterministic(f"cohort_theta__{_safe_name(cohort)}", cohort_theta[cohort])

        for subject in subjects:
            sid = _safe_name(subject.subject_id)
            zt = numpyro.sample(f"z_theta__{sid}", dist.Normal(0.0, 1.0).expand((PARAMETER_COUNT,)).to_event(1))
            theta = cohort_theta[subject.cohort] + Lw @ zt
            numpyro.deterministic(f"theta__{sid}", theta)
            x_cov = jnp.asarray(_spd(subject.initial_state_covariance))
            Lx = jnp.linalg.cholesky(x_cov)
            zx = numpyro.sample(f"z_x0__{sid}", dist.Normal(0.0, 1.0).expand((STATE_DIM,)).to_event(1))
            x = jnp.asarray(subject.initial_state_prior) + Lx @ zx
            numpyro.deterministic(f"x0__{sid}", x)
            if subject.random_effect_design is not None:
                Z = jnp.asarray(subject.random_effect_design)
                U = jnp.asarray(_spd(subject.random_effect_covariance))
                Lu = jnp.linalg.cholesky(U)
                zu = numpyro.sample(f"z_u__{sid}", dist.Normal(0.0, 1.0).expand((Z.shape[1],)).to_event(1))
                u = Lu @ zu
                numpyro.deterministic(f"u__{sid}", u)
            else:
                Z = u = None

            times = np.asarray(subject.time, dtype=float)
            Y = np.asarray(subject.observations, dtype=float)
            R = _spd(subject.R)
            Qc = None if subject.process_noise_spectral_density is None else _spd(subject.process_noise_spectral_density)
            for k in range(len(times)):
                if k > 0:
                    dt = float(times[k] - times[k - 1])
                    phi, offset = _affine_transition(theta, dt)
                    x = phi @ x + offset
                    if Qc is not None and np.max(np.abs(Qc)) > 0:
                        qd = _process_covariance(theta, jnp.asarray(Qc), dt)
                        qd = 0.5 * (qd + qd.T) + jnp.eye(STATE_DIM) * 1e-8
                        Lq = jnp.linalg.cholesky(qd)
                        eps = numpyro.sample(f"process_eps__{sid}__{k}", dist.Normal(0.0, 1.0).expand((STATE_DIM,)).to_event(1))
                        x = x + Lq @ eps
                    numpyro.deterministic(f"state__{sid}__{k}", x)
                mask = np.isfinite(Y[k])
                idx = np.flatnonzero(mask)
                if idx.size:
                    mean = observation_mean(subject.observation_specs, x, theta, Z, u)
                    idxj = jnp.asarray(idx, dtype=jnp.int32)
                    numpyro.sample(
                        f"y__{sid}__{k}",
                        dist.MultivariateNormal(mean[idxj], covariance_matrix=jnp.asarray(R[np.ix_(idx, idx)])),
                        obs=jnp.asarray(Y[k, idx], dtype=jnp.float64),
                    )
    return model
