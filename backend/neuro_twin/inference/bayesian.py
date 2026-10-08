"""Optional NumPyro Bayesian posterior for the exact state-space likelihood."""
from __future__ import annotations

from typing import Any

import numpy as np

from neuro_twin.inference.jax_likelihood import JAX_AVAILABLE, kalman_loglik_jax


def sample_posterior_numpyro(
    time: np.ndarray,
    observations: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    initial_mean: np.ndarray,
    initial_covariance: np.ndarray,
    *,
    prior_mean: np.ndarray,
    prior_scale: np.ndarray,
    process_noise_spectral_density: np.ndarray | None = None,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 2,
    rng_seed: int = 0,
    **mcmc_kwargs: Any,
):
    """Run NUTS over theta using the exact differentiable Kalman likelihood.

    NumPyro is intentionally optional; importing this module does not require
    it. The initial state is conditioned on ``initial_mean`` rather than
    sampled, keeping this backend directly comparable with the marginal MLE.
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX is required for the NumPyro backend")
    try:
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS
    except Exception as exc:  # pragma: no cover
        raise ImportError("NumPyro is required for Bayesian inference") from exc

    import jax.numpy as jnp

    prior_mean = np.asarray(prior_mean, dtype=float)
    prior_scale = np.asarray(prior_scale, dtype=float)
    if prior_mean.shape != (11,) or prior_scale.shape != (11,):
        raise ValueError("prior_mean and prior_scale must have shape (11,)")
    if np.any(prior_scale <= 0):
        raise ValueError("prior_scale must be positive")

    def model():
        theta = numpyro.sample(
            "theta",
            dist.Normal(jnp.asarray(prior_mean), jnp.asarray(prior_scale)).to_event(1),
        )
        ll = kalman_loglik_jax(
            time,
            observations,
            H,
            R,
            initial_mean,
            initial_covariance,
            theta,
            process_noise_spectral_density=process_noise_spectral_density,
        )
        numpyro.factor("kalman_log_likelihood", ll)

    kernel = NUTS(model, **mcmc_kwargs)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    import jax

    mcmc.run(jax.random.key(rng_seed))
    return mcmc
