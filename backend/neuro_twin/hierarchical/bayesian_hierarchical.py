"""Full Bayesian hierarchical longitudinal model with an exact transition.

This module upgrades Batch 09 from joint MAP/Laplace to an optional Bayesian
backend.  The model samples the latent trajectories explicitly, so the
nonlinear observation model is not linearized by an EKF inside the posterior.
The continuous-time P-I-N-Q dynamics remain unchanged; the affine ODE is
converted exactly to a continuous-discrete Gaussian state transition using a
matrix exponential and Van Loan discretisation for process noise.

NumPyro is optional.  The module can always validate the generative contract,
perform posterior-predictive simulations from externally supplied posterior
samples, and compute calibration/identifiability diagnostics.  NUTS and SVI
require the ``science`` extra with NumPyro installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from neuro_twin.hierarchical.joint_map import SubjectSeries, PARAMETER_COUNT, STATE_DIM
from neuro_twin.inference.jax_ekf import JAX_AVAILABLE

try:  # Optional dependency path.
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import expm
except Exception:  # pragma: no cover
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    expm = None  # type: ignore[assignment]

NUMPYRO_AVAILABLE = False
try:  # pragma: no cover - depends on environment.
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, Predictive
    from numpyro.infer.autoguide import AutoMultivariateNormal
    from numpyro.optim import Adam
    NUMPYRO_AVAILABLE = True
except Exception:  # pragma: no cover
    numpyro = None  # type: ignore[assignment]
    dist = None  # type: ignore[assignment]
    MCMC = NUTS = SVI = Trace_ELBO = Predictive = AutoMultivariateNormal = Adam = None  # type: ignore[assignment]


@dataclass(frozen=True)
class BayesianRunResult:
    """Container for MCMC or SVI output without coupling callers to NumPyro."""

    backend: str
    object: Any
    posterior_samples: dict[str, np.ndarray]
    diagnostics: dict[str, Any]


def require_numpyro() -> None:
    if not JAX_AVAILABLE:
        raise ImportError("JAX is required for the Bayesian backend")
    if not NUMPYRO_AVAILABLE:
        raise ImportError("NumPyro is required for Bayesian inference; install neuro-twin[science]")


def _theta_A_c(theta: Any):
    t = jnp.asarray(theta)
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


def _affine_transition(theta: Any, dt: Any):
    A, c = _theta_A_c(theta)
    M = jnp.zeros((5, 5), dtype=jnp.float64)
    M = M.at[:4, :4].set(A)
    M = M.at[:4, 4].set(c)
    E = expm(M * dt)
    return E[:4, :4], E[:4, 4]


def _process_covariance(theta: Any, Qc: Any, dt: Any):
    A, _ = _theta_A_c(theta)
    M = jnp.zeros((8, 8), dtype=jnp.float64)
    M = M.at[:4, :4].set(A)
    M = M.at[:4, 4:].set(Qc)
    M = M.at[4:, 4:].set(-A.T)
    E = expm(M * dt)
    phi = E[:4, :4]
    qd = E[:4, 4:] @ phi.T
    return 0.5 * (qd + qd.T)


def _link(name: str, eta: Any):
    if name == "identity":
        return eta
    if name == "exp":
        return jnp.exp(eta)
    if name == "log":
        return jnp.log(jnp.maximum(eta, 1e-12))
    if name == "softplus":
        return jax.nn.softplus(eta)
    if name == "sigmoid":
        return jax.nn.sigmoid(eta)
    raise ValueError(f"unsupported link: {name}")


def observation_mean(specs: tuple[dict[str, Any], ...], state: Any, theta: Any, random_effect_design=None, random_effect=None):
    vals = []
    for spec in specs:
        wx = jnp.asarray(spec["state_weights"], dtype=jnp.float64)
        wt = jnp.asarray(spec["parameter_weights"], dtype=jnp.float64)
        eta = wx @ state + wt @ theta + float(spec["bias"])
        vals.append(_link(spec["link"], eta))
    out = jnp.stack(vals)
    if random_effect_design is not None:
        out = out + jnp.asarray(random_effect_design, dtype=jnp.float64) @ jnp.asarray(random_effect, dtype=jnp.float64)
    return out


def _spd(cov: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    C = 0.5 * (np.asarray(cov, dtype=float) + np.asarray(cov, dtype=float).T)
    vals = np.linalg.eigvalsh(C)
    if np.min(vals) <= 0:
        C = C + np.eye(C.shape[0]) * max(jitter, -float(np.min(vals)) + jitter)
    return C


def _safe_name(value: str) -> str:
    return "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in value)


def build_hierarchical_model(
    subjects: list[SubjectSeries],
    *,
    theta_mean_prior: np.ndarray,
    theta_global_covariance: np.ndarray,
    theta_cohort_covariance: np.ndarray,
    theta_within_covariance: np.ndarray,
    process_noise_jitter: float = 1e-8,
) -> Callable[[], None]:
    """Construct a NumPyro model with explicit latent trajectories.

    The returned model is a genuine generative state-space model.  For each
    subject, x0, theta, persistent random effects (when configured), process
    innovations, and observed channels are represented explicitly.
    """
    require_numpyro()
    if not subjects:
        raise ValueError("at least one subject is required")

    mu_prior = np.asarray(theta_mean_prior, dtype=float)
    G = _spd(theta_global_covariance)
    C = _spd(theta_cohort_covariance)
    W = _spd(theta_within_covariance)
    if mu_prior.shape != (PARAMETER_COUNT,):
        raise ValueError("theta_mean_prior must have shape (11,)")
    for name, matrix in (("global", G), ("cohort", C), ("within", W)):
        if matrix.shape != (PARAMETER_COUNT, PARAMETER_COUNT):
            raise ValueError(f"{name} covariance must be 11x11")

    cohort_names = tuple(sorted({s.cohort for s in subjects}))
    cohort_index = {name: i for i, name in enumerate(cohort_names)}

    def model():
        global_theta = numpyro.sample(
            "global_theta",
            dist.MultivariateNormal(
                loc=jnp.asarray(mu_prior), covariance_matrix=jnp.asarray(G)
            ),
        )
        cohort_theta = []
        for cohort in cohort_names:
            cohort_theta.append(
                numpyro.sample(
                    f"cohort_theta__{_safe_name(cohort)}",
                    dist.MultivariateNormal(
                        loc=global_theta,
                        covariance_matrix=jnp.asarray(C),
                    ),
                )
            )

        for subject in subjects:
            sid = _safe_name(subject.subject_id)
            ctheta = cohort_theta[cohort_index[subject.cohort]]
            theta = numpyro.sample(
                f"theta__{sid}",
                dist.MultivariateNormal(
                    loc=ctheta,
                    covariance_matrix=jnp.asarray(W),
                ),
            )
            x = numpyro.sample(
                f"x0__{sid}",
                dist.MultivariateNormal(
                    loc=jnp.asarray(subject.initial_state_prior),
                    covariance_matrix=jnp.asarray(_spd(subject.initial_state_covariance)),
                ),
            )

            Z = None if subject.random_effect_design is None else jnp.asarray(subject.random_effect_design)
            if Z is not None:
                U = jnp.asarray(_spd(subject.random_effect_covariance))
                u = numpyro.sample(f"u__{sid}", dist.MultivariateNormal(loc=jnp.zeros(Z.shape[1]), covariance_matrix=U))
            else:
                u = None

            times = np.asarray(subject.time, dtype=float)
            Y = np.asarray(subject.observations, dtype=float)
            R = _spd(subject.R)
            Qc = np.zeros((4, 4), dtype=float) if subject.process_noise_spectral_density is None else _spd(subject.process_noise_spectral_density, jitter=process_noise_jitter)

            numpyro.deterministic(f"state__{sid}__0", x)
            for k in range(len(times)):
                if k > 0:
                    phi, offset = _affine_transition(theta, float(times[k] - times[k - 1]))
                    if subject.process_noise_spectral_density is None or np.max(np.abs(Qc)) == 0.0:
                        x = phi @ x + offset
                    else:
                        qd = _process_covariance(theta, jnp.asarray(Qc), float(times[k] - times[k - 1]))
                        qd = qd + jnp.eye(4) * process_noise_jitter
                        L = jnp.linalg.cholesky(0.5 * (qd + qd.T))
                        eps = numpyro.sample(
                            f"process_eps__{sid}__{k}",
                            dist.Normal(0.0, 1.0).expand((STATE_DIM,)).to_event(1),
                        )
                        x = phi @ x + offset + L @ eps
                    numpyro.deterministic(f"state__{sid}__{k}", x)

                mask = np.isfinite(Y[k])
                idx = np.flatnonzero(mask)
                if idx.size == 0:
                    continue
                mean_full = observation_mean(subject.observation_specs, x, theta, Z, u)
                y_obs = jnp.asarray(Y[k, idx], dtype=jnp.float64)
                mean_obs = mean_full[jnp.asarray(idx, dtype=jnp.int32)]
                R_obs = jnp.asarray(R[np.ix_(idx, idx)], dtype=jnp.float64)
                numpyro.sample(
                    f"y__{sid}__{k}",
                    dist.MultivariateNormal(loc=mean_obs, covariance_matrix=R_obs),
                    obs=y_obs,
                )

    return model


def sample_hierarchical_nuts(
    subjects: list[SubjectSeries],
    *,
    theta_mean_prior: np.ndarray,
    theta_global_covariance: np.ndarray,
    theta_cohort_covariance: np.ndarray,
    theta_within_covariance: np.ndarray,
    num_warmup: int = 500,
    num_samples: int = 500,
    num_chains: int = 1,
    rng_seed: int = 0,
    **nuts_kwargs: Any,
) -> BayesianRunResult:
    """Run NUTS for the full hierarchical nonlinear state-space posterior."""
    require_numpyro()
    model = build_hierarchical_model(
        subjects,
        theta_mean_prior=theta_mean_prior,
        theta_global_covariance=theta_global_covariance,
        theta_cohort_covariance=theta_cohort_covariance,
        theta_within_covariance=theta_within_covariance,
    )
    kernel = NUTS(model, **nuts_kwargs)
    mcmc = MCMC(kernel, num_warmup=int(num_warmup), num_samples=int(num_samples), num_chains=int(num_chains), progress_bar=False)
    mcmc.run(jax.random.key(int(rng_seed)))
    samples = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=False).items()}
    diagnostics = {"backend": "nuts", "num_warmup": num_warmup, "num_samples": num_samples, "num_chains": num_chains}
    try:
        from numpyro.diagnostics import summary
        diagnostics["summary"] = summary(mcmc.get_samples(group_by_chain=True), group_by_chain=True)
    except Exception:
        pass
    return BayesianRunResult("nuts", mcmc, samples, diagnostics)


def fit_hierarchical_svi(
    subjects: list[SubjectSeries],
    *,
    theta_mean_prior: np.ndarray,
    theta_global_covariance: np.ndarray,
    theta_cohort_covariance: np.ndarray,
    theta_within_covariance: np.ndarray,
    steps: int = 2000,
    learning_rate: float = 1e-3,
    posterior_draws: int = 500,
    rng_seed: int = 0,
) -> BayesianRunResult:
    """Fit an AutoMultivariateNormal SVI approximation to the same posterior."""
    require_numpyro()
    model = build_hierarchical_model(
        subjects,
        theta_mean_prior=theta_mean_prior,
        theta_global_covariance=theta_global_covariance,
        theta_cohort_covariance=theta_cohort_covariance,
        theta_within_covariance=theta_within_covariance,
    )
    guide = AutoMultivariateNormal(model)
    svi = SVI(model, guide, Adam(float(learning_rate)), Trace_ELBO())
    state = svi.init(jax.random.key(int(rng_seed)))
    state, losses = svi.run(state, int(steps), progress_bar=False)
    params = svi.get_params(state)
    samples = {k: np.asarray(v) for k, v in guide.sample_posterior(jax.random.key(int(rng_seed) + 1), params, sample_shape=(int(posterior_draws),)).items()}
    diagnostics = {
        "backend": "svi",
        "steps": int(steps),
        "learning_rate": float(learning_rate),
        "final_elbo_loss": float(np.asarray(losses)[-1]),
        "loss_tail_mean": float(np.mean(np.asarray(losses)[-min(100, len(losses)):])),
        "num_draws": int(posterior_draws),
    }
    return BayesianRunResult("svi", {"svi": svi, "guide": guide, "state": state, "params": params}, samples, diagnostics)


def _subject_theta_sample(samples: dict[str, np.ndarray], subject_id: str) -> np.ndarray:
    key = f"theta__{_safe_name(subject_id)}"
    if key not in samples:
        raise KeyError(f"posterior samples missing {key}")
    return np.asarray(samples[key], dtype=float)


def posterior_predictive_simulation(
    subjects: list[SubjectSeries],
    posterior_samples: dict[str, np.ndarray],
    *,
    n_draws: int | None = None,
    seed: int = 0,
) -> dict[str, dict[str, np.ndarray]]:
    """Generate posterior-predictive observation draws at each observed visit.

    New process and measurement noise are drawn for each posterior draw.  The
    latent initial state, subject parameters, and persistent random effects are
    conditioned on their posterior samples.
    """
    rng = np.random.default_rng(seed)
    # Find draw count from theta samples; all subject theta arrays must agree.
    first = _subject_theta_sample(posterior_samples, subjects[0].subject_id)
    total = first.shape[0]
    nd = total if n_draws is None else min(int(n_draws), total)
    if nd <= 0:
        raise ValueError("n_draws must be positive")
    draw_idx = rng.choice(total, size=nd, replace=False) if nd < total else np.arange(total)

    out: dict[str, dict[str, np.ndarray]] = {}
    for subject in subjects:
        sid = _safe_name(subject.subject_id)
        theta_draws = _subject_theta_sample(posterior_samples, subject.subject_id)[draw_idx]
        xkey = f"x0__{sid}"
        if xkey not in posterior_samples:
            raise KeyError(f"posterior samples missing {xkey}")
        x0_draws = np.asarray(posterior_samples[xkey], dtype=float)[draw_idx]
        u_draws = None
        if subject.random_effect_design is not None:
            ukey = f"u__{sid}"
            if ukey not in posterior_samples:
                raise KeyError(f"posterior samples missing {ukey}")
            u_draws = np.asarray(posterior_samples[ukey], dtype=float)[draw_idx]

        T, M = subject.observations.shape
        sims = np.full((nd, T, M), np.nan, dtype=float)
        R = _spd(subject.R)
        Qc = np.zeros((4, 4), dtype=float) if subject.process_noise_spectral_density is None else _spd(subject.process_noise_spectral_density)
        for d in range(nd):
            theta = theta_draws[d]
            x = x0_draws[d].copy()
            u = None if u_draws is None else u_draws[d]
            for k in range(T):
                if k > 0:
                    # Use the same exact affine transition as the model.
                    A = np.asarray(_theta_A_c(theta)[0])
                    c = np.asarray(_theta_A_c(theta)[1])
                    from scipy.linalg import expm as scipy_expm
                    Maug = np.zeros((5, 5), dtype=float)
                    Maug[:4, :4] = A
                    Maug[:4, 4] = c
                    E = scipy_expm(Maug * float(subject.time[k] - subject.time[k - 1]))
                    phi, offset = E[:4, :4], E[:4, 4]
                    x = phi @ x + offset
                    if subject.process_noise_spectral_density is not None and np.max(np.abs(Qc)) > 0.0:
                        # Reuse Van Loan in NumPy/Scipy to draw exact Qd noise.
                        V = np.zeros((8, 8), dtype=float)
                        V[:4, :4] = A
                        V[:4, 4:] = Qc
                        V[4:, 4:] = -A.T
                        EV = scipy_expm(V * float(subject.time[k] - subject.time[k - 1]))
                        Qd = EV[:4, 4:] @ EV[:4, :4].T
                        Qd = 0.5 * (Qd + Qd.T)
                        vals, vecs = np.linalg.eigh(Qd)
                        vals = np.maximum(vals, 0.0)
                        x = x + vecs @ (np.sqrt(vals) * rng.normal(size=4))
                mean = np.asarray(observation_mean(subject.observation_specs, jnp.asarray(x), jnp.asarray(theta), None if subject.random_effect_design is None else jnp.asarray(subject.random_effect_design), None if u is None else jnp.asarray(u)), dtype=float)
                sims[d, k] = rng.multivariate_normal(mean, R)
        out[subject.subject_id] = {
            "draws": sims,
            "observed_mask": np.isfinite(subject.observations),
            "draw_indices": draw_idx.copy(),
        }
    return out


def posterior_predictive_checks(
    subjects: list[SubjectSeries],
    posterior_samples: dict[str, np.ndarray],
    *,
    seed: int = 0,
    n_draws: int | None = None,
    interval: float = 0.95,
) -> dict[str, Any]:
    """Compute calibrated posterior-predictive summaries and coverage."""
    if not (0.5 < interval < 1.0):
        raise ValueError("interval must lie in (0.5, 1)")
    sims = posterior_predictive_simulation(subjects, posterior_samples, n_draws=n_draws, seed=seed)
    alpha = 1.0 - interval
    per_subject = {}
    total_values = 0
    total_inside = 0
    rmses = []
    for subject in subjects:
        S = sims[subject.subject_id]["draws"]
        Y = np.asarray(subject.observations, dtype=float)
        mask = np.isfinite(Y)
        mean = np.nanmean(S, axis=0)
        lo = np.nanquantile(S, alpha / 2.0, axis=0)
        hi = np.nanquantile(S, 1.0 - alpha / 2.0, axis=0)
        observed = Y[mask]
        predicted = mean[mask]
        inside = (observed >= lo[mask]) & (observed <= hi[mask])
        coverage = float(np.mean(inside)) if inside.size else float("nan")
        rmse = float(np.sqrt(np.mean((observed - predicted) ** 2))) if observed.size else float("nan")
        total_values += int(inside.size)
        total_inside += int(np.sum(inside))
        if np.isfinite(rmse):
            rmses.append(rmse)
        per_subject[subject.subject_id] = {
            "n_observed": int(inside.size),
            "coverage": coverage,
            "rmse": rmse,
            "predictive_mean": mean,
            "lower": lo,
            "upper": hi,
        }
    aggregate = float(total_inside / total_values) if total_values else float("nan")
    return {
        "interval": float(interval),
        "aggregate_coverage": aggregate,
        "total_observed": int(total_values),
        "mean_subject_rmse": float(np.mean(rmses)) if rmses else float("nan"),
        "per_subject": per_subject,
    }


def calibration_gate(
    ppc: dict[str, Any],
    *,
    target_coverage: float = 0.95,
    tolerance: float = 0.05,
    min_observed: int = 100,
) -> dict[str, Any]:
    """Return a transparent engineering calibration gate.

    This is not a clinical validation criterion.  It is a reproducibility gate
    that flags severe under/over-coverage while declaring small samples
    ``INCONCLUSIVE`` rather than fabricating confidence.
    """
    n = int(ppc.get("total_observed", 0))
    coverage = float(ppc.get("aggregate_coverage", np.nan))
    if n < min_observed or not np.isfinite(coverage):
        status = "INCONCLUSIVE"
    else:
        status = "PASS" if abs(coverage - target_coverage) <= tolerance else "FAIL"
    return {
        "status": status,
        "aggregate_coverage": coverage,
        "target_coverage": float(target_coverage),
        "tolerance": float(tolerance),
        "minimum_observed": int(min_observed),
        "observed": n,
        "scientific_scope": "engineering calibration gate; not clinical validation",
    }


def posterior_parameter_identifiability(
    posterior_samples: dict[str, np.ndarray],
    *,
    prefix: str = "theta__",
) -> dict[str, Any]:
    """Estimate posterior information geometry from subject theta draws."""
    keys = sorted(k for k in posterior_samples if k.startswith(prefix))
    if not keys:
        raise ValueError("no theta posterior samples found")
    blocks = [np.asarray(posterior_samples[k], dtype=float) for k in keys]
    pooled = np.concatenate(blocks, axis=0)
    if pooled.ndim != 2 or pooled.shape[1] != PARAMETER_COUNT:
        raise ValueError("theta posterior samples must have shape (draws, 11)")
    cov = np.cov(pooled, rowvar=False)
    cov = 0.5 * (cov + cov.T)
    vals = np.linalg.eigvalsh(cov)
    vals = np.maximum(vals, 0.0)
    positive = vals[vals > max(vals.max() * 1e-10 if vals.size else 0.0, 1e-14)]
    cond = float(vals.max() / vals.min()) if vals.size and vals.min() > 0 else float("inf")
    probs = vals / vals.sum() if vals.sum() > 0 else np.ones_like(vals) / len(vals)
    entropy = -float(np.sum(probs[probs > 0] * np.log(probs[probs > 0])))
    effective_rank = float(np.exp(entropy))
    return {
        "subjects": keys,
        "n_draws_pooled": int(pooled.shape[0]),
        "posterior_covariance": cov,
        "eigenvalues": vals,
        "positive_eigenvalues": positive,
        "condition_number": cond,
        "effective_rank_entropy": effective_rank,
    }


def numpyro_model_contract_check(
    subjects: list[SubjectSeries],
    *,
    theta_mean_prior: np.ndarray,
    theta_global_covariance: np.ndarray,
    theta_cohort_covariance: np.ndarray,
    theta_within_covariance: np.ndarray,
) -> dict[str, Any]:
    """Validate the NumPyro generative graph structurally without sampling."""
    if not NUMPYRO_AVAILABLE:
        return {"available": False, "reason": "NumPyro not installed"}
    model = build_hierarchical_model(
        subjects,
        theta_mean_prior=theta_mean_prior,
        theta_global_covariance=theta_global_covariance,
        theta_cohort_covariance=theta_cohort_covariance,
        theta_within_covariance=theta_within_covariance,
    )
    from numpyro.handlers import seed, trace
    tr = trace(seed(model, jax.random.key(0))).get_trace()
    sample_sites = sorted(k for k, v in tr.items() if v.get("type") == "sample")
    return {"available": True, "sample_site_count": len(sample_sites), "sample_sites": sample_sites}
