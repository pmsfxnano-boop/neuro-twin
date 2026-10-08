"""Uncertainty propagation for PET arterial input functions (AIFs).

Batch 18 adds an explicit nuisance-distribution layer around the deterministic
Batch 17 frame-integrated kinetic model.  The purpose is to quantify how
uncertainty in measured plasma activity, parent-fraction measurements, and a
fixed-but-uncertain arterial delay propagates into kinetic parameters.

The implementation is deliberately Monte-Carlo based rather than pretending
that a first-order covariance approximation is globally reliable for bounded
parent fractions or nonlinear delay effects.  A local delta-method summary is
reported only as a diagnostic when enough successful samples are available.

Assumptions are explicit:
- total plasma activity uses a lognormal perturbation model on positive values;
- parent fraction uses a logistic-normal perturbation, preserving (0,1);
- temporal correlation follows exp(-|dt|/ell);
- delay uses a non-negative truncated normal;
- the tissue observation model remains the exact frame-integrated Batch 17 model.

No tracer-specific prior is embedded here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import math
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import truncnorm

from .frame_models import frame_integrated_1tcm, frame_integrated_2tcm, fit_frame_integrated_compartment
from .kinetics import ArterialInputFunction, KineticFitResult, PETKineticError

AIF_UNCERTAINTY_VERSION = "0.1.0-b18"


def _as_1d(values: Sequence[float], name: str) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise PETKineticError(f"{name} must be a non-empty finite 1D array")
    return x


def _correlation_matrix(time: np.ndarray, length_scale: float | None) -> np.ndarray:
    n = time.size
    if n == 1:
        return np.ones((1, 1), dtype=float)
    if length_scale is None or length_scale <= 0 or not math.isfinite(float(length_scale)):
        return np.eye(n, dtype=float)
    dt = np.abs(time[:, None] - time[None, :])
    return np.exp(-dt / float(length_scale))


def _stable_cholesky(cov: np.ndarray, jitter: float = 1e-10) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        scale = max(float(np.max(np.diag(cov))), 1.0)
        return np.linalg.cholesky(cov + (jitter * scale) * np.eye(cov.shape[0]))


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p / (1.0 - p))


def _expit(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


@dataclass(frozen=True)
class AIFUncertaintySpec:
    """Parametric nuisance distribution for an AIF.

    `total_plasma_rel_sd` is a multiplicative/log-scale SD.  `parent_fraction_sd`
    is an approximate SD on the logit scale.  Correlation length is expressed in
    the same time unit as the AIF.  `delay_sd` is in the same time unit as well.
    """

    total_plasma_rel_sd: float = 0.0
    parent_fraction_sd: float = 0.0
    total_plasma_corr_length: float | None = None
    parent_fraction_corr_length: float | None = None
    delay_sd: float = 0.0
    n_samples: int = 200
    seed: int = 1729
    min_parent_fraction: float = 1e-6
    min_total_plasma: float = 1e-12

    def __post_init__(self) -> None:
        vals = (self.total_plasma_rel_sd, self.parent_fraction_sd, self.delay_sd)
        if any((not math.isfinite(float(v)) or float(v) < 0) for v in vals):
            raise PETKineticError("AIF uncertainty SDs must be finite and >= 0")
        if self.total_plasma_corr_length is not None and (self.total_plasma_corr_length <= 0 or not math.isfinite(self.total_plasma_corr_length)):
            raise PETKineticError("total_plasma_corr_length must be positive when provided")
        if self.parent_fraction_corr_length is not None and (self.parent_fraction_corr_length <= 0 or not math.isfinite(self.parent_fraction_corr_length)):
            raise PETKineticError("parent_fraction_corr_length must be positive when provided")
        if int(self.n_samples) < 2:
            raise PETKineticError("n_samples must be >= 2")
        if not (0 < self.min_parent_fraction < 0.5):
            raise PETKineticError("min_parent_fraction must lie in (0, 0.5)")
        if self.min_total_plasma <= 0 or not math.isfinite(float(self.min_total_plasma)):
            raise PETKineticError("min_total_plasma must be > 0")


@dataclass(frozen=True)
class AIFSampleNuisance:
    total_plasma_multiplier: np.ndarray
    parent_fraction_logit_shift: np.ndarray
    delay: float


@dataclass(frozen=True)
class AIFPropagationResult:
    """Monte-Carlo propagation summary.

    `parameter_draws` contains one row per successful fit and follows
    `parameter_names`.  `parameter_mean/covariance` describe propagated
    uncertainty conditional on the tissue observations supplied to the fits.
    """

    model: str
    parameter_names: tuple[str, ...]
    parameter_draws: np.ndarray
    parameter_mean: np.ndarray
    parameter_sd: np.ndarray
    parameter_covariance: np.ndarray
    aif_success_rate: float
    fit_success_rate: float
    delay_mean: float
    delay_sd: float
    nuisance_diagnostics: dict[str, float]
    assumptions: tuple[str, ...]

    @property
    def n_success(self) -> int:
        return int(self.parameter_draws.shape[0])


def sample_aif_ensemble(
    aif: ArterialInputFunction,
    spec: AIFUncertaintySpec,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[tuple[ArterialInputFunction, ...], tuple[AIFSampleNuisance, ...]]:
    """Draw AIF nuisance realizations while preserving physical bounds."""
    rng = np.random.default_rng(spec.seed) if rng is None else rng
    t = aif.time_array
    n = t.size

    rel = float(spec.total_plasma_rel_sd)
    pf_sd = float(spec.parent_fraction_sd)
    total_cov = (rel**2) * _correlation_matrix(t, spec.total_plasma_corr_length)
    pf_cov = (pf_sd**2) * _correlation_matrix(t, spec.parent_fraction_corr_length)
    Lt = _stable_cholesky(total_cov) if rel > 0 else np.zeros_like(total_cov)
    Lp = _stable_cholesky(pf_cov) if pf_sd > 0 else np.zeros_like(pf_cov)

    # Log-scale multiplicative error keeps total plasma non-negative.
    z_total = rng.normal(size=(spec.n_samples, n)) @ Lt.T if rel > 0 else np.zeros((spec.n_samples, n))
    total_multiplier = np.exp(z_total - 0.5 * (rel**2))

    # Logistic-normal perturbation preserves the (0,1) support of parent fraction.
    eps = float(spec.min_parent_fraction)
    fp = np.clip(aif.parent_fraction if hasattr(aif, "parent_fraction") else np.asarray(aif.parent_fraction), eps, 1.0 - eps)
    base_logit = _logit(fp)
    z_parent = rng.normal(size=(spec.n_samples, n)) @ Lp.T if pf_sd > 0 else np.zeros((spec.n_samples, n))
    sampled_fp = _expit(base_logit[None, :] + z_parent)

    if spec.delay_sd > 0:
        a = (0.0 - float(aif.delay)) / spec.delay_sd
        delay = truncnorm.rvs(a, np.inf, loc=float(aif.delay), scale=float(spec.delay_sd), size=spec.n_samples, random_state=rng)
    else:
        delay = np.full(spec.n_samples, float(aif.delay), dtype=float)

    samples: list[ArterialInputFunction] = []
    nuisances: list[AIFSampleNuisance] = []
    cp0 = np.asarray(aif.total_plasma, dtype=float)
    for i in range(spec.n_samples):
        cp = np.maximum(cp0 * total_multiplier[i], spec.min_total_plasma)
        fp_i = np.clip(sampled_fp[i], eps, 1.0 - eps)
        sample = ArterialInputFunction(
            tuple(t),
            tuple(cp),
            tuple(fp_i),
            float(delay[i]),
        )
        samples.append(sample)
        nuisances.append(AIFSampleNuisance(total_multiplier[i], z_parent[i], float(delay[i])))
    return tuple(samples), tuple(nuisances)


def _fit_one(
    frame_starts: Sequence[float],
    frame_durations: Sequence[float],
    tac: Sequence[float],
    aif: ArterialInputFunction,
    model: Literal["1TCM", "2TCM"],
    initial: Sequence[float] | None,
    observation_sd: Sequence[float] | None,
    vascular_fraction: float,
) -> KineticFitResult:
    return fit_frame_integrated_compartment(
        frame_starts,
        frame_durations,
        tac,
        aif,
        model=model,
        initial=initial,
        observation_sd=observation_sd,
        vascular_fraction=vascular_fraction,
    )


def propagate_aif_uncertainty(
    frame_starts: Sequence[float],
    frame_durations: Sequence[float],
    tac: Sequence[float],
    aif: ArterialInputFunction,
    *,
    model: Literal["1TCM", "2TCM"] = "1TCM",
    initial: Sequence[float] | None = None,
    observation_sd: Sequence[float] | None = None,
    vascular_fraction: float = 0.0,
    spec: AIFUncertaintySpec | None = None,
    resample_observation_noise: bool = False,
) -> AIFPropagationResult:
    """Propagate AIF and optional tissue-observation uncertainty to parameters.

    Holding the observed TAC fixed isolates AIF-induced uncertainty.  When
    `resample_observation_noise=True` and `observation_sd` is supplied, the same
    Monte-Carlo draw also perturbs the tissue frame observations, producing a
    total (AIF + tissue-noise) parametric uncertainty under the supplied model.
    """
    spec = spec or AIFUncertaintySpec()
    observed = _as_1d(tac, "tac")
    if observation_sd is not None:
        sd = _as_1d(observation_sd, "observation_sd")
        if sd.size != observed.size or np.any(sd <= 0):
            raise PETKineticError("observation_sd must be positive and match TAC")
    else:
        sd = None

    rng = np.random.default_rng(spec.seed)
    samples, nuisances = sample_aif_ensemble(aif, spec, rng=rng)
    draws: list[np.ndarray] = []
    failures = 0
    fit_failures = 0
    tac_failures = 0
    names: tuple[str, ...] | None = None
    for i, (aif_i, nuisance) in enumerate(zip(samples, nuisances)):
        y_i = observed.copy()
        if resample_observation_noise:
            if sd is None:
                raise PETKineticError("observation_sd is required when resample_observation_noise=True")
            y_i = y_i + rng.normal(0.0, sd)
            y_i = np.maximum(y_i, 0.0)
        try:
            fit = _fit_one(
                frame_starts,
                frame_durations,
                y_i,
                aif_i,
                model,
                initial,
                observation_sd,
                vascular_fraction,
            )
            if names is None:
                names = fit.parameter_names
            if fit.success and np.all(np.isfinite([fit.parameters[n] for n in fit.parameter_names])):
                draws.append(np.asarray([fit.parameters[n] for n in fit.parameter_names], dtype=float))
            else:
                fit_failures += 1
        except (PETKineticError, ValueError, np.linalg.LinAlgError, FloatingPointError):
            failures += 1

    if names is None or not draws:
        raise PETKineticError("no successful kinetic fits in AIF uncertainty propagation")
    arr = np.vstack(draws)
    mean = arr.mean(axis=0)
    cov = np.cov(arr, rowvar=False, ddof=1) if arr.shape[0] > 1 else np.zeros((arr.shape[1], arr.shape[1]))
    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    sd_out = np.sqrt(np.maximum(np.diag(cov), 0.0))
    diagnostics = {
        "n_samples_requested": float(spec.n_samples),
        "n_samples_success": float(arr.shape[0]),
        "n_aif_draw_failures": float(failures),
        "n_fit_failures": float(fit_failures),
        "delay_sample_mean": float(np.mean([n.delay for n in nuisances])),
        "delay_sample_sd": float(np.std([n.delay for n in nuisances], ddof=1)) if len(nuisances) > 1 else 0.0,
    }
    if resample_observation_noise and sd is not None:
        diagnostics["observation_noise_resampled"] = 1.0
    else:
        diagnostics["observation_noise_resampled"] = 0.0
    return AIFPropagationResult(
        model=model,
        parameter_names=names,
        parameter_draws=arr,
        parameter_mean=mean,
        parameter_sd=sd_out,
        parameter_covariance=cov,
        aif_success_rate=float((spec.n_samples - failures) / spec.n_samples),
        fit_success_rate=float(arr.shape[0] / spec.n_samples),
        delay_mean=float(np.mean([n.delay for n in nuisances])),
        delay_sd=float(np.std([n.delay for n in nuisances], ddof=1)) if len(nuisances) > 1 else 0.0,
        nuisance_diagnostics=diagnostics,
        assumptions=(
            "total plasma uncertainty modeled on log scale",
            "parent fraction uncertainty modeled on logit scale",
            "temporal correlation follows exponential kernel",
            "delay uncertainty is truncated normal on non-negative support",
            "kinetic forward model is the deterministic Batch 17 frame-integrated model",
        ),
    )


def local_aif_sensitivity_from_ensemble(result: AIFPropagationResult) -> dict[str, float]:
    """Return coefficient of variation for propagated parameter uncertainty."""
    out: dict[str, float] = {}
    for j, name in enumerate(result.parameter_names):
        mu = float(result.parameter_mean[j])
        sd = float(result.parameter_sd[j])
        out[f"{name}_cv"] = float(sd / abs(mu)) if mu != 0 else math.inf
    return out
