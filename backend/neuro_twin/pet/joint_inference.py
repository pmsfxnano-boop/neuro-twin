"""Joint PET AIF/kinetic inference with latent delay and dispersion.

Batch 19 adds a low-dimensional Bayesian nuisance layer around the exact
frame-integrated PET model. The tissue model and the AIF uncertainty are fit
jointly rather than by a two-stage plug-in procedure.

Model assumptions are explicit:
- Gaussian frame measurement likelihood with supplied observation SD;
- lognormal prior on a common plasma scale;
- logistic-normal prior on a common parent-fraction logit shift;
- truncated-normal prior on arterial delay (support >= 0);
- lognormal prior on the positive dispersion time constant tau;
- causal first-order (exponential) dispersion filter:
      dD/dt = (Cp_parent(t-delay) - D)/tau.
- frame observations are exact time averages under piecewise-linear AIF input.

This is a local parametric Bayesian model. MAP + Laplace is an approximation,
not a substitute for full MCMC/NUTS. A rejection sampler from the local
Laplace posterior is provided only as a diagnostic ensemble.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence
import math

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares
from scipy.stats import norm

from .kinetics import ArterialInputFunction, PETKineticError
from .frame_models import PETFrameSchedule

JOINT_PET_VERSION = "0.1.0-b19"


def _as_1d(values: Sequence[float], name: str) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or a.size == 0 or not np.all(np.isfinite(a)):
        raise PETKineticError(f"{name} must be a non-empty finite 1D array")
    return a


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0 - 1e-12)
    return np.log(p / (1.0 - p))


def _expit(x: np.ndarray | float) -> np.ndarray | float:
    x_arr = np.asarray(x, dtype=float)
    pos = x_arr >= 0
    out = np.empty_like(x_arr)
    out[pos] = 1.0 / (1.0 + np.exp(-x_arr[pos]))
    ex = np.exp(x_arr[~pos])
    out[~pos] = ex / (1.0 + ex)
    return float(out) if np.ndim(x) == 0 else out


def _positive_log_prior(value: float, mean: float, sd: float) -> float:
    if value <= 0 or mean <= 0 or sd <= 0:
        raise PETKineticError("positive lognormal prior requires positive value/mean/sd")
    return (math.log(value) - math.log(mean)) / float(sd)


def _truncated_normal_residual(value: float, mean: float, sd: float) -> float:
    if sd <= 0:
        if abs(value - mean) > 1e-12:
            return math.inf
        return 0.0
    return (value - mean) / float(sd)


@dataclass(frozen=True)
class JointPETSpec:
    """Prior and fitting contract for joint PET nuisance/kinetic inference.

    `plasma_log_scale_sd` and `parent_fraction_logit_sd` are prior SDs on
    nuisance corrections, not empirical measurement errors. Setting either to
    zero fixes that nuisance correction at its nominal value.
    """

    model: Literal["1TCM", "2TCM"] = "1TCM"
    vascular_fraction: float = 0.0
    plasma_log_scale_sd: float = 0.0
    parent_fraction_logit_sd: float = 0.0
    delay_mean: float = 0.0
    delay_sd: float = 0.5
    dispersion_tau_mean: float = 1.0
    dispersion_tau_log_sd: float = 0.5
    delay_upper: float = 10.0
    dispersion_tau_lower: float = 1e-3
    dispersion_tau_upper: float = 30.0
    include_plasma_scale: bool = False
    include_parent_fraction_shift: bool = False
    include_dispersion: bool = True

    def __post_init__(self) -> None:
        if self.model not in {"1TCM", "2TCM"}:
            raise PETKineticError("model must be 1TCM or 2TCM")
        if not (0 <= self.vascular_fraction < 1):
            raise PETKineticError("vascular_fraction must be in [0,1)")
        if self.plasma_log_scale_sd < 0 or self.parent_fraction_logit_sd < 0:
            raise PETKineticError("nuisance prior SDs must be >= 0")
        if self.include_plasma_scale and self.plasma_log_scale_sd <= 0:
            raise PETKineticError("plasma scale requires positive prior SD when enabled")
        if self.include_parent_fraction_shift and self.parent_fraction_logit_sd <= 0:
            raise PETKineticError("parent fraction shift requires positive prior SD when enabled")
        if self.delay_mean < 0 or self.delay_sd <= 0 or self.delay_upper <= self.delay_mean:
            raise PETKineticError("invalid delay prior")
        if self.include_dispersion:
            if self.dispersion_tau_mean <= 0 or self.dispersion_tau_log_sd <= 0:
                raise PETKineticError("invalid dispersion prior")
            if self.dispersion_tau_lower <= 0 or self.dispersion_tau_upper <= self.dispersion_tau_lower:
                raise PETKineticError("invalid dispersion bounds")


@dataclass(frozen=True)
class JointPETFitResult:
    model: str
    parameter_names: tuple[str, ...]
    map_estimate: np.ndarray
    covariance: np.ndarray | None
    residual: np.ndarray
    success: bool
    message: str
    objective: float
    log_likelihood: float
    log_prior: float
    jacobian_rank: int
    condition_number: float
    credible_interval_95: np.ndarray | None
    posterior_draws: np.ndarray
    posterior_draw_acceptance: float
    diagnostics: dict[str, float]
    assumptions: tuple[str, ...]

    def parameter_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in zip(self.parameter_names, self.map_estimate)}


# ----------------------------- forward model -----------------------------


def _parent_input_at(aif: ArterialInputFunction, t: float, plasma_log_scale: float, parent_shift: float) -> float:
    source_t = float(t) - float(aif.delay)
    if source_t < aif.time_array[0]:
        return 0.0
    if source_t > aif.time_array[-1]:
        raise PETKineticError("AIF does not cover requested time")
    total = float(np.interp(source_t, aif.time_array, aif.total_plasma))
    fp = float(np.interp(source_t, aif.time_array, aif.parent_fraction))
    fp2 = float(_expit(_logit(np.array([fp]))[0] + parent_shift))
    return total * math.exp(plasma_log_scale) * fp2


def _dispersed_frame_model(
    schedule: PETFrameSchedule,
    aif: ArterialInputFunction,
    kinetic: np.ndarray,
    *,
    model: Literal["1TCM", "2TCM"],
    delay: float,
    tau: float,
    plasma_log_scale: float,
    parent_shift: float,
    vascular_fraction: float,
) -> np.ndarray:
    """Exact frame averages for a causal exponential input-dispersion filter."""
    if delay < 0 or tau <= 0:
        raise PETKineticError("delay must be >=0 and dispersion tau >0")

    # We use the AIF time grid as the piecewise-linear source grid, but overwrite
    # the AIF's nominal delay with the joint latent delay.
    max_time = float(schedule.end_array[-1])
    if max_time - delay > float(aif.time_array[-1]) + 1e-12:
        raise PETKineticError("AIF support is insufficient for the latent delay")

    # Build a shifted-source helper with local interpolation.  Segment boundaries
    # include PET frame boundaries and shifted AIF knots, exactly as Batch 17.
    shifted_knots = aif.time_array + delay
    bounds = np.unique(np.concatenate([
        np.array([0.0, max_time]),
        schedule.start_array,
        schedule.end_array,
        shifted_knots[(shifted_knots >= 0) & (shifted_knots <= max_time + 1e-12)],
    ]))
    bounds.sort()

    if model == "1TCM":
        if kinetic.size != 2:
            raise PETKineticError("1TCM requires K1,k2")
        k1, k2 = map(float, kinetic)
        if k1 <= 0 or k2 <= 0:
            raise PETKineticError("kinetic rates must be positive")
        # state = [dispersion state, tissue state]
        A = np.array([
            [-1.0 / tau, 0.0],
            [k1, -k2],
        ], dtype=float)
        B = np.array([1.0 / tau, 0.0], dtype=float)
        C = np.array([0.0, 1.0 - vascular_fraction], dtype=float)
    else:
        if kinetic.size != 4:
            raise PETKineticError("2TCM requires K1,k2,k3,k4")
        k1, k2, k3, k4 = map(float, kinetic)
        if min(k1, k2, k3, k4) <= 0:
            raise PETKineticError("kinetic rates must be positive")
        A = np.array([
            [-1.0 / tau, 0.0, 0.0],
            [k1, -(k2 + k3), k4],
            [0.0, k3, -k4],
        ], dtype=float)
        B = np.array([1.0 / tau, 0.0, 0.0], dtype=float)
        C = np.array([0.0, 1.0 - vascular_fraction, 1.0 - vascular_fraction], dtype=float)

    x = np.zeros(A.shape[0], dtype=float)
    out = np.zeros(schedule.n_frames, dtype=float)
    starts = schedule.start_array
    ends = schedule.end_array
    frame_idx = 0

    def u_at(t: float) -> float:
        # Joint latent delay replaces the nominal AIF delay.
        src = float(t) - delay
        if src < aif.time_array[0]:
            return 0.0
        if src > aif.time_array[-1]:
            raise PETKineticError("AIF support insufficient after latent delay")
        total = float(np.interp(src, aif.time_array, aif.total_plasma))
        fp = float(np.interp(src, aif.time_array, aif.parent_fraction))
        fp = float(_expit(_logit(np.array([fp]))[0] + parent_shift))
        return total * math.exp(plasma_log_scale) * fp

    for left, right in zip(bounds[:-1], bounds[1:]):
        left, right = float(left), float(right)
        if right <= left + 1e-14:
            continue
        while frame_idx < schedule.n_frames and left >= ends[frame_idx] - 1e-12:
            frame_idx += 1
        u0 = u_at(left)
        u1 = u_at(right)
        slope = (u1 - u0) / (right - left)

        m = x.size
        # augmented [x, u(t), 1, frame-integral]
        M = np.zeros((m + 3, m + 3), dtype=float)
        M[:m, :m] = A
        M[:m, m] = B
        M[m, m + 1] = slope
        M[m + 2, :m] = C
        M[m + 2, m] = float(vascular_fraction)
        z0 = np.zeros(m + 3, dtype=float)
        z0[:m] = x
        z0[m] = u0
        z0[m + 1] = 1.0
        z1 = expm(M * (right - left)) @ z0
        x = z1[:m]
        if frame_idx < schedule.n_frames:
            if left >= starts[frame_idx] - 1e-12 and right <= ends[frame_idx] + 1e-12:
                out[frame_idx] += float(z1[m + 2]) / (ends[frame_idx] - starts[frame_idx])

    return np.maximum(out, 0.0)


# -------------------------- joint MAP / Laplace ---------------------------


def _parameter_layout(spec: JointPETSpec) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    names = ["K1", "k2"] if spec.model == "1TCM" else ["K1", "k2", "k3", "k4"]
    lo = [1e-8] * len(names)
    hi = [10.0] * len(names)
    if spec.include_plasma_scale:
        names.append("plasma_log_scale")
        lo.append(-5.0); hi.append(5.0)
    if spec.include_parent_fraction_shift:
        names.append("parent_fraction_logit_shift")
        lo.append(-8.0); hi.append(8.0)
    names.append("delay")
    lo.append(0.0); hi.append(spec.delay_upper)
    if spec.include_dispersion:
        names.append("dispersion_tau")
        lo.append(spec.dispersion_tau_lower); hi.append(spec.dispersion_tau_upper)
    return tuple(names), np.asarray(lo), np.asarray(hi), np.asarray(lo, dtype=float)


def _priors(params: dict[str, float], spec: JointPETSpec) -> np.ndarray:
    r = []
    if spec.include_plasma_scale:
        r.append(params["plasma_log_scale"] / spec.plasma_log_scale_sd)
    if spec.include_parent_fraction_shift:
        r.append(params["parent_fraction_logit_shift"] / spec.parent_fraction_logit_sd)
    r.append(_truncated_normal_residual(params["delay"], spec.delay_mean, spec.delay_sd))
    if spec.include_dispersion:
        r.append(_positive_log_prior(params["dispersion_tau"], spec.dispersion_tau_mean, spec.dispersion_tau_log_sd))
    return np.asarray(r, dtype=float)


def _kinetic_from_p(params: dict[str, float], spec: JointPETSpec) -> np.ndarray:
    return np.asarray([params[n] for n in (("K1", "k2") if spec.model == "1TCM" else ("K1", "k2", "k3", "k4"))], dtype=float)


def _map_residual(
    p: np.ndarray,
    names: tuple[str, ...],
    lo: np.ndarray,
    aif: ArterialInputFunction,
    starts: Sequence[float],
    durations: Sequence[float],
    tac: np.ndarray,
    observation_sd: np.ndarray,
    spec: JointPETSpec,
) -> np.ndarray:
    params = {n: float(v) for n, v in zip(names, p)}
    pred = _dispersed_frame_model(
        PETFrameSchedule(tuple(starts), tuple(durations)), aif,
        _kinetic_from_p(params, spec), model=spec.model,
        delay=params["delay"],
        tau=params.get("dispersion_tau", 1e-3),
        plasma_log_scale=params.get("plasma_log_scale", 0.0),
        parent_shift=params.get("parent_fraction_logit_shift", 0.0),
        vascular_fraction=spec.vascular_fraction,
    )
    obs = (pred - tac) / observation_sd
    pr = _priors(params, spec)
    return np.concatenate([obs, pr])


def fit_joint_pet_map_laplace(
    frame_starts: Sequence[float],
    frame_durations: Sequence[float],
    tac: Sequence[float],
    aif: ArterialInputFunction,
    observation_sd: Sequence[float],
    *,
    spec: JointPETSpec | None = None,
    initial: Sequence[float] | None = None,
    laplace_draws: int = 2000,
    seed: int = 1729,
) -> JointPETFitResult:
    """Joint MAP fit and local Laplace posterior approximation."""
    spec = spec or JointPETSpec()
    starts = _as_1d(frame_starts, "frame_starts")
    durations = _as_1d(frame_durations, "frame_durations")
    tac_a = _as_1d(tac, "tac")
    sd = _as_1d(observation_sd, "observation_sd")
    if not (starts.size == durations.size == tac_a.size == sd.size):
        raise PETKineticError("frame_starts, frame_durations, tac and observation_sd must match")
    if np.any(sd <= 0) or np.any(tac_a < 0):
        raise PETKineticError("invalid TAC / observation SD")
    if np.any(np.diff(starts) < 0) or np.any(durations <= 0):
        raise PETKineticError("invalid frame schedule")

    names, lo, hi, _ = _parameter_layout(spec)
    if initial is None:
        p0 = []
        p0.extend([0.12, 0.20] if spec.model == "1TCM" else [0.11, 0.22, 0.07, 0.05])
        if spec.include_plasma_scale: p0.append(0.0)
        if spec.include_parent_fraction_shift: p0.append(0.0)
        p0.append(min(max(spec.delay_mean, 1e-6), spec.delay_upper - 1e-6))
        if spec.include_dispersion: p0.append(min(max(spec.dispersion_tau_mean, spec.dispersion_tau_lower * 2), spec.dispersion_tau_upper / 2))
        p0 = np.asarray(p0, dtype=float)
    else:
        p0 = np.asarray(initial, dtype=float)
    if p0.size != len(names):
        raise PETKineticError("initial parameter vector does not match joint layout")
    if np.any(p0 <= lo) or np.any(p0 >= hi):
        raise PETKineticError("initial parameters must be strictly inside bounds")

    def resid(p: np.ndarray) -> np.ndarray:
        return _map_residual(p, names, lo, aif, starts, durations, tac_a, sd, spec)

    fit = least_squares(resid, p0, bounds=(lo, hi), method="trf", x_scale="jac", max_nfev=1000)
    r = resid(fit.x)
    J = np.asarray(fit.jac, dtype=float)
    rank = int(np.linalg.matrix_rank(J))
    svals = np.linalg.svd(J, compute_uv=False)
    cond = float(svals[0] / svals[-1]) if svals.size and svals[-1] > 0 else math.inf
    cov = None
    ci = None
    if rank == len(names):
        try:
            cov = np.linalg.inv(J.T @ J)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(J.T @ J)
        sd_post = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
        ci = np.column_stack([fit.x - 1.96 * sd_post, fit.x + 1.96 * sd_post])
        ci[:, 0] = np.maximum(ci[:, 0], lo)
        ci[:, 1] = np.minimum(ci[:, 1], hi)

    n_obs = tac_a.size
    data_r = r[:n_obs]
    log_lik = -0.5 * float(np.sum(data_r**2 + np.log(2.0 * np.pi * sd**2)))
    log_prior = -0.5 * float(np.sum(r[n_obs:]**2))

    draws = np.empty((0, len(names)), dtype=float)
    acc = 0.0
    if cov is not None and laplace_draws > 0:
        rng = np.random.default_rng(seed)
        raw = rng.multivariate_normal(fit.x, cov, size=int(laplace_draws), method="svd")
        keep = np.all((raw >= lo) & (raw <= hi), axis=1)
        draws = raw[keep]
        acc = float(np.mean(keep))

    assumptions = (
        "Gaussian frame measurement likelihood with supplied observation_sd",
        "joint low-dimensional AIF nuisance corrections and kinetic parameters",
        "causal first-order exponential input dispersion",
        "piecewise-linear parent plasma input",
        "frame-integrated observation operator",
        "MAP plus local Laplace approximation; not full MCMC",
    )
    diagnostics = {
        "n_frames": float(n_obs),
        "data_weighted_rss": float(np.sum(data_r**2)),
        "posterior_draws": float(draws.shape[0]),
        "laplace_acceptance": acc,
        "jacobian_smallest_singular_value": float(svals[-1]) if svals.size else 0.0,
    }
    return JointPETFitResult(
        model=spec.model,
        parameter_names=names,
        map_estimate=np.asarray(fit.x, dtype=float),
        covariance=cov,
        residual=r,
        success=bool(fit.success),
        message=str(fit.message),
        objective=float(0.5 * np.sum(r**2)),
        log_likelihood=log_lik,
        log_prior=log_prior,
        jacobian_rank=rank,
        condition_number=cond,
        credible_interval_95=ci,
        posterior_draws=draws,
        posterior_draw_acceptance=acc,
        diagnostics=diagnostics,
        assumptions=assumptions,
    )


def transformed_parameter_correlation(result: JointPETFitResult) -> np.ndarray | None:
    """Return the local Laplace correlation matrix, if identifiable."""
    if result.covariance is None:
        return None
    v = np.sqrt(np.clip(np.diag(result.covariance), 0.0, np.inf))
    denom = np.outer(v, v)
    corr = np.divide(result.covariance, denom, out=np.zeros_like(result.covariance), where=denom > 0)
    np.fill_diagonal(corr, 1.0)
    return corr


def posterior_predictive(
    result: JointPETFitResult,
    aif: ArterialInputFunction,
    frame_starts: Sequence[float],
    frame_durations: Sequence[float],
    *,
    observation_sd: Sequence[float] | None = None,
    n_draws: int = 500,
    seed: int = 1729,
) -> np.ndarray:
    """Generate predictions from the accepted local-Laplace draws."""
    if result.posterior_draws.shape[0] == 0:
        raise PETKineticError("posterior predictive requires accepted Laplace draws")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, result.posterior_draws.shape[0], size=min(n_draws, result.posterior_draws.shape[0]))
    schedule = PETFrameSchedule(tuple(frame_starts), tuple(frame_durations))
    draws = []
    for i in idx:
        p = result.posterior_draws[int(i)]
        d = {n: float(v) for n, v in zip(result.parameter_names, p)}
        pred = _dispersed_frame_model(
            schedule, aif,
            np.asarray([d[n] for n in (("K1", "k2") if result.model == "1TCM" else ("K1", "k2", "k3", "k4"))]),
            model=result.model,
            delay=d["delay"],
            tau=d.get("dispersion_tau", 1e-3),
            plasma_log_scale=d.get("plasma_log_scale", 0.0),
            parent_shift=d.get("parent_fraction_logit_shift", 0.0),
            vascular_fraction=0.0,
        )
        if observation_sd is not None:
            sd = _as_1d(observation_sd, "observation_sd")
            if sd.size != pred.size or np.any(sd <= 0):
                raise PETKineticError("observation_sd mismatch")
            pred = pred + rng.normal(scale=sd)
        draws.append(pred)
    return np.asarray(draws, dtype=float)
