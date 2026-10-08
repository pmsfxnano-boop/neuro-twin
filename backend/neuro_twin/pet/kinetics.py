"""PET kinetic modeling with explicit assumptions and auditable diagnostics.

This module deliberately separates:
1) arterial input preparation,
2) compartment-model inference,
3) graphical analysis, and
4) reference-tissue inference.

No tracer-specific constants, reference regions, delays, or calibration mappings
are hidden here. The caller must provide them explicitly.

Time convention: all kinetic functions accept a common time unit, typically
minutes. Rate constants use the inverse of that same unit.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.interpolate import PchipInterpolator


PET_KINETICS_VERSION = "0.1.0-b16"


class PETKineticError(ValueError):
    """Raised for invalid PET kinetic inputs or failed model assumptions."""


def _as_1d(values: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
        raise PETKineticError(f"{name} must be a non-empty finite 1D array")
    return arr


def _validate_times(times: np.ndarray, name: str) -> None:
    if np.any(np.diff(times) <= 0):
        raise PETKineticError(f"{name} must be strictly increasing")


def _validate_dynamic_series(times, tac, name="TAC") -> tuple[np.ndarray, np.ndarray]:
    t = _as_1d(times, "times")
    c = _as_1d(tac, name)
    if t.size != c.size:
        raise PETKineticError("times and TAC length mismatch")
    _validate_times(t, "times")
    if np.any(c < 0):
        raise PETKineticError(f"{name} must be non-negative")
    return t, c


@dataclass(frozen=True)
class ArterialInputFunction:
    """Metabolite-corrected arterial plasma input function.

    ``total_plasma`` and ``parent_fraction`` are sampled at ``time``. The parent
    curve is total plasma multiplied by the measured parent fraction. A fixed
    delay may be supplied, but it is never estimated automatically in this
    module because delay and dispersion are tracer/protocol dependent.
    """

    time: tuple[float, ...]
    total_plasma: tuple[float, ...]
    parent_fraction: tuple[float, ...]
    delay: float = 0.0

    def __post_init__(self) -> None:
        t = _as_1d(self.time, "AIF time")
        cp = _as_1d(self.total_plasma, "total_plasma")
        fp = _as_1d(self.parent_fraction, "parent_fraction")
        if not (t.size == cp.size == fp.size):
            raise PETKineticError("AIF arrays must have equal length")
        _validate_times(t, "AIF time")
        if t[0] < 0:
            raise PETKineticError("AIF time must be non-negative")
        if np.any(cp < 0):
            raise PETKineticError("total plasma activity must be non-negative")
        if np.any((fp < 0) | (fp > 1)):
            raise PETKineticError("parent fraction must lie in [0, 1]")
        if not math.isfinite(float(self.delay)) or self.delay < 0:
            raise PETKineticError("AIF delay must be finite and >= 0")

    @property
    def time_array(self) -> np.ndarray:
        return np.asarray(self.time, dtype=float)

    @property
    def parent_plasma(self) -> np.ndarray:
        return np.asarray(self.total_plasma, dtype=float) * np.asarray(self.parent_fraction, dtype=float)

    def parent_curve(self, *, extrapolate_zero_before: bool = True) -> PchipInterpolator:
        t = self.time_array
        cp = self.parent_plasma
        if np.any(np.diff(t) <= 0):
            raise PETKineticError("AIF time must be strictly increasing")
        interp = PchipInterpolator(t, cp, extrapolate=False)
        return interp

    def at(self, query_time: Sequence[float]) -> np.ndarray:
        q = _as_1d(query_time, "query_time")
        # Fixed positive delay: tissue at t sees the AIF at t-delay.
        source_t = q - float(self.delay)
        out = np.zeros_like(q)
        valid = source_t >= self.time_array[0]
        if np.any(valid):
            if np.any(source_t[valid] > self.time_array[-1]):
                raise PETKineticError("AIF does not cover all requested times")
            out[valid] = np.asarray(self.parent_curve()(source_t[valid]), dtype=float)
        return out

    def with_parent_fraction(self, parent_fraction: Sequence[float]) -> "ArterialInputFunction":
        fp = _as_1d(parent_fraction, "parent_fraction")
        if fp.size != len(self.time):
            raise PETKineticError("parent_fraction length mismatch")
        return ArterialInputFunction(self.time, self.total_plasma, tuple(fp), self.delay)


@dataclass(frozen=True)
class KineticFitResult:
    model: str
    parameters: dict[str, float]
    covariance: np.ndarray | None
    parameter_names: tuple[str, ...]
    residual: np.ndarray
    weighted_rss: float
    aic: float | None
    aicc: float | None
    success: bool
    message: str
    condition_number: float | None
    jacobian_rank: int | None
    assumptions: tuple[str, ...]
    diagnostics: dict[str, float]

    @property
    def n_parameters(self) -> int:
        return len(self.parameter_names)


def _safe_covariance(jac: np.ndarray, residual: np.ndarray, n_obs: int, n_par: int, weights: np.ndarray | None) -> tuple[np.ndarray | None, float | None, int | None]:
    if jac.size == 0:
        return None, None, None
    rank = int(np.linalg.matrix_rank(jac))
    cond = None
    try:
        s = np.linalg.svd(jac, compute_uv=False)
        if s.size and s[-1] > 0:
            cond = float(s[0] / s[-1])
        else:
            cond = math.inf
    except np.linalg.LinAlgError:
        cond = math.inf
    if n_obs <= n_par or rank < n_par:
        return None, cond, rank
    jt_j = jac.T @ jac
    try:
        inv = np.linalg.inv(jt_j)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(jt_j)
    if weights is None:
        sigma2 = float(np.sum(residual**2) / (n_obs - n_par))
    else:
        sigma2 = 1.0
    cov = inv * sigma2
    return cov, cond, rank


def _information_criteria(residual: np.ndarray, n_par: int, sigma: np.ndarray | None) -> tuple[float | None, float | None]:
    n = residual.size
    if n == 0 or n <= n_par + 1:
        return None, None
    if sigma is not None:
        sig = np.asarray(sigma, dtype=float)
        if sig.shape != residual.shape or np.any(sig <= 0) or not np.all(np.isfinite(sig)):
            return None, None
        ll = float(-0.5 * np.sum((residual / sig) ** 2 + np.log(2.0 * np.pi * sig**2)))
    else:
        rss = float(np.sum(residual**2))
        if rss <= 0:
            return None, None
        # Gaussian ML with a single unknown residual scale; suitable for
        # within-problem comparison when the same TAC/error model is used.
        ll = float(-0.5 * n * (np.log(2.0 * np.pi * rss / n) + 1.0))
    aic = 2.0 * n_par - 2.0 * ll
    aicc = aic + (2.0 * n_par * (n_par + 1)) / (n - n_par - 1)
    return float(aic), float(aicc)


def _frame_sigma(sigma: Sequence[float] | None, n: int) -> np.ndarray | None:
    if sigma is None:
        return None
    arr = _as_1d(sigma, "observation_sd")
    if arr.size != n or np.any(arr <= 0):
        raise PETKineticError("observation_sd must be positive and match TAC length")
    return arr


def _interp_segment_coeff(t0: float, t1: float, input_fn, *, n_probe: int = 2) -> tuple[float, float]:
    """Return coefficients a,b for Cp(t0+u)=a+b*u over one segment."""
    cp0 = float(input_fn.at([t0])[0])
    cp1 = float(input_fn.at([t1])[0])
    dt = float(t1 - t0)
    return cp0, (cp1 - cp0) / dt


def _simulate_linear_system_piecewise_linear_input(
    times: np.ndarray,
    input_fn: ArterialInputFunction,
    A: np.ndarray,
    B: np.ndarray,
    initial_state: np.ndarray,
) -> np.ndarray:
    """Exact-per-segment propagation for dx/dt=A x + B Cp(t), linear Cp per segment.

    A 4x4 augmented exponential makes the propagation deterministic and much
    faster than repeatedly invoking an adaptive ODE solver inside least-squares.
    """
    t = np.asarray(times, dtype=float)
    n = t.size
    x = np.asarray(initial_state, dtype=float).copy()
    out = np.zeros((n, x.size), dtype=float)
    out[0] = x
    for i in range(1, n):
        t0, t1 = float(t[i - 1]), float(t[i])
        dt = t1 - t0
        a, b = _interp_segment_coeff(t0, t1, input_fn)
        m = x.size
        aug = np.zeros((m + 2, m + 2), dtype=float)
        aug[:m, :m] = A
        aug[:m, m] = np.asarray(B, dtype=float).reshape(m)
        aug[:m, m + 1] = 0.0
        # u(t) = a + b * tau, and d(1)/dt = 0; du/dt = b*1.
        aug[m, m + 1] = b
        from scipy.linalg import expm
        E = expm(aug * dt)
        z0 = np.concatenate([x, [a, 1.0]])
        x = E @ z0
        x = x[:m]
        out[i] = x
    return out


def _simulate_1tcm(times: np.ndarray, cp: ArterialInputFunction, k1: float, k2: float, vascular_fraction: float = 0.0) -> np.ndarray:
    if k1 <= 0 or k2 <= 0 or not (0 <= vascular_fraction < 1):
        raise PETKineticError("invalid 1TCM parameters")
    q = np.asarray(times, dtype=float)
    if q[0] < 0 or q[-1] > cp.time_array[-1] + cp.delay:
        raise PETKineticError("requested tissue times are outside AIF support")
    A = np.array([[-k2]], dtype=float)
    B = np.array([k1], dtype=float)
    tissue = _simulate_linear_system_piecewise_linear_input(q, cp, A, B, np.array([0.0]))[:, 0]
    whole = (1.0 - vascular_fraction) * tissue + vascular_fraction * cp.at(q)
    return np.maximum(whole, 0.0)


def _simulate_2tcm(times: np.ndarray, cp: ArterialInputFunction, k1: float, k2: float, k3: float, k4: float, vascular_fraction: float = 0.0) -> np.ndarray:
    if any(x <= 0 for x in (k1, k2, k3, k4)) or not (0 <= vascular_fraction < 1):
        raise PETKineticError("invalid 2TCM parameters")
    q = np.asarray(times, dtype=float)
    if q[0] < 0 or q[-1] > cp.time_array[-1] + cp.delay:
        raise PETKineticError("requested tissue times are outside AIF support")
    A = np.array([[-(k2 + k3), k4], [k3, -k4]], dtype=float)
    B = np.array([k1, 0.0], dtype=float)
    tissue = _simulate_linear_system_piecewise_linear_input(q, cp, A, B, np.array([0.0, 0.0]))
    whole = (1.0 - vascular_fraction) * tissue.sum(axis=1) + vascular_fraction * cp.at(q)
    return np.maximum(whole, 0.0)

def fit_plasma_compartment(
    times: Sequence[float],
    tac: Sequence[float],
    aif: ArterialInputFunction,
    *,
    model: Literal["1TCM", "2TCM"] = "1TCM",
    observation_sd: Sequence[float] | None = None,
    vascular_fraction: float = 0.0,
    initial: Sequence[float] | None = None,
) -> KineticFitResult:
    """Fit a reversible plasma-input 1TCM or 2TCM using nonlinear least squares.

    Initial conditions are zero at time 0. This function uses the measured
    frame midpoint TAC as the fitting target; frame-integration modeling is a
    separate future processing stage and must not be implied here.
    """
    t, y = _validate_dynamic_series(times, tac)
    if t[0] > 1e-12:
        raise PETKineticError("plasma-input compartment fitting requires TAC time to start at 0")
    sigma = _frame_sigma(observation_sd, y.size)

    if model == "1TCM":
        names = ("K1", "k2")
        p0 = np.asarray(initial if initial is not None else (0.10, 0.20), dtype=float)
        bounds_lo = np.array([1e-8, 1e-8])
        bounds_hi = np.array([10.0, 10.0])
        simulator = lambda z: _simulate_1tcm(t, aif, float(z[0]), float(z[1]), vascular_fraction)
        model_assumptions = ("reversible one-tissue compartment", "zero initial tissue concentration", "fixed vascular fraction")
    elif model == "2TCM":
        names = ("K1", "k2", "k3", "k4")
        p0 = np.asarray(initial if initial is not None else (0.10, 0.20, 0.05, 0.05), dtype=float)
        bounds_lo = np.full(4, 1e-8)
        bounds_hi = np.full(4, 10.0)
        simulator = lambda z: _simulate_2tcm(t, aif, *map(float, z), vascular_fraction)
        model_assumptions = ("reversible two-tissue compartment", "zero initial tissue concentration", "fixed vascular fraction")
    else:
        raise PETKineticError("model must be '1TCM' or '2TCM'")
    if p0.shape != bounds_lo.shape or np.any(p0 <= bounds_lo) or np.any(p0 >= bounds_hi):
        raise PETKineticError("initial parameter vector has wrong shape or invalid values")

    def residual(z):
        r = simulator(z) - y
        return r if sigma is None else r / sigma

    result = least_squares(residual, p0, bounds=(bounds_lo, bounds_hi), method="trf", x_scale="jac", max_nfev=600)
    r = residual(result.x)
    cov, cond, rank = _safe_covariance(result.jac, r, y.size, len(names), sigma)
    aic, aicc = _information_criteria(r, len(names), sigma)
    diagnostics = {
        "rmse": float(np.sqrt(np.mean((simulator(result.x) - y) ** 2))),
        "max_abs_residual": float(np.max(np.abs(simulator(result.x) - y))),
        "n_observations": float(y.size),
    }
    return KineticFitResult(
        model=model,
        parameters={name: float(value) for name, value in zip(names, result.x)},
        covariance=cov,
        parameter_names=names,
        residual=r,
        weighted_rss=float(np.sum(r**2)),
        aic=aic,
        aicc=aicc,
        success=bool(result.success),
        message=str(result.message),
        condition_number=cond,
        jacobian_rank=rank,
        assumptions=model_assumptions,
        diagnostics=diagnostics,
    )


def plasma_distribution_volume(result: KineticFitResult) -> float:
    if result.model == "1TCM":
        return result.parameters["K1"] / result.parameters["k2"]
    if result.model == "2TCM":
        k1, k2, k3, k4 = (result.parameters[x] for x in ("K1", "k2", "k3", "k4"))
        return (k1 / k2) * (1.0 + k3 / k4)
    raise PETKineticError("distribution volume is defined here only for plasma 1TCM/2TCM results")


def _cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y, dtype=float)
    if y.size > 1:
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def _interpolate_positive_curve(source_t: np.ndarray, source_y: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    if target_t[0] < source_t[0] - 1e-12 or target_t[-1] > source_t[-1] + 1e-12:
        raise PETKineticError("input curve does not cover target times")
    interp = PchipInterpolator(source_t, source_y, extrapolate=False)
    return np.asarray(interp(target_t), dtype=float)


@dataclass(frozen=True)
class GraphicalFitResult:
    method: str
    slope: float
    intercept: float
    slope_sd: float | None
    n_points: int
    t_star: float
    r2: float
    condition_number: float
    assumptions: tuple[str, ...]
    diagnostics: dict[str, float]


def _linear_graphical_fit(x: np.ndarray, y: np.ndarray, t_star: float, sigma: np.ndarray | None = None, method: str = "Logan") -> GraphicalFitResult:
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(np.asarray(x))
    if t_star > np.max(np.asarray([0.0, t_star])):
        pass
    sel = mask
    X = np.column_stack([x[sel], np.ones(np.sum(sel))])
    yy = y[sel]
    if yy.size < 3:
        raise PETKineticError("graphical analysis needs at least 3 post-t* points")
    if sigma is None:
        beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
        residual = yy - X @ beta
    else:
        w = 1.0 / np.asarray(sigma, dtype=float)[sel]
        Xw = X * w[:, None]
        yw = yy * w
        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        residual = yy - X @ beta
    svals = np.linalg.svd(X, compute_uv=False)
    cond = float(svals[0] / svals[-1]) if svals[-1] > 0 else math.inf
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((yy - yy.mean())**2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0
    sd = None
    if yy.size > 2 and sigma is None:
        s2 = ss_res / (yy.size - 2)
        cov = s2 * np.linalg.pinv(X.T @ X)
        sd = float(math.sqrt(max(cov[0, 0], 0.0)))
    return GraphicalFitResult(
        method=method,
        slope=float(beta[0]),
        intercept=float(beta[1]),
        slope_sd=sd,
        n_points=int(yy.size),
        t_star=float(t_star),
        r2=r2,
        condition_number=cond,
        assumptions=("linear asymptotic regime begins at or after t*",),
        diagnostics={"rss": ss_res},
    )


def logan_plasma(
    tissue_times: Sequence[float],
    tissue_tac: Sequence[float],
    plasma_times: Sequence[float],
    plasma_parent: Sequence[float],
    *,
    t_star: float,
) -> GraphicalFitResult:
    """Plasma-input Logan graphical analysis for reversible tracers.

    Returns the asymptotic slope estimate of V_T. The method is explicitly
    diagnostic: t* is supplied by the caller and is not optimized here.
    """
    t, ct = _validate_dynamic_series(tissue_times, tissue_tac)
    if t[0] > 1e-12:
        raise PETKineticError("plasma-input Logan requires tissue TAC time to start at 0")
    pt = _as_1d(plasma_times, "plasma_times")
    cp = _as_1d(plasma_parent, "plasma_parent")
    if pt.size != cp.size:
        raise PETKineticError("plasma_times/plasma_parent length mismatch")
    _validate_times(pt, "plasma_times")
    if np.any(cp < 0):
        raise PETKineticError("plasma parent input must be non-negative")
    cp_t = _interpolate_positive_curve(pt, cp, t)
    int_ct = _cumulative_trapezoid(ct, t)
    int_cp = _cumulative_trapezoid(cp_t, t)
    valid = t >= float(t_star)
    if np.any(ct[valid] <= 0):
        raise PETKineticError("tissue TAC must be > 0 over all Logan fit points")
    x = int_cp[valid] / ct[valid]
    y = int_ct[valid] / ct[valid]
    return _linear_graphical_fit(x, y, float(t_star), method="Logan-plasma")


def logan_reference(
    tissue_times: Sequence[float],
    tissue_tac: Sequence[float],
    reference_tac: Sequence[float],
    *,
    t_star: float,
) -> GraphicalFitResult:
    """Reference-region Logan graphical analysis returning DVR."""
    t, ct = _validate_dynamic_series(tissue_times, tissue_tac)
    if t[0] > 1e-12:
        raise PETKineticError("reference Logan requires tissue TAC time to start at 0")
    tr, cr = _validate_dynamic_series(tissue_times, reference_tac, name="reference TAC")
    if not np.allclose(tr, t, atol=1e-10, rtol=1e-10):
        raise PETKineticError("reference TAC must use the same time grid")
    int_ct = _cumulative_trapezoid(ct, t)
    int_cr = _cumulative_trapezoid(cr, t)
    valid = t >= float(t_star)
    if np.any(ct[valid] <= 0):
        raise PETKineticError("target TAC must be > 0 over all Logan fit points")
    x = int_cr[valid] / ct[valid]
    y = int_ct[valid] / ct[valid]
    return _linear_graphical_fit(x, y, float(t_star), method="Logan-reference")


def _reference_convolution(times: np.ndarray, reference: np.ndarray, decay_rate: float) -> np.ndarray:
    if times[0] > 1e-12:
        raise PETKineticError("reference TAC time must start at 0 for SRTM convolution")
    if decay_rate <= 0:
        raise PETKineticError("SRTM decay rate must be > 0")
    conv = np.zeros_like(reference, dtype=float)
    for i in range(1, times.size):
        tau = times[: i + 1]
        integrand = reference[: i + 1] * np.exp(-decay_rate * (times[i] - tau))
        conv[i] = np.trapezoid(integrand, tau)
    return conv


def srtm_predict(
    times: Sequence[float],
    reference_tac: Sequence[float],
    *,
    r1: float,
    k2: float,
    bp_nd: float,
) -> np.ndarray:
    """Standard 3-parameter SRTM operational equation.

    Convention used here follows the common form
    C_T = R1*C_R + [k2 - R1*k2/(1+BP)] C_R ⊗ exp[-k2 t/(1+BP)].
    """
    t, cr = _validate_dynamic_series(times, reference_tac, name="reference TAC")
    if r1 <= 0 or k2 <= 0 or bp_nd < 0:
        raise PETKineticError("invalid SRTM parameters")
    lam = k2 / (1.0 + bp_nd)
    conv = _reference_convolution(t, cr, lam)
    return r1 * cr + (k2 - r1 * k2 / (1.0 + bp_nd)) * conv


def fit_srtm(
    times: Sequence[float],
    target_tac: Sequence[float],
    reference_tac: Sequence[float],
    *,
    observation_sd: Sequence[float] | None = None,
    initial: Sequence[float] = (1.0, 0.2, 1.0),
) -> KineticFitResult:
    """Fit SRTM parameters R1, k2, BP_ND by constrained nonlinear least squares."""
    t, y = _validate_dynamic_series(times, target_tac, name="target TAC")
    tr, r = _validate_dynamic_series(times, reference_tac, name="reference TAC")
    if not np.allclose(tr, t, atol=1e-10, rtol=1e-10):
        raise PETKineticError("target/reference time grids must match")
    sigma = _frame_sigma(observation_sd, y.size)
    p0 = np.asarray(initial, dtype=float)
    lo = np.array([1e-8, 1e-8, 0.0])
    hi = np.array([10.0, 10.0, 100.0])
    if p0.shape != (3,) or np.any(p0 <= lo) or np.any(p0 >= hi):
        raise PETKineticError("invalid SRTM initial values")

    def residual(z):
        raw = srtm_predict(t, r, r1=float(z[0]), k2=float(z[1]), bp_nd=float(z[2])) - y
        return raw if sigma is None else raw / sigma

    result = least_squares(residual, p0, bounds=(lo, hi), method="trf", x_scale="jac", max_nfev=600)
    rr = residual(result.x)
    cov, cond, rank = _safe_covariance(result.jac, rr, y.size, 3, sigma)
    aic, aicc = _information_criteria(rr, 3, sigma)
    diagnostics = {
        "rmse": float(np.sqrt(np.mean((srtm_predict(t, r, r1=float(result.x[0]), k2=float(result.x[1]), bp_nd=float(result.x[2])) - y) ** 2))),
        "n_observations": float(y.size),
    }
    return KineticFitResult(
        model="SRTM",
        parameters={"R1": float(result.x[0]), "k2": float(result.x[1]), "BP_ND": float(result.x[2])},
        covariance=cov,
        parameter_names=("R1", "k2", "BP_ND"),
        residual=rr,
        weighted_rss=float(np.sum(rr**2)),
        aic=aic,
        aicc=aicc,
        success=bool(result.success),
        message=str(result.message),
        condition_number=cond,
        jacobian_rank=rank,
        assumptions=("reference region has appropriate nondisplaceable kinetics", "three-parameter SRTM operational form", "zero initial condition"),
        diagnostics=diagnostics,
    )


def srtm_dvr(result: KineticFitResult) -> float:
    if result.model != "SRTM":
        raise PETKineticError("result must be an SRTM fit")
    return 1.0 + result.parameters["BP_ND"]


def assess_graphical_stability(
    fit_fn,
    t_stars: Sequence[float],
) -> dict[str, float]:
    """Run a family of graphical fits and quantify slope stability across t*.

    The caller provides a zero-argument function returning a GraphicalFitResult
    for each candidate t*. The function deliberately reports instability rather
    than choosing a t* silently.
    """
    results = [fit_fn(float(ts)) for ts in t_stars]
    slopes = np.asarray([r.slope for r in results], dtype=float)
    if slopes.size < 2:
        raise PETKineticError("at least two t* values are needed for stability assessment")
    return {
        "slope_mean": float(slopes.mean()),
        "slope_sd": float(slopes.std(ddof=1)),
        "slope_cv": float(slopes.std(ddof=1) / max(abs(slopes.mean()), 1e-12)),
        "min_slope": float(slopes.min()),
        "max_slope": float(slopes.max()),
    }
