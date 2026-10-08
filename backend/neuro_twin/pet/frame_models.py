"""Frame-integrated dynamic PET kinetic models.

Batch 17 adds a strict distinction between the instantaneous model concentration
and the reconstructed PET frame measurement.  PET frames are modeled as

    y_i = (1/Delta_i) * integral_{t_i}^{t_i+Delta_i} C_PET(t) dt + eps_i

rather than silently equating the frame average to C_PET at the midpoint.

The forward model is exact conditional on a piecewise-linear parent plasma
input.  Segment boundaries include all PET frame boundaries and all shifted
AIF knots, so the state and frame integrals are propagated with augmented matrix
exponentials.  This is deterministic and differentiable in principle, while
remaining compatible with the existing SciPy least-squares backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

from .kinetics import (
    ArterialInputFunction,
    KineticFitResult,
    PETKineticError,
    _frame_sigma,
    _information_criteria,
    _safe_covariance,
)


PET_FRAME_KINETICS_VERSION = "0.1.0-b17"


@dataclass(frozen=True)
class PETFrameSchedule:
    """Non-overlapping dynamic PET frames in one consistent time unit."""

    start: tuple[float, ...]
    duration: tuple[float, ...]

    def __post_init__(self) -> None:
        s = np.asarray(self.start, dtype=float)
        d = np.asarray(self.duration, dtype=float)
        if s.ndim != 1 or d.ndim != 1 or s.size == 0 or s.size != d.size:
            raise PETKineticError("frame start/duration must be non-empty 1D arrays of equal length")
        if not np.all(np.isfinite(s)) or not np.all(np.isfinite(d)):
            raise PETKineticError("frame timing must be finite")
        if np.any(s < 0) or np.any(d <= 0):
            raise PETKineticError("frame starts must be >= 0 and durations > 0")
        e = s + d
        if np.any(np.diff(s) < 0):
            raise PETKineticError("frame starts must be non-decreasing")
        if s.size > 1 and np.any(s[1:] < e[:-1] - 1e-12):
            raise PETKineticError("PET frames must not overlap")

    @property
    def start_array(self) -> np.ndarray:
        return np.asarray(self.start, dtype=float)

    @property
    def duration_array(self) -> np.ndarray:
        return np.asarray(self.duration, dtype=float)

    @property
    def end_array(self) -> np.ndarray:
        return self.start_array + self.duration_array

    @property
    def midpoint_array(self) -> np.ndarray:
        return self.start_array + 0.5 * self.duration_array

    @property
    def n_frames(self) -> int:
        return len(self.start)

    @classmethod
    def from_bids(cls, frame_times_start: Sequence[float], frame_duration: Sequence[float]) -> "PETFrameSchedule":
        return cls(tuple(float(x) for x in frame_times_start), tuple(float(x) for x in frame_duration))


def _linear_parent_at(aif: ArterialInputFunction, query: float) -> float:
    source_t = float(query) - float(aif.delay)
    t = aif.time_array
    cp = aif.parent_plasma
    if source_t < t[0]:
        return 0.0
    if source_t > t[-1]:
        raise PETKineticError("AIF does not cover all requested frame times")
    return float(np.interp(source_t, t, cp))


def _shifted_aif_knots(aif: ArterialInputFunction, max_time: float) -> np.ndarray:
    shifted = np.asarray(aif.time_array, dtype=float) + float(aif.delay)
    return shifted[(shifted >= 0.0) & (shifted <= max_time + 1e-12)]


def _simulate_frame_linear_input(
    schedule: PETFrameSchedule,
    aif: ArterialInputFunction,
    A: np.ndarray,
    B: np.ndarray,
    output_C: np.ndarray,
    output_D: float,
    initial_state: np.ndarray,
) -> np.ndarray:
    """Propagate state and exact frame integrals under piecewise-linear input."""
    starts = schedule.start_array
    ends = schedule.end_array
    max_time = float(ends[-1])
    if max_time > float(aif.time_array[-1] + aif.delay) + 1e-12:
        raise PETKineticError("AIF support does not cover the final PET frame")

    boundaries = np.unique(
        np.concatenate(
            [
                np.array([0.0, max_time]),
                starts,
                ends,
                _shifted_aif_knots(aif, max_time),
            ]
        )
    )
    boundaries.sort()

    x = np.asarray(initial_state, dtype=float).copy()
    out = np.zeros(schedule.n_frames, dtype=float)
    frame_index = 0
    m = x.size
    C = np.asarray(output_C, dtype=float).reshape(m)
    B = np.asarray(B, dtype=float).reshape(m)

    for left, right in zip(boundaries[:-1], boundaries[1:]):
        left = float(left)
        right = float(right)
        if right - left <= 1e-14:
            continue

        while frame_index < schedule.n_frames and left >= ends[frame_index] - 1e-12:
            frame_index += 1

        a = _linear_parent_at(aif, left)
        b = (_linear_parent_at(aif, right) - a) / (right - left)

        # State: x; u; 1; integral(Cx + D*u).
        aug_dim = m + 3
        M = np.zeros((aug_dim, aug_dim), dtype=float)
        M[:m, :m] = A
        M[:m, m] = B
        M[m, m + 1] = b
        M[m + 2, :m] = C
        M[m + 2, m] = float(output_D)

        z0 = np.zeros(aug_dim, dtype=float)
        z0[:m] = x
        z0[m] = a
        z0[m + 1] = 1.0
        E = expm(M * (right - left))
        z1 = E @ z0
        x = z1[:m]
        integral = float(z1[m + 2])

        if frame_index < schedule.n_frames:
            s = starts[frame_index]
            e = ends[frame_index]
            if left >= s - 1e-12 and right <= e + 1e-12:
                out[frame_index] += integral / (e - s)

    return np.maximum(out, 0.0)


def frame_integrated_1tcm(
    frame_starts: Sequence[float],
    frame_durations: Sequence[float],
    aif: ArterialInputFunction,
    *,
    k1: float,
    k2: float,
    vascular_fraction: float = 0.0,
) -> np.ndarray:
    """Predict frame-average PET activity for a reversible 1TCM."""
    schedule = PETFrameSchedule(tuple(frame_starts), tuple(frame_durations))
    if k1 <= 0 or k2 <= 0 or not (0 <= vascular_fraction < 1):
        raise PETKineticError("invalid 1TCM parameters")
    A = np.array([[-k2]], dtype=float)
    B = np.array([k1], dtype=float)
    C = np.array([1.0 - vascular_fraction], dtype=float)
    return _simulate_frame_linear_input(schedule, aif, A, B, C, vascular_fraction, np.zeros(1))


def frame_integrated_2tcm(
    frame_starts: Sequence[float],
    frame_durations: Sequence[float],
    aif: ArterialInputFunction,
    *,
    k1: float,
    k2: float,
    k3: float,
    k4: float,
    vascular_fraction: float = 0.0,
) -> np.ndarray:
    """Predict frame-average PET activity for a reversible 2TCM."""
    schedule = PETFrameSchedule(tuple(frame_starts), tuple(frame_durations))
    if any(v <= 0 for v in (k1, k2, k3, k4)) or not (0 <= vascular_fraction < 1):
        raise PETKineticError("invalid 2TCM parameters")
    A = np.array([[-(k2 + k3), k4], [k3, -k4]], dtype=float)
    B = np.array([k1, 0.0], dtype=float)
    C = np.array([1.0 - vascular_fraction, 1.0 - vascular_fraction], dtype=float)
    return _simulate_frame_linear_input(schedule, aif, A, B, C, vascular_fraction, np.zeros(2))


def fit_frame_integrated_compartment(
    frame_starts: Sequence[float],
    frame_durations: Sequence[float],
    tac: Sequence[float],
    aif: ArterialInputFunction,
    *,
    model: Literal["1TCM", "2TCM"] = "1TCM",
    observation_sd: Sequence[float] | None = None,
    vascular_fraction: float = 0.0,
    initial: Sequence[float] | None = None,
) -> KineticFitResult:
    """Fit 1TCM/2TCM to frame-averaged PET measurements.

    The likelihood is conditional on the measured TAC representing the temporal
    average over each frame.  The forward model therefore integrates the model
    concentration over each frame rather than evaluating it at the midpoint.
    """
    schedule = PETFrameSchedule(tuple(frame_starts), tuple(frame_durations))
    y = np.asarray(tac, dtype=float)
    if y.ndim != 1 or y.size != schedule.n_frames or not np.all(np.isfinite(y)):
        raise PETKineticError("frame TAC must be a finite 1D array matching the frame schedule")
    if np.any(y < 0):
        raise PETKineticError("frame TAC must be non-negative")
    sigma = _frame_sigma(observation_sd, y.size)

    if model == "1TCM":
        names = ("K1", "k2")
        p0 = np.asarray(initial if initial is not None else (0.10, 0.20), dtype=float)
        lo = np.array([1e-8, 1e-8])
        hi = np.array([10.0, 10.0])
        simulator = lambda z: frame_integrated_1tcm(
            schedule.start_array, schedule.duration_array, aif,
            k1=float(z[0]), k2=float(z[1]), vascular_fraction=vascular_fraction,
        )
        assumptions = (
            "reversible one-tissue compartment",
            "zero initial tissue concentration at TimeZero",
            "frame-averaged likelihood",
            "piecewise-linear parent plasma input",
            "fixed vascular fraction",
        )
    elif model == "2TCM":
        names = ("K1", "k2", "k3", "k4")
        p0 = np.asarray(initial if initial is not None else (0.10, 0.20, 0.05, 0.05), dtype=float)
        lo = np.full(4, 1e-8)
        hi = np.full(4, 10.0)
        simulator = lambda z: frame_integrated_2tcm(
            schedule.start_array, schedule.duration_array, aif,
            k1=float(z[0]), k2=float(z[1]), k3=float(z[2]), k4=float(z[3]),
            vascular_fraction=vascular_fraction,
        )
        assumptions = (
            "reversible two-tissue compartment",
            "zero initial tissue concentration at TimeZero",
            "frame-averaged likelihood",
            "piecewise-linear parent plasma input",
            "fixed vascular fraction",
        )
    else:
        raise PETKineticError("model must be '1TCM' or '2TCM'")

    if p0.shape != lo.shape or np.any(p0 <= lo) or np.any(p0 >= hi):
        raise PETKineticError("initial parameter vector has wrong shape or invalid values")

    def residual(z: np.ndarray) -> np.ndarray:
        r = simulator(z) - y
        return r if sigma is None else r / sigma

    result = least_squares(
        residual, p0, bounds=(lo, hi), method="trf", x_scale="jac", max_nfev=800
    )
    r = residual(result.x)
    cov, cond, rank = _safe_covariance(result.jac, r, y.size, len(names), sigma)
    aic, aicc = _information_criteria(r, len(names), sigma)
    predicted = simulator(result.x)
    diagnostics = {
        "rmse": float(np.sqrt(np.mean((predicted - y) ** 2))),
        "max_abs_residual": float(np.max(np.abs(predicted - y))),
        "n_frames": float(y.size),
        "min_frame_duration": float(np.min(schedule.duration_array)),
        "max_frame_duration": float(np.max(schedule.duration_array)),
    }
    return KineticFitResult(
        model=f"frame-{model}",
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
        assumptions=assumptions,
        diagnostics=diagnostics,
    )


def compare_frame_models(results: Sequence[KineticFitResult]) -> list[dict[str, float | str | None]]:
    """Rank fitted models by AICc when all share the same observation model."""
    if not results:
        raise PETKineticError("at least one fitted model is required")
    if any(r.aicc is None for r in results):
        raise PETKineticError("all models require finite AICc for comparison")
    aiccs = np.asarray([float(r.aicc) for r in results], dtype=float)
    order = np.argsort(aiccs)
    best = float(np.min(aiccs))
    return [
        {
            "model": results[int(i)].model,
            "aicc": float(aiccs[int(i)]),
            "delta_aicc": float(aiccs[int(i)] - best),
            "rank": float(rank + 1),
        }
        for rank, i in enumerate(order)
    ]


def frame_integrated_rmse_against_midpoint(
    frame_starts: Sequence[float],
    frame_durations: Sequence[float],
    frame_average_tac: Sequence[float],
    instantaneous_midpoint_prediction: Sequence[float],
) -> float:
    """Quantify the approximation error of replacing a frame average by midpoint value."""
    schedule = PETFrameSchedule(tuple(frame_starts), tuple(frame_durations))
    y = np.asarray(frame_average_tac, dtype=float)
    m = np.asarray(instantaneous_midpoint_prediction, dtype=float)
    if y.shape != m.shape or y.size != schedule.n_frames:
        raise PETKineticError("frame average and midpoint arrays must match schedule")
    return float(np.sqrt(np.mean((y - m) ** 2)))


def frame_schedule_from_bids(
    frame_times_start_seconds: Sequence[float],
    frame_duration_seconds: Sequence[float],
    *,
    kinetic_time_unit: Literal["seconds", "minutes", "hours"] = "minutes",
) -> PETFrameSchedule:
    """Convert BIDS PET frame timing (seconds) into an explicit kinetic time unit.

    BIDS ``FrameTimesStart`` and ``FrameDuration`` are specified in seconds.
    This adapter makes the unit conversion explicit so rate constants in the
    kinetic model cannot silently be combined with seconds-based timing.
    """
    scale = {"seconds": 1.0, "minutes": 1.0 / 60.0, "hours": 1.0 / 3600.0}.get(kinetic_time_unit)
    if scale is None:
        raise PETKineticError("kinetic_time_unit must be 'seconds', 'minutes', or 'hours'")
    return PETFrameSchedule(
        tuple(float(x) * scale for x in frame_times_start_seconds),
        tuple(float(x) * scale for x in frame_duration_seconds),
    )
