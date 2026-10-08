"""PET quantitative features with explicit unit and uncertainty contracts.

No tracer-specific or Centiloid calibration is hard-coded. The caller must supply
the appropriate validated calibration for the tracer/analysis method.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np


class PETQuantificationError(ValueError):
    """Raised when PET quantitative assumptions are not satisfied."""


def frame_activity_time(frame_start: float, frame_duration: float) -> float:
    """Return frame midpoint in seconds relative to TimeZero."""
    frame_start = float(frame_start)
    frame_duration = float(frame_duration)
    if not (math.isfinite(frame_start) and math.isfinite(frame_duration)) or frame_duration <= 0:
        raise PETQuantificationError("frame start/duration must be finite with duration > 0")
    return frame_start + 0.5 * frame_duration


def regional_mean(image: np.ndarray, mask: np.ndarray) -> float:
    image = np.asarray(image, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if image.shape != mask.shape:
        raise PETQuantificationError("image and ROI mask must have identical shape")
    values = image[mask]
    if values.size == 0:
        raise PETQuantificationError("ROI selects zero voxels")
    if not np.all(np.isfinite(values)):
        raise PETQuantificationError("ROI contains non-finite PET values")
    return float(values.mean())


def region_suv(activity_bq_per_ml: float, body_weight_kg: float, injected_radioactivity_bq: float) -> float:
    """Compute weight-normalized SUV from Bq/mL, kg, and Bq.

    SUV here is the dimensionless tissue activity concentration normalized by
    injected activity per gram. The 1000 factor converts kg to g.
    """
    vals = (activity_bq_per_ml, body_weight_kg, injected_radioactivity_bq)
    if any(not math.isfinite(float(x)) or float(x) <= 0 for x in vals):
        raise PETQuantificationError("activity, body weight, and injected radioactivity must be positive")
    return float(activity_bq_per_ml * body_weight_kg * 1000.0 / injected_radioactivity_bq)


def region_suvr(target_value: float, reference_value: float) -> float:
    target_value = float(target_value)
    reference_value = float(reference_value)
    if not math.isfinite(target_value) or not math.isfinite(reference_value):
        raise PETQuantificationError("target/reference values must be finite")
    if reference_value <= 0:
        raise PETQuantificationError("reference value must be > 0 for SUVR")
    return target_value / reference_value


def propagated_ratio_uncertainty(
    numerator: float,
    denominator: float,
    numerator_sd: float,
    denominator_sd: float,
    *,
    covariance: float = 0.0,
) -> float:
    """First-order delta-method SD for R = numerator / denominator."""
    n, d, sn, sd, cov = map(float, (numerator, denominator, numerator_sd, denominator_sd, covariance))
    if not all(math.isfinite(x) for x in (n, d, sn, sd, cov)):
        raise PETQuantificationError("ratio uncertainty inputs must be finite")
    if d == 0 or sn < 0 or sd < 0:
        raise PETQuantificationError("invalid ratio uncertainty inputs")
    grad_n = 1.0 / d
    grad_d = -n / (d * d)
    var = grad_n * grad_n * sn * sn + grad_d * grad_d * sd * sd + 2.0 * grad_n * grad_d * cov
    if var < -1e-10:
        raise PETQuantificationError("negative propagated variance; check covariance")
    return float(math.sqrt(max(var, 0.0)))


def centiloid_transform(value: float, slope: float, intercept: float) -> float:
    """Apply a caller-supplied validated tracer/method-specific Centiloid mapping."""
    v, a, b = map(float, (value, slope, intercept))
    if not all(math.isfinite(x) for x in (v, a, b)):
        raise PETQuantificationError("Centiloid mapping parameters must be finite")
    return a * v + b
