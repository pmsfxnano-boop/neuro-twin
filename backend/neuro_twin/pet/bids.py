"""Strict PET-BIDS metadata validation for quantitative analysis.

This module validates the metadata required to make temporal and quantitative PET
calculations auditable. It does not silently repair missing timing or decay
correction metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import math

import numpy as np


class PETMetadataError(ValueError):
    """Raised when PET metadata are insufficient or internally inconsistent."""


@dataclass(frozen=True)
class PETBIDSMetadata:
    units: str
    tracer_name: str
    time_zero: str
    scan_start: float
    injection_start: float
    frame_times_start: tuple[float, ...]
    frame_duration: tuple[float, ...]
    image_decay_corrected: bool
    image_decay_correction_time: float
    injected_radioactivity: float | None = None
    injected_radioactivity_units: str | None = None
    body_weight_kg: float | None = None
    recon_method_name: str | None = None
    tracer_radionuclide: str | None = None
    acquisition_mode: str | None = None

    @property
    def n_frames(self) -> int:
        return len(self.frame_times_start)

    @property
    def frame_ends(self) -> np.ndarray:
        return np.asarray(self.frame_times_start, dtype=float) + np.asarray(self.frame_duration, dtype=float)

    @property
    def frame_midpoints(self) -> np.ndarray:
        return np.asarray(self.frame_times_start, dtype=float) + 0.5 * np.asarray(self.frame_duration, dtype=float)


def _finite_array(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
        raise PETMetadataError(f"{name} must be a non-empty finite 1D array")
    return arr


def validate_pet_metadata(meta: Mapping[str, Any], *, n_frames: int | None = None) -> PETBIDSMetadata:
    required = [
        "Units", "TracerName", "TimeZero", "ScanStart", "InjectionStart",
        "FrameTimesStart", "FrameDuration", "ImageDecayCorrected", "ImageDecayCorrectionTime",
    ]
    missing = [k for k in required if k not in meta]
    if missing:
        raise PETMetadataError(f"missing required PET metadata: {missing}")

    starts = _finite_array(meta["FrameTimesStart"], "FrameTimesStart")
    durations = _finite_array(meta["FrameDuration"], "FrameDuration")
    if starts.size != durations.size:
        raise PETMetadataError("FrameTimesStart and FrameDuration must have equal length")
    if n_frames is not None and int(n_frames) != starts.size:
        raise PETMetadataError(f"metadata declares {starts.size} frames but image has {n_frames}")
    if np.any(durations <= 0):
        raise PETMetadataError("FrameDuration values must be > 0")
    if np.any(np.diff(starts) < 0):
        raise PETMetadataError("FrameTimesStart must be non-decreasing")
    ends = starts + durations
    if np.any(ends <= starts):
        raise PETMetadataError("every frame must have positive temporal extent")

    scan_start = float(meta["ScanStart"])
    injection_start = float(meta["InjectionStart"])
    decay_time = float(meta["ImageDecayCorrectionTime"])
    if not all(math.isfinite(x) for x in (scan_start, injection_start, decay_time)):
        raise PETMetadataError("ScanStart, InjectionStart, and ImageDecayCorrectionTime must be finite")
    if scan_start < 0 or injection_start < 0 or decay_time < 0:
        raise PETMetadataError("PET times relative to TimeZero must be non-negative")
    if decay_time > max(float(ends[-1]), scan_start, injection_start):
        raise PETMetadataError("ImageDecayCorrectionTime exceeds recorded PET timing window")

    units = str(meta["Units"]).strip()
    tracer = str(meta["TracerName"]).strip()
    if not units or not tracer:
        raise PETMetadataError("Units and TracerName cannot be empty")

    injected = meta.get("InjectedRadioactivity")
    injected_units = meta.get("InjectedRadioactivityUnits")
    if injected is not None:
        injected = float(injected)
        if not math.isfinite(injected) or injected <= 0:
            raise PETMetadataError("InjectedRadioactivity must be positive when supplied")
        if injected_units is None or not str(injected_units).strip():
            raise PETMetadataError("InjectedRadioactivityUnits required with InjectedRadioactivity")

    body_weight = meta.get("BodyWeight")
    if body_weight is None and "BodyWeightKg" in meta:
        body_weight = meta["BodyWeightKg"]
    if body_weight is not None:
        body_weight = float(body_weight)
        if not math.isfinite(body_weight) or body_weight <= 0:
            raise PETMetadataError("BodyWeight must be positive when supplied")

    return PETBIDSMetadata(
        units=units,
        tracer_name=tracer,
        time_zero=str(meta["TimeZero"]),
        scan_start=scan_start,
        injection_start=injection_start,
        frame_times_start=tuple(float(x) for x in starts),
        frame_duration=tuple(float(x) for x in durations),
        image_decay_corrected=bool(meta["ImageDecayCorrected"]),
        image_decay_correction_time=decay_time,
        injected_radioactivity=injected,
        injected_radioactivity_units=str(injected_units) if injected_units is not None else None,
        body_weight_kg=body_weight,
        recon_method_name=str(meta["ReconMethodName"]) if meta.get("ReconMethodName") else None,
        tracer_radionuclide=str(meta["TracerRadionuclide"]) if meta.get("TracerRadionuclide") else None,
        acquisition_mode=str(meta["AcquisitionMode"]) if meta.get("AcquisitionMode") else None,
    )
