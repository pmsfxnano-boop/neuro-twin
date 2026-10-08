"""PET quantitative feature extraction with explicit timing/provenance.

The extractor operates on already validated PET-BIDS metadata and NIfTI arrays.
It does not perform tracer-specific kinetic modeling or infer reference regions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import hashlib

import numpy as np

from neuro_twin.imaging.nifti import NiftiImage
from neuro_twin.pet.bids import PETBIDSMetadata
from neuro_twin.pet.quantification import PETQuantificationError, region_suv, region_suvr


PET_EXTRACTOR_VERSION = "0.1.0-b15"


@dataclass(frozen=True)
class PETFeatureLineage:
    source_file: str
    source_sha256: str
    extractor: str
    extractor_version: str
    feature_name: str
    tracer_name: str
    frame_index: int | None
    reference_region: str | None


@dataclass(frozen=True)
class PETFeature:
    subject_id: str | None
    session_id: str | None
    feature: str
    value: float
    unit: str
    lineage: PETFeatureLineage
    acquisition_time_sec: float
    availability_time_sec: float


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _check_image(image: NiftiImage, metadata: PETBIDSMetadata) -> tuple[np.ndarray, int]:
    data = np.asarray(image.data, dtype=float)
    if data.ndim not in (3, 4):
        raise PETQuantificationError("PET image must be 3D or 4D")
    frames = 1 if data.ndim == 3 else data.shape[3]
    if frames != metadata.n_frames:
        raise PETQuantificationError("PET image frame count does not match PET-BIDS metadata")
    if not np.all(np.isfinite(data)):
        raise PETQuantificationError("PET image contains non-finite values")
    if not metadata.image_decay_corrected:
        raise PETQuantificationError(
            "quantitative PET extraction requires ImageDecayCorrected=true; "
            "raw decay correction is a separate versioned processing stage"
        )
    return data, frames


def extract_roi_tac(
    image: NiftiImage,
    mask: np.ndarray,
    metadata: PETBIDSMetadata,
    *,
    roi_name: str,
    subject_id: str | None = None,
    session_id: str | None = None,
) -> list[PETFeature]:
    """Extract a regional time-activity curve (frame mean activity)."""
    data, frames = _check_image(image, metadata)
    mask = np.asarray(mask, dtype=bool)
    spatial_shape = data.shape[:3]
    if mask.shape != spatial_shape:
        raise PETQuantificationError("ROI mask must match PET spatial shape")
    if not np.any(mask):
        raise PETQuantificationError("ROI mask is empty")
    source_hash = sha256_file(image.path)
    result: list[PETFeature] = []
    for i in range(frames):
        volume = data if data.ndim == 3 else data[..., i]
        value = float(volume[mask].mean())
        result.append(
            PETFeature(
                subject_id=subject_id,
                session_id=session_id,
                feature=f"tac_{roi_name}_frame_{i:03d}",
                value=value,
                unit=metadata.units,
                lineage=PETFeatureLineage(
                    source_file=image.path,
                    source_sha256=source_hash,
                    extractor="pet_roi_tac",
                    extractor_version=PET_EXTRACTOR_VERSION,
                    feature_name=f"tac_{roi_name}_frame_{i:03d}",
                    tracer_name=metadata.tracer_name,
                    frame_index=i,
                    reference_region=None,
                ),
                acquisition_time_sec=float(metadata.frame_midpoints[i]),
                availability_time_sec=float(metadata.frame_ends[i]),
            )
        )
    return result


def compute_roi_suv(
    activity_value: float,
    metadata: PETBIDSMetadata,
) -> float:
    if metadata.injected_radioactivity is None or metadata.injected_radioactivity_units is None:
        raise PETQuantificationError("InjectedRadioactivity is required for SUV")
    if metadata.body_weight_kg is None:
        raise PETQuantificationError("BodyWeight is required for weight-normalized SUV")
    if metadata.units.lower() not in {"bq/ml", "bq/ml"}:
        raise PETQuantificationError(f"SUV backend expects Units='Bq/mL', got {metadata.units!r}")
    return region_suv(activity_value, metadata.body_weight_kg, metadata.injected_radioactivity)


def compute_suvr(target_value: float, reference_value: float) -> float:
    return region_suvr(target_value, reference_value)
