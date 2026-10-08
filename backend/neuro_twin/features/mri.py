"""Quantitative MRI feature extraction with explicit lineage.

The extractor never maps an image feature to P/I/N/Q. It produces measurable
features that later enter the observation model through a versioned registry.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from neuro_twin.imaging.nifti import NiftiImage


EXTRACTOR_VERSION = "0.1.0-b13"


@dataclass(frozen=True)
class FeatureLineage:
    source_file: str
    source_sha256: str
    extractor: str
    extractor_version: str
    feature_name: str
    mask_file: str | None
    mask_sha256: str | None
    mask_semantics: str | None


@dataclass(frozen=True)
class MRIFeature:
    subject_id: str | None
    session_id: str | None
    feature: str
    value: float
    unit: str
    lineage: FeatureLineage


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _summary(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        raise ValueError("feature mask selects zero voxels")
    if not np.all(np.isfinite(x)):
        raise ValueError("selected image values must be finite")
    q05, q50, q95 = np.quantile(x, [0.05, 0.50, 0.95])
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if x.size > 1 else 0.0
    return {
        "mean_intensity": mean,
        "std_intensity": std,
        "median_intensity": float(q50),
        "p05_intensity": float(q05),
        "p95_intensity": float(q95),
        "coefficient_of_variation": float(std / abs(mean)) if mean != 0 else float("nan"),
    }


def extract_masked_intensity_features(
    image: NiftiImage,
    *,
    mask: np.ndarray | None = None,
    mask_path: str | Path | None = None,
    mask_semantics: str | None = None,
) -> list[MRIFeature]:
    """Extract reproducible scalar intensity features.

    A mask is optional for image-level descriptive features but mandatory for
    anatomy-specific interpretation. When a mask is supplied, it must match the
    image shape exactly. Registration/resampling is deliberately outside this
    extractor so that spatial transforms are separately versioned and auditable.
    """
    data = np.asarray(image.data, dtype=float)
    source_path = Path(image.path)
    source_hash = sha256_file(source_path)
    selected = data if mask is None else data[np.asarray(mask, dtype=bool)]
    if mask is not None and np.asarray(mask).shape != data.shape:
        raise ValueError("mask shape must exactly match image shape")
    if selected.size == 0:
        raise ValueError("selected region contains zero voxels")

    summary = _summary(selected)
    features = dict(summary)
    features["voxel_count"] = float(selected.size)
    features["voxel_volume_mm3"] = float(image.header.voxel_volume_mm3)
    features["volume_mm3"] = float(selected.size * image.header.voxel_volume_mm3) if mask is not None else float("nan")

    lineage = lambda name: FeatureLineage(
        source_file=str(source_path),
        source_sha256=source_hash,
        extractor="mri_masked_intensity",
        extractor_version=EXTRACTOR_VERSION,
        feature_name=name,
        mask_file=str(mask_path) if mask_path is not None else None,
        mask_sha256=sha256_file(mask_path) if mask_path is not None else None,
        mask_semantics=mask_semantics,
    )

    result: list[MRIFeature] = []
    for name, value in features.items():
        if not np.isfinite(value):
            # Image-level volume is intentionally undefined without an anatomical mask.
            continue
        unit = "mm3" if name.endswith("_mm3") else ("count" if name == "voxel_count" else "a.u.")
        result.append(MRIFeature(
            subject_id=None,
            session_id=None,
            feature=name,
            value=float(value),
            unit=unit,
            lineage=lineage(name),
        ))
    return result


def extract_region_label_features(
    image: NiftiImage,
    labels: np.ndarray,
    *,
    label_names: Mapping[int, str],
    labels_path: str | Path | None = None,
    labels_semantics: str = "integer_region_label_map",
) -> list[MRIFeature]:
    """Extract region volume + intensity summary for a pre-aligned label map."""
    labels = np.asarray(labels)
    if labels.shape != image.data.shape:
        raise ValueError("label map must be shape-aligned with the image")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("label map must contain integer labels")
    image_hash = sha256_file(image.path)
    label_hash = sha256_file(labels_path) if labels_path is not None else None
    records: list[MRIFeature] = []
    voxel_vol = float(image.header.voxel_volume_mm3)
    for label_id, label_name in sorted(label_names.items()):
        region = labels == int(label_id)
        count = int(region.sum())
        if count == 0:
            continue
        values = np.asarray(image.data[region], dtype=float)
        stats = _summary(values)
        base = f"roi_{label_name}"
        for suffix, value, unit in [
            ("volume", count * voxel_vol, "mm3"),
            ("mean_intensity", stats["mean_intensity"], "a.u."),
            ("std_intensity", stats["std_intensity"], "a.u."),
        ]:
            name = f"{base}_{suffix}"
            lineage = FeatureLineage(
                source_file=image.path,
                source_sha256=image_hash,
                extractor="mri_labelmap_region",
                extractor_version=EXTRACTOR_VERSION,
                feature_name=name,
                mask_file=str(labels_path) if labels_path is not None else None,
                mask_sha256=label_hash,
                mask_semantics=f"{labels_semantics}:label={label_id}:{label_name}",
            )
            records.append(MRIFeature(None, None, name, float(value), unit, lineage))
    return records


def features_to_jsonable(features: list[MRIFeature]) -> list[dict]:
    return [asdict(x) for x in features]
