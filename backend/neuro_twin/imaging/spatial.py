"""Explicit spatial integrity checks and opt-in label-map resampling.

No spatial transform is ever implicit.  A mismatch between source and target
images is either rejected or handled by an explicitly requested, provenance-bearing
nearest-neighbour resampling operation.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from typing import Any

import numpy as np
from scipy.ndimage import affine_transform

from neuro_twin.imaging.nifti import NiftiHeader


SPATIAL_VERSION = "0.1.0-b14"


@dataclass(frozen=True)
class SpatialCompatibility:
    aligned: bool
    shape_equal: bool
    affine_close: bool
    max_abs_affine_error: float
    source_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class SpatialTransformProvenance:
    operation: str
    version: str
    interpolation: str
    source_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    source_affine: tuple[tuple[float, ...], ...]
    target_affine: tuple[tuple[float, ...], ...]
    affine_voxel_map: tuple[tuple[float, ...], ...]
    offset: tuple[float, ...]
    source_label_hash: str
    result_hash: str
    parameters_hash: str


def _affine_tuple(a: np.ndarray) -> tuple[tuple[float, ...], ...]:
    a = np.asarray(a, dtype=float)
    return tuple(tuple(float(v) for v in row) for row in a)


def compare_spatial(
    source: NiftiHeader,
    target: NiftiHeader,
    *,
    affine_atol: float = 1e-5,
) -> SpatialCompatibility:
    reasons: list[str] = []
    src_aff = np.asarray(source.affine, dtype=float)
    tgt_aff = np.asarray(target.affine, dtype=float)
    shape_equal = tuple(source.shape) == tuple(target.shape)
    affine_err = float(np.max(np.abs(src_aff - tgt_aff)))
    affine_close = bool(np.allclose(src_aff, tgt_aff, atol=affine_atol, rtol=0.0))
    if not shape_equal:
        reasons.append("shape_mismatch")
    if not affine_close:
        reasons.append("affine_mismatch")
    return SpatialCompatibility(
        aligned=shape_equal and affine_close,
        shape_equal=shape_equal,
        affine_close=affine_close,
        max_abs_affine_error=affine_err,
        source_shape=tuple(source.shape),
        target_shape=tuple(target.shape),
        reasons=tuple(reasons),
    )


def _array_sha256(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(values)
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def resample_label_map_nearest(
    labels: np.ndarray,
    source_affine: np.ndarray,
    target_shape: tuple[int, int, int],
    target_affine: np.ndarray,
    *,
    fill_value: int = 0,
    parameters: dict[str, Any] | None = None,
) -> tuple[np.ndarray, SpatialTransformProvenance]:
    """Resample an integer label map into target voxel space.

    The mapping is computed as source_voxel = inv(source_affine) @ target_affine.
    Nearest-neighbour interpolation is mandatory because this function is for
    categorical label maps, not intensities.
    """
    source = np.asarray(labels)
    if source.ndim != 3 or len(target_shape) != 3:
        raise ValueError("label-map resampling currently supports 3D arrays only")
    if not np.issubdtype(source.dtype, np.integer):
        raise ValueError("label-map resampling requires integer labels")
    src_aff = np.asarray(source_affine, dtype=float)
    tgt_aff = np.asarray(target_affine, dtype=float)
    if src_aff.shape != (4, 4) or tgt_aff.shape != (4, 4):
        raise ValueError("source and target affine must be 4x4")
    if abs(float(np.linalg.det(src_aff[:3, :3]))) < 1e-12:
        raise ValueError("source affine is singular")
    mapping = np.linalg.inv(src_aff) @ tgt_aff
    matrix = mapping[:3, :3]
    offset = mapping[:3, 3]
    result = affine_transform(
        source,
        matrix=matrix,
        offset=offset,
        output_shape=tuple(int(v) for v in target_shape),
        order=0,
        mode="constant",
        cval=int(fill_value),
        prefilter=False,
    ).astype(source.dtype, copy=False)
    params = {
        "fill_value": int(fill_value),
        "interpolation": "nearest",
        **(parameters or {}),
    }
    params_json = json.dumps(params, sort_keys=True, separators=(",", ":")).encode()
    provenance_no_hash = {
        "operation": "label_map_resample",
        "version": SPATIAL_VERSION,
        "interpolation": "nearest",
        "source_shape": tuple(int(v) for v in source.shape),
        "target_shape": tuple(int(v) for v in target_shape),
        "source_affine": _affine_tuple(src_aff),
        "target_affine": _affine_tuple(tgt_aff),
        "affine_voxel_map": _affine_tuple(mapping),
        "offset": tuple(float(v) for v in offset),
        "source_label_hash": _array_sha256(source),
        "result_hash": _array_sha256(result),
        "parameters_hash": hashlib.sha256(params_json).hexdigest(),
    }
    return result, SpatialTransformProvenance(**provenance_no_hash)
