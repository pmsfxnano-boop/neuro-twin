"""Deterministic imaging QC before quantitative feature extraction."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from .nifti import NiftiImage, NiftiHeader


@dataclass(frozen=True)
class ImagingQCResult:
    accepted: bool
    reasons: tuple[str, ...]
    shape: tuple[int, ...]
    voxel_volume_mm3: float | None
    finite_fraction: float
    nonzero_fraction: float | None
    affine_determinant: float | None


def run_nifti_qc(image: NiftiImage | NiftiHeader, *, require_3d: bool = True, reject_constant_image: bool = True, check_qform_sform: bool = True, affine_tolerance: float = 1e-5) -> ImagingQCResult:
    reasons: list[str] = []
    header = image if isinstance(image, NiftiHeader) else image.header
    shape = header.shape
    if require_3d and len(shape) != 3:
        reasons.append("expected_3d_image")
    try:
        affine = header.affine
        det = float(np.linalg.det(affine[:3, :3]))
        if not isfinite(det) or abs(det) < 1e-12:
            reasons.append("singular_affine")
    except Exception:
        det = None
        reasons.append("invalid_affine")
    try:
        voxel_volume = header.voxel_volume_mm3
    except Exception:
        voxel_volume = None
        reasons.append("invalid_voxel_volume")

    finite_fraction = 1.0
    nonzero_fraction: float | None = None
    if check_qform_sform and header.qform_code > 0 and header.sform_code > 0:
        try:
            q_aff = header.qform_affine
            s_aff = header.sform_affine
            if not np.allclose(q_aff, s_aff, atol=affine_tolerance, rtol=0.0):
                reasons.append("qform_sform_mismatch")
        except Exception:
            reasons.append("invalid_qform_sform")

    if isinstance(image, NiftiImage):
        finite = np.isfinite(image.data)
        finite_fraction = float(finite.mean()) if finite.size else 0.0
        if finite_fraction < 1.0:
            reasons.append("non_finite_voxels")
        if image.data.size:
            nonzero_fraction = float(np.count_nonzero(image.data) / image.data.size)
        if reject_constant_image and np.allclose(image.data, image.data.flat[0] if image.data.size else 0.0):
            reasons.append("constant_image")
    return ImagingQCResult(
        accepted=not reasons,
        reasons=tuple(reasons),
        shape=shape,
        voxel_volume_mm3=voxel_volume,
        finite_fraction=finite_fraction,
        nonzero_fraction=nonzero_fraction,
        affine_determinant=det,
    )
