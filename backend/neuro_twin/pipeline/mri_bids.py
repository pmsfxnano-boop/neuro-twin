"""BIDS-bound MRI extraction with explicit spatial policy."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from neuro_twin.features.mri import MRIFeature, extract_masked_intensity_features
from neuro_twin.imaging.bids_entities import BIDSFileRecord, parse_bids_image
from neuro_twin.imaging.nifti import NiftiImage, read_nifti
from neuro_twin.imaging.qc import ImagingQCResult, run_nifti_qc
from neuro_twin.imaging.spatial import SpatialCompatibility, SpatialTransformProvenance, compare_spatial, resample_label_map_nearest


@dataclass(frozen=True)
class BIDSBoundMRIResult:
    bids: BIDSFileRecord
    qc: ImagingQCResult
    spatial: SpatialCompatibility | None
    transform: SpatialTransformProvenance | None
    features: tuple[MRIFeature, ...]


def extract_bids_mri(
    image_path: str | Path,
    *,
    mask_image_path: str | Path | None = None,
    mask_labels: np.ndarray | None = None,
    mask_semantics: str | None = None,
    allow_resample_label_map: bool = False,
) -> BIDSBoundMRIResult:
    image = read_nifti(image_path, load_data=True)
    qc = run_nifti_qc(image)
    if not qc.accepted:
        raise ValueError(f"MRI QC failed: {', '.join(qc.reasons)}")

    bids = parse_bids_image(image_path)
    spatial = None
    transform = None
    mask = mask_labels
    mask_path = mask_image_path

    if mask_image_path is not None:
        mask_img = read_nifti(mask_image_path, load_data=True)
        mask_qc = run_nifti_qc(mask_img, reject_constant_image=False)
        if not mask_qc.accepted:
            raise ValueError(f"mask QC failed: {', '.join(mask_qc.reasons)}")
        spatial = compare_spatial(mask_img.header, image.header)
        if not spatial.aligned:
            if not allow_resample_label_map:
                raise ValueError(
                    "mask is not spatially aligned with image; enable explicit "
                    "allow_resample_label_map to perform nearest-neighbour resampling"
                )
            rounded = np.rint(mask_img.data)
            if not np.allclose(mask_img.data, rounded, atol=1e-6, rtol=0.0):
                raise ValueError("resampling requires an integer-valued label map")
            mask, transform = resample_label_map_nearest(
                rounded.astype(np.int32),
                mask_img.affine,
                image.data.shape,
                image.affine,
                parameters={"source_file": str(mask_image_path), "target_file": str(image_path)},
            )
        else:
            mask = np.rint(mask_img.data).astype(bool)

        if mask is None or not bool(np.any(mask)):
            raise ValueError("mask contains no selected voxels")

    features = extract_masked_intensity_features(
        image,
        mask=mask,
        mask_path=mask_path,
        mask_semantics=mask_semantics,
    )
    bound_features = tuple(
        MRIFeature(
            subject_id=bids.subject_id,
            session_id=bids.session_id,
            feature=f.feature,
            value=f.value,
            unit=f.unit,
            lineage=f.lineage,
        )
        for f in features
    )
    return BIDSBoundMRIResult(
        bids=bids,
        qc=qc,
        spatial=spatial,
        transform=transform,
        features=bound_features,
    )


def result_to_jsonable(result: BIDSBoundMRIResult) -> dict:
    return {
        "bids": asdict(result.bids),
        "qc": asdict(result.qc),
        "spatial": asdict(result.spatial) if result.spatial else None,
        "transform": asdict(result.transform) if result.transform else None,
        "features": [
            {
                "subject_id": f.subject_id,
                "session_id": f.session_id,
                "feature": f.feature,
                "value": f.value,
                "unit": f.unit,
                "lineage": asdict(f.lineage),
            }
            for f in result.features
        ],
    }
