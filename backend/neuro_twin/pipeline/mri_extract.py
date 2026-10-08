"""Orchestration boundary for versioned MRI feature extraction."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

from neuro_twin.features.mri import MRIFeature, extract_masked_intensity_features
from neuro_twin.imaging.nifti import read_nifti
from neuro_twin.imaging.qc import ImagingQCResult, run_nifti_qc


@dataclass(frozen=True)
class MRIExtractionResult:
    source_file: str
    qc: ImagingQCResult
    features: tuple[MRIFeature, ...]


def extract_mri(path: str | Path, *, mask=None, mask_path=None, mask_semantics=None) -> MRIExtractionResult:
    image = read_nifti(path, load_data=True)
    qc = run_nifti_qc(image)
    if not qc.accepted:
        raise ValueError(f"MRI QC failed: {', '.join(qc.reasons)}")
    features = extract_masked_intensity_features(
        image,
        mask=mask,
        mask_path=mask_path,
        mask_semantics=mask_semantics,
    )
    return MRIExtractionResult(str(path), qc, tuple(features))


def extraction_to_jsonable(result: MRIExtractionResult) -> dict:
    return {
        "source_file": result.source_file,
        "qc": asdict(result.qc),
        "features": [asdict(f) for f in result.features],
    }
