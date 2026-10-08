"""Explicit observation/calibration registry.

The registry separates *what is measured* from *how a study-specific assay or
imaging feature is calibrated*. It intentionally does not claim that pTau217,
NfL, MRI, PET, etc. equal P/I/N/Q unless a model configuration explicitly says
so.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObservationChannel:
    name: str
    modality: str
    expected_unit: str | None
    description: str
    measurement_error_family: str = "gaussian"


DEFAULT_CHANNELS = {
    "pTau217": ObservationChannel("pTau217", "blood", None, "plasma phosphorylated tau 217"),
    "Abeta42": ObservationChannel("Abeta42", "blood", None, "amyloid beta 42"),
    "Abeta40": ObservationChannel("Abeta40", "blood", None, "amyloid beta 40"),
    "GFAP": ObservationChannel("GFAP", "blood", None, "glial fibrillary acidic protein"),
    "NfL": ObservationChannel("NfL", "blood", None, "neurofilament light"),
    "cognition": ObservationChannel("cognition", "clinical", None, "cognitive measure; instrument-specific"),
    "MRI": ObservationChannel("MRI", "mri_feature", None, "MRI-derived quantitative feature"),
    "PET": ObservationChannel("PET", "pet_feature", None, "PET-derived quantitative feature"),
}


@dataclass(frozen=True)
class LinearCalibration:
    """Assay/scanner calibration y_canonical = scale*y_raw + offset."""

    scale: float = 1.0
    offset: float = 0.0
    covariance: np.ndarray | None = None

    def apply(self, value: float) -> float:
        return self.scale * float(value) + self.offset
