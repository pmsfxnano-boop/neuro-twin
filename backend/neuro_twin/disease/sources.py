"""Controlled clinical-data source contracts.

The adapter accepts authorized exports; it never embeds credentials or attempts
to bypass access controls. This keeps the scientific pipeline independent from
portal/session details while preserving provenance and PIT semantics.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ControlledSourceSpec:
    key: str
    disease: str
    provider: str
    access_tier: str
    canonical_fields: tuple[str, ...]
    visit_field: str
    subject_field: str
    notes: str


ADNI_PLASMA_V1 = ControlledSourceSpec(
    key="adni-upenn-plasma-biomarkers-v1",
    disease="alzheimer",
    provider="ADNI/LONI",
    access_tier="controlled",
    canonical_fields=("AB42", "AB40", "pTau217", "NfL", "GFAP"),
    visit_field="VISCODE",
    subject_field="PTID",
    notes=(
        "Current ADNI archive includes the UPENN plasma biomarker dataset for "
        "AB42, AB40, pTau217, NfL and GFAP; assay-specific interpretation must "
        "respect the ADNI data advisory and method/version provenance."
    ),
)


PPMI_CORE_V1 = ControlledSourceSpec(
    key="ppmi-core-biomarkers-clinical-v1",
    disease="parkinson",
    provider="Parkinson's Precision Medicine Initiative",
    access_tier="controlled",
    canonical_fields=("alphaSyn_SAA", "NfL", "MDS_UPDRS", "digital_function"),
    visit_field="visit",
    subject_field="subject_id",
    notes=(
        "PPMI provides individual-level clinical, imaging, sensor and biomarker "
        "data to qualified researchers after the study data-use process."
    ),
)


CONTROLLED_SOURCES: dict[str, ControlledSourceSpec] = {
    ADNI_PLASMA_V1.key: ADNI_PLASMA_V1,
    PPMI_CORE_V1.key: PPMI_CORE_V1,
}


def get_controlled_source(key: str) -> ControlledSourceSpec:
    return CONTROLLED_SOURCES[key]
