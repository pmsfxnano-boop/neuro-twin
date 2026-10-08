"""Versioned registry of disease datasets and endpoint contracts.

Dataset metadata are descriptive only. Clinical labels must be supplied by the
source adapter with provenance; this module never infers diagnosis from subject
identifiers, imaging intensities, or filenames.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class DiseaseCohortDefinition:
    key: str
    disease: str
    dataset_id: str
    dataset_version: str
    doi: str
    license: str
    modalities: tuple[str, ...]
    longitudinal: bool
    endpoint: Literal["binary_diagnosis", "severity", "progression"]
    positive_label: str
    negative_label: str
    label_source: str
    notes: tuple[str, ...] = ()


PARKINSON_ANT = DiseaseCohortDefinition(
    key="parkinson.ant_openneuro",
    disease="Parkinson disease",
    dataset_id="ds001907",
    dataset_version="3.2.0",
    doi="10.18112/openneuro.ds001907.v3.2.0",
    license="CC0",
    modalities=("MRI", "clinical"),
    longitudinal=True,
    endpoint="binary_diagnosis",
    positive_label="PD",
    negative_label="healthy_control",
    label_source="dataset-provided participant/group metadata; adapter must preserve exact source record",
    notes=(
        "46 participants and two sessions are reported for the published snapshot.",
        "Latest dataset README reports 21 Parkinson disease and 25 healthy-aging participants.",
        "The dataset page reports missing/irregular diffusion acquisitions for a subset; modality QC is mandatory.",
    ),
)

ALZHEIMER_EEG = DiseaseCohortDefinition(
    key="alzheimers.eeg_openneuro",
    disease="Alzheimer disease",
    dataset_id="ds004504",
    dataset_version="1.0.6",
    doi="10.18112/openneuro.ds004504.v1.0.6",
    license="CC0",
    modalities=("EEG", "clinical"),
    longitudinal=False,
    endpoint="binary_diagnosis",
    positive_label="AD",
    negative_label="healthy_control",
    label_source="dataset-provided diagnostic group metadata; adapter must preserve exact source record",
    notes=(
        "88 participants are reported: 36 AD, 23 FTD, 29 cognitively healthy.",
        "The binary AD-vs-control endpoint must exclude FTD rather than collapsing it into the negative class.",
        "MMSE is reported in the source and should be treated as an auxiliary clinical endpoint, not a diagnostic label.",
    ),
)
