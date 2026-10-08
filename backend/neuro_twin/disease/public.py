"""Public disease cohorts backed by OpenNeuro dataset metadata.

Large imaging and EEG files are not committed to NEURO-TWIN. The adapter loads
the small BIDS participants.tsv metadata from the OpenNeuroDatasets mirrors.
The byte-level SHA-256 is persisted so later runs can detect source drift.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from .cohorts import CohortRecord, CohortSummary, parse_participants_tsv, summarize_cohort


@dataclass(frozen=True)
class PublicCohortSpec:
    key: str
    disease: str
    dataset_id: str
    dataset_version: str
    source_name: str
    participants_url: str
    diagnosis_field: str
    diagnosis_map: dict[str, str]
    severity_fields: dict[str, str]
    age_field: str
    sex_field: str
    expected_n: int


PUBLIC_COHORTS: dict[str, PublicCohortSpec] = {
    "alzheimer_ds004504_v1.0.6": PublicCohortSpec(
        key="alzheimer_ds004504_v1.0.6",
        disease="Alzheimer disease",
        dataset_id="openneuro-ds004504",
        dataset_version="1.0.6",
        source_name="OpenNeuro ds004504",
        participants_url=(
            "https://raw.githubusercontent.com/OpenNeuroDatasets/"
            "ds004504/main/participants.tsv"
        ),
        diagnosis_field="Group",
        diagnosis_map={"A": "AD", "F": "FTD", "C": "CONTROL"},
        severity_fields={"mmse": "MMSE"},
        age_field="Age",
        sex_field="Gender",
        expected_n=88,
    ),
    "parkinson_ds005892_v1.0.0": PublicCohortSpec(
        key="parkinson_ds005892_v1.0.0",
        disease="Parkinson disease",
        dataset_id="openneuro-ds005892",
        dataset_version="1.0.0",
        source_name="OpenNeuro ds005892",
        participants_url=(
            "https://raw.githubusercontent.com/OpenNeuroDatasets/"
            "ds005892/main/participants.tsv"
        ),
        diagnosis_field="group",
        diagnosis_map={
            "Control": "CONTROL",
            "PD-NC": "PD_NC",
            "PD-MCI": "PD_MCI",
        },
        severity_fields={},
        age_field="age",
        sex_field="sex",
        expected_n=55,
    ),
}


def fetch_public_cohort(
    spec: PublicCohortSpec,
    *,
    timeout_seconds: float = 30.0,
    client: httpx.Client | None = None,
) -> tuple[list[CohortRecord], CohortSummary, dict[str, Any]]:
    owns_client = client is None
    client = client or httpx.Client(
        timeout=timeout_seconds,
        follow_redirects=True,
        headers={"User-Agent": "NEURO-TWIN-disease-cohort/0.1"},
    )
    try:
        response = client.get(spec.participants_url)
        response.raise_for_status()
        payload = response.content
    finally:
        if owns_client:
            client.close()

    records, raw_sha = parse_participants_tsv(
        payload,
        cohort=spec.key,
        source_version=spec.dataset_version,
        diagnosis_field=spec.diagnosis_field,
        diagnosis_map=spec.diagnosis_map,
        severity_fields=spec.severity_fields,
        age_field=spec.age_field,
        sex_field=spec.sex_field,
    )
    if len(records) != spec.expected_n:
        raise RuntimeError(
            f"{spec.key}: expected {spec.expected_n} participants, got {len(records)}"
        )

    summary = summarize_cohort(
        records,
        source_version=spec.dataset_version,
        raw_sha256=raw_sha,
    )
    provenance = {
        "dataset_id": spec.dataset_id,
        "dataset_version": spec.dataset_version,
        "source_name": spec.source_name,
        "retrieval_uri": spec.participants_url,
        "raw_sha256": raw_sha,
        "participant_count": len(records),
        "disease": spec.disease,
    }
    return records, summary, provenance
