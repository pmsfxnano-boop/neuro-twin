"""Canonical disease-cohort layer.

Diagnosis/group labels are kept separate from model observations. A disease
label can stratify or score a result, but it must not silently become an input
to the latent P-I-N-Q model.
"""
from __future__ import annotations

import csv
import hashlib
import io
import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CohortRecord:
    subject_id: str
    cohort: str
    diagnosis: str
    age_years: float | None
    sex: str | None
    severity: dict[str, float]
    source_record_id: str


@dataclass(frozen=True)
class CohortSummary:
    cohort: str
    source_version: str
    raw_sha256: str
    n_total: int
    diagnosis_counts: dict[str, int]
    age_mean: float | None
    age_sd: float | None
    severity_summary: dict[str, dict[str, float | None]]
    assumptions: tuple[str, ...] = ()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _float_or_none(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    result = float(stripped)
    if not math.isfinite(result):
        raise ValueError(f"non-finite numeric field: {value!r}")
    return result


def parse_participants_tsv(
    payload: bytes,
    *,
    cohort: str,
    source_version: str,
    diagnosis_field: str,
    diagnosis_map: dict[str, str],
    severity_fields: dict[str, str] | None = None,
    subject_field: str = "participant_id",
    age_field: str | None = "Age",
    sex_field: str | None = "sex",
) -> tuple[list[CohortRecord], str]:
    """Parse a BIDS participants.tsv snapshot.

    diagnosis_map is an explicit source-code mapping. Unknown source labels are
    rejected rather than silently collapsed.
    """
    raw_sha = sha256_bytes(payload)
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig")), delimiter="\t"))
    if not rows:
        raise ValueError("participants.tsv is empty")

    required = {subject_field, diagnosis_field}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"participants.tsv missing required fields: {sorted(missing)}")

    severity_fields = severity_fields or {}
    records: list[CohortRecord] = []
    for idx, row in enumerate(rows, start=2):
        subject_id = (row.get(subject_field) or "").strip()
        if not subject_id:
            raise ValueError(f"missing subject identifier at row {idx}")

        source_diag = (row.get(diagnosis_field) or "").strip()
        if source_diag not in diagnosis_map:
            raise ValueError(
                f"unknown diagnosis code {source_diag!r} at row {idx}; "
                f"known={sorted(diagnosis_map)}"
            )

        severity: dict[str, float] = {}
        for canonical_name, source_field in severity_fields.items():
            value = _float_or_none(row.get(source_field))
            if value is not None:
                severity[canonical_name] = value

        records.append(
            CohortRecord(
                subject_id=subject_id,
                cohort=cohort,
                diagnosis=diagnosis_map[source_diag],
                age_years=_float_or_none(row.get(age_field)) if age_field else None,
                sex=(
                    (row.get(sex_field) or "").strip() or None
                    if sex_field
                    else None
                ),
                severity=severity,
                source_record_id=f"participants.tsv:row={idx};subject={subject_id}",
            )
        )

    return records, raw_sha


def _mean_sd(values: Iterable[float]) -> tuple[float | None, float | None]:
    x = [float(v) for v in values if math.isfinite(float(v))]
    if not x:
        return None, None
    mean = sum(x) / len(x)
    if len(x) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in x) / (len(x) - 1)
    return mean, math.sqrt(var)


def summarize_cohort(
    records: list[CohortRecord],
    *,
    source_version: str,
    raw_sha256: str,
) -> CohortSummary:
    if not records:
        raise ValueError("cannot summarize an empty cohort")

    diagnoses: dict[str, int] = {}
    for record in records:
        diagnoses[record.diagnosis] = diagnoses.get(record.diagnosis, 0) + 1

    age_mean, age_sd = _mean_sd(
        r.age_years for r in records if r.age_years is not None
    )

    severity_names = sorted({name for r in records for name in r.severity})
    severity_summary: dict[str, dict[str, float | None]] = {}
    for name in severity_names:
        values = [r.severity[name] for r in records if name in r.severity]
        mean, sd = _mean_sd(values)
        severity_summary[name] = {
            "n": float(len(values)),
            "mean": mean,
            "sd": sd,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }

    return CohortSummary(
        cohort=records[0].cohort,
        source_version=source_version,
        raw_sha256=raw_sha256,
        n_total=len(records),
        diagnosis_counts=dict(sorted(diagnoses.items())),
        age_mean=age_mean,
        age_sd=age_sd,
        severity_summary=severity_summary,
        assumptions=(
            "Diagnosis/group labels are evaluation metadata, not model inputs.",
            "A cohort summary is descriptive and is not clinical validation.",
        ),
    )
