"""Canonical import of authorized ADNI/PPMI exports.

The parser is intentionally strict: unknown source columns are ignored only when
not requested, required canonical fields must be explicitly mapped, and duplicate
subject/visit rows are rejected so repeated measurements cannot be silently merged.
"""
from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ExportRecord:
    subject_id: str
    visit: str
    values: dict[str, float]
    source_row: int


def _float(value: str) -> float:
    result = float(value.strip())
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"non-finite numeric value: {value!r}")
    return result


def parse_authorized_export(
    payload: bytes,
    *,
    subject_field: str,
    visit_field: str,
    field_map: dict[str, str],
    required_fields: Iterable[str],
) -> tuple[list[ExportRecord], str]:
    raw_sha256 = hashlib.sha256(payload).hexdigest()
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
    if not rows:
        raise ValueError("export is empty")
    required_headers = {subject_field, visit_field, *field_map.values()}
    missing = sorted(required_headers - set(rows[0]))
    if missing:
        raise ValueError(f"export missing required columns: {missing}")

    required = tuple(required_fields)
    records: list[ExportRecord] = []
    seen: set[tuple[str, str]] = set()
    for row_number, row in enumerate(rows, start=2):
        subject = (row[subject_field] or "").strip()
        visit = (row[visit_field] or "").strip()
        if not subject or not visit:
            raise ValueError(f"missing subject/visit at row {row_number}")

        key = (subject, visit)
        if key in seen:
            raise ValueError(f"duplicate subject/visit: {subject!r}, {visit!r}")
        seen.add(key)

        values: dict[str, float] = {}
        for canonical in required:
            source_column = field_map.get(canonical)
            if source_column is None:
                raise ValueError(f"no source-column mapping for required field {canonical!r}")
            raw = (row[source_column] or "").strip()
            if raw:
                values[canonical] = _float(raw)

        records.append(
            ExportRecord(
                subject_id=subject,
                visit=visit,
                values=values,
                source_row=row_number,
            )
        )

    return records, raw_sha256
