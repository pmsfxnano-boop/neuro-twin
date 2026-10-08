"""BIDS tabular record extraction without loading image voxels."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BIDSParticipant:
    subject_id: str
    fields: dict[str, str]


@dataclass(frozen=True)
class BIDSEvent:
    subject_id: str
    session_id: str | None
    onset: float | None
    duration: float | None
    fields: dict[str, str]


def read_tsv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def participants(path: str | Path) -> list[BIDSParticipant]:
    rows = read_tsv(path)
    result: list[BIDSParticipant] = []
    for row in rows:
        subject = row.get("participant_id")
        if not subject:
            raise ValueError("participants.tsv requires participant_id")
        result.append(BIDSParticipant(subject, dict(row)))
    return result


def events(path: str | Path, *, subject_id: str, session_id: str | None = None) -> list[BIDSEvent]:
    result: list[BIDSEvent] = []
    for row in read_tsv(path):
        onset = float(row["onset"]) if row.get("onset") not in (None, "", "n/a", "NaN") else None
        duration = float(row["duration"]) if row.get("duration") not in (None, "", "n/a", "NaN") else None
        result.append(BIDSEvent(subject_id, session_id, onset, duration, dict(row)))
    return result


def read_sidecar(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError("BIDS JSON sidecar must be an object")
    return payload
