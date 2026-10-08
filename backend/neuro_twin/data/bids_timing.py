"""Strict BIDS acquisition-time resolution for point-in-time inference."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import csv


class BIDSTimingError(ValueError):
    """Raised when acquisition time cannot be resolved unambiguously."""


@dataclass(frozen=True)
class AcquisitionTimeResolution:
    acquisition_time: datetime
    source: str
    matched_filename: str


def parse_rfc3339_datetime(value: str) -> datetime:
    raw = value.strip()
    if not raw:
        raise BIDSTimingError("empty acquisition timestamp")
    normalized = raw.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise BIDSTimingError(f"invalid RFC3339 acquisition time: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def resolve_acquisition_time_from_scans_tsv(
    scans_tsv: str | Path,
    *,
    image_relative_path: str,
) -> AcquisitionTimeResolution:
    """Resolve an image acquisition time from BIDS *_scans.tsv `acq_time`.

    The lookup is exact on the relative filename as required for provenance. No
    filesystem mtime, processing time, or session date is substituted when the
    acquisition timestamp is absent.
    """
    scans_tsv = Path(scans_tsv)
    with scans_tsv.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if not rows or "filename" not in rows[0]:
        raise BIDSTimingError("scans.tsv must contain a filename column")
    matches = [r for r in rows if (r.get("filename") or "") == image_relative_path]
    if len(matches) != 1:
        raise BIDSTimingError(
            f"expected exactly one scans.tsv row for {image_relative_path!r}, found {len(matches)}"
        )
    value = matches[0].get("acq_time")
    if value in (None, "", "n/a", "N/A"):
        raise BIDSTimingError(f"acq_time missing for {image_relative_path!r}")
    return AcquisitionTimeResolution(
        acquisition_time=parse_rfc3339_datetime(value),
        source="BIDS_scans_tsv.acq_time",
        matched_filename=image_relative_path,
    )
