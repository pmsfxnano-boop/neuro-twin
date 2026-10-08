"""Small, metadata-only BIDS readers for OpenNeuro/OASIS-like exports.

The reader deliberately does not load imaging voxels. It extracts a stable
metadata/event index first; image processing belongs to a separate versioned
feature-extraction job.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError("BIDS JSON sidecar must be an object")
    return payload


def read_events_tsv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def index_bids_tree(root: str | Path) -> dict[str, list[str]]:
    root = Path(root)
    result: dict[str, list[str]] = {"participants": [], "events": [], "json": [], "nifti": []}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        lower = name.lower()
        if name == "participants.tsv":
            result["participants"].append(str(path.relative_to(root)))
        elif lower.endswith("_events.tsv"):
            result["events"].append(str(path.relative_to(root)))
        elif lower.endswith(".json"):
            result["json"].append(str(path.relative_to(root)))
        elif lower.endswith((".nii", ".nii.gz")):
            result["nifti"].append(str(path.relative_to(root)))
    return result
