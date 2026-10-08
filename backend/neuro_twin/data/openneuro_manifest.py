"""Convert an OpenNeuro GraphQL snapshot file tree to a stable BIDS manifest."""
from __future__ import annotations

from pathlib import PurePosixPath
from typing import Iterable

from neuro_twin.data.snapshot import SourceObject


def classify_openneuro_file(path: str, *, directory: bool = False) -> str:
    if directory:
        return "directory"
    name = PurePosixPath(path).name.lower()
    if name == "participants.tsv":
        return "participants_tsv"
    if name == "dataset_description.json":
        return "dataset_description"
    if name.endswith("_events.tsv"):
        return "events_tsv"
    if name.endswith(".nii.gz") or name.endswith(".nii"):
        return "nifti"
    if name.endswith(".json"):
        return "json"
    if name.endswith(".tsv"):
        return "tsv"
    if name.endswith(".dcm"):
        return "dicom"
    return "other"


def to_source_objects(files: Iterable[dict]) -> list[SourceObject]:
    objects: list[SourceObject] = []
    for item in files:
        filename = str(item["filename"])
        objects.append(
            SourceObject(
                object_id=str(item["id"]),
                path=filename,
                kind=classify_openneuro_file(filename, directory=bool(item.get("directory"))),
                size=item.get("size"),
                annexed=item.get("annexed"),
            )
        )
    return objects


def summarize_bids_objects(objects: Iterable[SourceObject]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for obj in objects:
        summary[obj.kind] = summary.get(obj.kind, 0) + 1
    return dict(sorted(summary.items()))
