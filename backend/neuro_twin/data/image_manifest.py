"""Stable image manifest extracted from BIDS tree metadata.

No voxels are loaded. The manifest is the contract between ingestion and a
separate, versioned imaging-feature worker.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from neuro_twin.data.bids import index_bids_tree, read_json


@dataclass(frozen=True)
class ImageFileRecord:
    relative_path: str
    suffix: str
    subject_id: str | None
    session_id: str | None
    modality: str | None
    task: str | None
    run: str | None
    sidecar: str | None
    size_bytes: int
    metadata: dict[str, Any]



def _entities(stem: str) -> dict[str, str]:
    entities: dict[str, str] = {}
    for token in stem.split("_"):
        if "-" in token:
            key, value = token.split("-", 1)
            if key and value:
                entities[key] = value
    return entities


def build_image_manifest(root: str | Path) -> list[ImageFileRecord]:
    root = Path(root)
    tree = index_bids_tree(root)
    json_by_stem = {Path(p).stem: p for p in tree["json"]}
    records: list[ImageFileRecord] = []
    for rel in tree["nifti"]:
        path = root / rel
        name = path.name
        stem = name[:-7] if name.endswith(".nii.gz") else name[:-4]
        e = _entities(stem)
        sidecar_rel = json_by_stem.get(stem)
        metadata = read_json(root / sidecar_rel) if sidecar_rel else {}
        modality = e.get("suffix")
        if modality is None:
            # BIDS suffix is the final non-entity token.
            modality = stem.split("_")[-1]
        records.append(ImageFileRecord(
            relative_path=rel,
            suffix=modality,
            subject_id=e.get("sub"),
            session_id=e.get("ses"),
            modality=modality,
            task=e.get("task"),
            run=e.get("run"),
            sidecar=sidecar_rel,
            size_bytes=path.stat().st_size,
            metadata=metadata,
        ))
    return records


def manifest_to_jsonable(records: list[ImageFileRecord]) -> list[dict[str, Any]]:
    return [asdict(r) for r in records]
