"""Public-source synchronization orchestration with offline-safe semantics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neuro_twin.data.adapters_public import OpenNeuroAdapter
from neuro_twin.data.openneuro_manifest import to_source_objects
from neuro_twin.data.snapshot import SourceObject, SourceSnapshot, build_snapshot, persist_snapshot


@dataclass(frozen=True)
class SyncResult:
    status: str
    snapshot: SourceSnapshot
    storage_path: str | None
    objects: tuple[SourceObject, ...]


def build_openneuro_snapshot(
    *,
    dataset_id: str,
    dataset_version: str,
    dataset_payload: dict[str, Any],
    file_payload: list[dict[str, Any]],
    endpoint: str = OpenNeuroAdapter.GRAPHQL,
    notes: tuple[str, ...] = (),
) -> tuple[SourceSnapshot, list[SourceObject]]:
    objects = to_source_objects(file_payload)
    snapshot = build_snapshot(
        source="openneuro",
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        endpoint=endpoint,
        request_payload={"dataset_id": dataset_id, "tag": dataset_version, "recursive": True},
        raw_payload={"dataset": dataset_payload, "files": file_payload},
        objects=objects,
        access_tier="public_api",
        license=dataset_payload.get("license"),
        notes=notes,
    )
    return snapshot, objects


def sync_openneuro_metadata(
    *,
    dataset_id: str,
    dataset_version: str,
    storage_root: str,
    client: OpenNeuroAdapter | None = None,
) -> SyncResult:
    adapter = client or OpenNeuroAdapter()
    data = adapter._graphql(
        """
        query Snapshot($datasetId: String!, $tag: String!) {
          snapshot(datasetId: $datasetId, tag: $tag) {
            id tag description { Name BIDSVersion DatasetDOI License }
            files(recursive: true) { id filename size directory annexed }
          }
        }
        """,
        {"datasetId": dataset_id, "tag": dataset_version},
    )
    snap = data.get("snapshot") or {}
    description = snap.get("description") or {}
    files = snap.get("files") or []
    snapshot, objects = build_openneuro_snapshot(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        dataset_payload=description,
        file_payload=files,
    )
    path = persist_snapshot(storage_root, snapshot, raw_payload={"snapshot": snap}, objects=objects)
    return SyncResult("LIVE", snapshot, str(path), tuple(objects))
