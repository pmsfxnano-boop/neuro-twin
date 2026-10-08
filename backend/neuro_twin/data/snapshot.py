"""Immutable source-snapshot contracts and content-addressed persistence.

A scientific ingestion run must be reproducible even when an upstream source
changes.  This module records the exact request, source version, raw payload
hash and normalized manifest hash without interpreting the data clinically.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SourceObject(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    object_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    size: int | None = Field(default=None, ge=0)
    annexed: bool | None = None
    sha256: str | None = None


class SourceSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    snapshot_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    endpoint: str = Field(min_length=1)
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    retrieved_at: datetime
    access_tier: str = Field(min_length=1)
    license: str | None = None
    object_count: int = Field(ge=0)
    notes: tuple[str, ...] = ()


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(payload: Any) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def build_snapshot(
    *,
    source: str,
    dataset_id: str,
    dataset_version: str,
    endpoint: str,
    request_payload: Any,
    raw_payload: Any,
    objects: list[SourceObject],
    access_tier: str,
    license: str | None = None,
    notes: tuple[str, ...] = (),
    retrieved_at: datetime | None = None,
) -> SourceSnapshot:
    request_hash = sha256_json(request_payload)
    raw_hash = sha256_json(raw_payload)
    manifest_hash = sha256_json([x.model_dump(mode="json") for x in objects])
    retrieved_at = retrieved_at or datetime.now(timezone.utc)
    snapshot_material = {
        "source": source,
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "request_sha256": request_hash,
        "raw_sha256": raw_hash,
        "manifest_sha256": manifest_hash,
    }
    snapshot_id = hashlib.sha256(canonical_json_bytes(snapshot_material)).hexdigest()[:24]
    return SourceSnapshot(
        snapshot_id=snapshot_id,
        source=source,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        endpoint=endpoint,
        request_sha256=request_hash,
        raw_sha256=raw_hash,
        manifest_sha256=manifest_hash,
        retrieved_at=retrieved_at,
        access_tier=access_tier,
        license=license,
        object_count=len(objects),
        notes=notes,
    )


def persist_snapshot(
    root: str | Path,
    snapshot: SourceSnapshot,
    *,
    raw_payload: Any,
    objects: list[SourceObject],
) -> Path:
    root = Path(root)
    target = root / snapshot.source / snapshot.dataset_id / snapshot.dataset_version / snapshot.snapshot_id
    target.mkdir(parents=True, exist_ok=False)
    (target / "raw.json").write_bytes(canonical_json_bytes(raw_payload))
    (target / "manifest.json").write_bytes(canonical_json_bytes([o.model_dump(mode="json") for o in objects]))
    (target / "snapshot.json").write_bytes(canonical_json_bytes(snapshot.model_dump(mode="json")))
    return target
