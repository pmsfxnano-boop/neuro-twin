"""Adapter boundary for external research sources."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

from neuro_twin.data.schema import Observation, AccessTier


@dataclass(frozen=True)
class FetchRequest:
    subject_ids: tuple[str, ...] = ()
    start: datetime | None = None
    end: datetime | None = None
    dataset_id: str | None = None
    query: dict[str, Any] | None = None


class SourceAdapter(ABC):
    """Strict adapter contract.

    Adapters may speak REST, GraphQL, Synapse, S3/NITRC bulk transport, or
    another mechanism. The downstream scientific stack must not know which.
    """

    name: str
    access_tier: AccessTier

    @abstractmethod
    def discover(self, request: FetchRequest) -> Iterable[dict[str, Any]]:
        """Discover source records/objects matching the request."""

    @abstractmethod
    def fetch(self, request: FetchRequest) -> Iterable[dict[str, Any]]:
        """Fetch raw source records. Never return normalized Observation here."""

    @abstractmethod
    def normalize(self, raw: dict[str, Any]) -> Observation:
        """Convert one raw source record into the canonical Observation."""

    def validate(self, observation: Observation) -> Observation:
        """Final schema validation hook; adapters may override for source QC."""
        return Observation.model_validate(observation.model_dump())
