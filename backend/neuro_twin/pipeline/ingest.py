"""Deterministic ingestion pipeline for canonical observations."""
from __future__ import annotations

from neuro_twin.data.adapters import FetchRequest, SourceAdapter
from neuro_twin.data.qc import QCResult, run_qc
from neuro_twin.data.schema import ObservationBatch


def ingest(adapter: SourceAdapter, request: FetchRequest, dataset_id: str, dataset_version: str) -> tuple[ObservationBatch, tuple[QCResult, ...]]:
    observations = []
    qc_results = []
    for raw in adapter.fetch(request):
        observation = adapter.normalize(raw)
        observation = adapter.validate(observation)
        qc = run_qc(observation)
        qc_results.append(qc)
        if qc.accepted:
            observations.append(observation)

    batch = ObservationBatch(
        observations=observations,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
    )
    return batch, tuple(qc_results)
