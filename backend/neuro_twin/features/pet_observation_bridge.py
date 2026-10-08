"""Convert PET features into canonical point-in-time observations."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from neuro_twin.data.schema import Observation, ObservationKind, Provenance, QualityFlags
from neuro_twin.features.pet import PETFeature


@dataclass(frozen=True)
class PETObservationContext:
    source_name: str
    dataset_id: str
    dataset_version: str
    acquisition_datetime: datetime
    ingest_time: datetime
    source_record_id: str
    retrieval_uri: str | None = None
    license: str | None = None
    access_tier: str = "public_dataset"
    processing_pipeline: str = "neuro_twin.pet_feature_extractor"
    processing_version: str = "0.1.0-b15"
    center: str | None = None


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def pet_feature_to_observation(
    feature: PETFeature,
    *,
    subject_id: str,
    event_id: str,
    context: PETObservationContext,
    result_time: datetime,
    uncertainty: float | None = None,
    extra: dict | None = None,
) -> Observation:
    acquisition = _utc(context.acquisition_datetime)
    result = _utc(result_time)
    ingest = _utc(context.ingest_time)
    if result < acquisition:
        raise ValueError("result_time cannot precede acquisition time")
    if ingest < result:
        raise ValueError("ingest_time cannot precede result_time")
    extras = {
        "dataset_id": context.dataset_id,
        "pet": {
            "tracer_name": feature.lineage.tracer_name,
            "frame_index": feature.lineage.frame_index,
            "acquisition_time_sec": feature.acquisition_time_sec,
            "frame_end_sec": feature.availability_time_sec,
            "reference_region": feature.lineage.reference_region,
        },
        "feature_lineage": {
            "extractor": feature.lineage.extractor,
            "extractor_version": feature.lineage.extractor_version,
        },
        "point_in_time": {
            "acquisition_datetime": acquisition.isoformat(),
            "result_time": result.isoformat(),
            "available_at": result.isoformat(),
        },
    }
    if extra:
        extras.update(extra)
    return Observation(
        subject_id=subject_id,
        event_id=event_id,
        acquisition_time=acquisition,
        result_time=result,
        ingest_time=ingest,
        source=context.source_name,
        center=context.center,
        modality="PET",
        observation_kind=ObservationKind.PET_FEATURE,
        feature=feature.feature,
        value=float(feature.value),
        unit=feature.unit,
        uncertainty=uncertainty,
        quality=QualityFlags(schema_valid=True, temporal_integrity=True, provenance_complete=True),
        provenance=Provenance(
            source_name=context.source_name,
            source_version=context.dataset_version,
            source_record_id=context.source_record_id,
            retrieval_uri=context.retrieval_uri,
            processing_pipeline=context.processing_pipeline,
            processing_version=context.processing_version,
            license=context.license,
            access_tier=context.access_tier,
            raw_hash=feature.lineage.source_sha256,
        ),
        extra=extras,
    )
