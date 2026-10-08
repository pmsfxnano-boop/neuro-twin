"""Convert versioned MRI features into canonical point-in-time observations."""
from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass

from neuro_twin.data.schema import Observation, ObservationKind, Provenance, QualityFlags
from neuro_twin.features.mri import MRIFeature


@dataclass(frozen=True)
class ImagingObservationContext:
    source_name: str
    dataset_id: str
    dataset_version: str
    acquisition_time: datetime
    processing_time: datetime
    ingest_time: datetime
    source_record_id: str
    retrieval_uri: str | None = None
    license: str | None = None
    access_tier: str = "public_dataset"
    processing_pipeline: str = "neuro_twin.mri_feature_extractor"
    processing_version: str = "0.1.0-b14"
    center: str | None = None
    assay: str | None = None


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def feature_to_observation(
    feature: MRIFeature,
    *,
    subject_id: str,
    event_id: str,
    context: ImagingObservationContext,
) -> Observation:
    acquisition = _ensure_utc(context.acquisition_time)
    processing = _ensure_utc(context.processing_time)
    ingest = _ensure_utc(context.ingest_time)
    if processing < acquisition:
        raise ValueError("processing_time cannot precede acquisition_time")
    if ingest < processing:
        raise ValueError("ingest_time cannot precede processing_time")
    return Observation(
        subject_id=subject_id,
        event_id=event_id,
        acquisition_time=acquisition,
        result_time=processing,
        ingest_time=ingest,
        source=context.source_name,
        center=context.center,
        modality="MRI",
        assay=context.assay,
        observation_kind=ObservationKind.MRI_FEATURE,
        feature=feature.feature,
        value=float(feature.value),
        unit=feature.unit,
        uncertainty=None,
        missing=False,
        quality=QualityFlags(
            schema_valid=True,
            temporal_integrity=True,
            provenance_complete=True,
        ),
        provenance=Provenance(
            source_name=context.source_name,
            source_version=context.dataset_version,
            source_record_id=context.source_record_id,
            retrieval_uri=context.retrieval_uri,
            raw_hash=feature.lineage.source_sha256,
            processing_pipeline=context.processing_pipeline,
            processing_version=context.processing_version,
            license=context.license,
            access_tier=context.access_tier,
        ),
        extra={
            "dataset_id": context.dataset_id,
            "feature_lineage": {
                "mask_file": feature.lineage.mask_file,
                "mask_sha256": feature.lineage.mask_sha256,
                "mask_semantics": feature.lineage.mask_semantics,
                "extractor": feature.lineage.extractor,
                "extractor_version": feature.lineage.extractor_version,
            },
            "point_in_time": {
                "acquisition_time": acquisition.isoformat(),
                "processing_time": processing.isoformat(),
                "available_at": processing.isoformat(),
            },
        },
    )
