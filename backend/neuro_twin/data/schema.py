"""Canonical, source-independent data contracts.

The scientific engine consumes only these structures. Source adapters must
normalize external representations into this contract before inference.
"""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AccessTier(StrEnum):
    PUBLIC_API = "public_api"
    PUBLIC_DATASET = "public_dataset"
    CONTROLLED_DATASET = "controlled_dataset"


class ObservationKind(StrEnum):
    BIOMARKER = "biomarker"
    COGNITION = "cognition"
    MRI_FEATURE = "mri_feature"
    PET_FEATURE = "pet_feature"
    CLINICAL = "clinical"


class Provenance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_name: str = Field(min_length=1)
    source_version: str | None = None
    source_record_id: str | None = None
    retrieval_uri: str | None = None
    raw_hash: str | None = None
    processing_pipeline: str | None = None
    processing_version: str | None = None
    license: str | None = None
    access_tier: AccessTier


class QualityFlags(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_valid: bool = True
    temporal_integrity: bool = True
    unit_consistent: bool = True
    assay_consistent: bool = True
    center_effect_flag: bool = False
    scanner_effect_flag: bool = False
    outlier_flag: bool = False
    missingness_flag: bool = False
    provenance_complete: bool = True


class Observation(BaseModel):
    """One normalized measurement/event available to the model."""

    model_config = ConfigDict(extra="forbid")

    subject_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    acquisition_time: datetime
    result_time: datetime | None = None
    ingest_time: datetime
    source: str = Field(min_length=1)
    center: str | None = None
    modality: str = Field(min_length=1)
    assay: str | None = None
    observation_kind: ObservationKind
    feature: str = Field(min_length=1)
    value: float | list[float]
    unit: str | None = None
    uncertainty: float | None = Field(default=None, ge=0.0)
    missing: bool = False
    quality: QualityFlags = Field(default_factory=QualityFlags)
    provenance: Provenance
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("result_time")
    @classmethod
    def result_not_before_acquisition(cls, value: datetime | None, info):
        if value is None:
            return value
        acquisition = info.data.get("acquisition_time")
        if acquisition is not None and value < acquisition:
            raise ValueError("result_time cannot precede acquisition_time")
        return value

    @field_validator("ingest_time")
    @classmethod
    def ingest_not_before_available(cls, value: datetime, info):
        acquisition = info.data.get("acquisition_time")
        result = info.data.get("result_time")
        available = result or acquisition
        if available is not None and value < available:
            raise ValueError("ingest_time cannot precede the observation's availability time")
        return value

    @field_validator("value")
    @classmethod
    def finite_value(cls, value):
        import math

        values = value if isinstance(value, list) else [value]
        if any(not math.isfinite(float(v)) for v in values):
            raise ValueError("observation values must be finite")
        return value


class ObservationBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observations: list[Observation]
    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)

    @field_validator("observations")
    @classmethod
    def no_duplicate_events(cls, value: list[Observation]):
        keys = [(x.subject_id, x.event_id, x.feature) for x in value]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate subject/event/feature observations")
        return value
