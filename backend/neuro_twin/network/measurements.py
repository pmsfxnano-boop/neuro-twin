"""Measurement-layer contracts for the six-domain Neuro-Twin network.

This layer describes what a measurement represents and whether its metadata
are adequate for research use. It never creates a latent-state observation
operator or turns biomarker associations into direct state measurements.
"""
from __future__ import annotations

import hashlib
import json
from datetime import timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from neuro_twin.data.schema import Observation, ObservationKind
from neuro_twin.network.evidence import Domain, EvidenceCitation
from neuro_twin.network.model import DOMAIN_NAMES
from neuro_twin.network.ontology import NeuroDomainOntology


Disease = Literal["alzheimer", "parkinson"]
MeasurementValueType = Literal[
    "continuous",
    "ratio",
    "binary",
    "ordinal",
    "count",
]
MeasurementRelation = Literal[
    "candidate_proxy",
    "phenotype_indicator",
    "image_derived_feature",
]
UncertaintyPolicy = Literal["required", "recommended", "not_applicable"]
UncertaintyKind = Literal[
    "standard_deviation",
    "standard_error",
    "coefficient_of_variation",
    "confidence_interval_half_width",
]
UncertaintySource = Literal[
    "assay_validation",
    "source_reported",
    "quality_control_estimate",
    "versioned_derived_estimate",
]


class MeasurementReference(BaseModel):
    """Bibliographic or official source document supporting a measurement record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    organization: str = Field(min_length=2)
    title: str = Field(min_length=8)
    year: int | None = Field(default=None, ge=1900, le=2100)
    doi: str | None = None
    pmid: str | None = None
    url: str = Field(min_length=8)

    @model_validator(mode="after")
    def validate_reference(self) -> "MeasurementReference":
        if not self.url.startswith("https://"):
            raise ValueError("reference URL must use HTTPS")
        if not self.doi and not self.pmid and not self.organization:
            raise ValueError("reference requires a DOI, PMID, or source organization")
        return self


class MeasurementUncertainty(BaseModel):
    """Typed uncertainty metadata; the numeric field alone is not sufficient."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: float = Field(gt=0.0)
    kind: UncertaintyKind
    source: UncertaintySource
    method_version: str = Field(min_length=1)
    unit: str | None = None


class MeasurementDefinition(BaseModel):
    """Catalogue entry describing a source measurement and its construct context.

    A measurement may be associated with a subconstruct but is not thereby a
    direct observation of the six-dimensional latent state. Latent bindings
    remain deliberately unregistered in this initial catalogue.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    measurement_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{3,100}$")
    disease: Disease
    canonical_feature: str = Field(min_length=2)
    source_feature_aliases: tuple[str, ...] = ()
    modality: str = Field(min_length=2)
    observation_kind: ObservationKind
    specimen: str | None = None
    analyte_or_metric: str = Field(min_length=2)
    value_type: MeasurementValueType
    canonical_unit: str = Field(min_length=1)
    accepted_units: tuple[str, ...] = Field(min_length=1)
    value_direction: str = Field(min_length=8)
    target_domain: Domain
    target_subconstruct: str = Field(pattern=r"^[a-z][a-z0-9_]{2,64}$")
    construct_relation: MeasurementRelation
    uncertainty_policy: UncertaintyPolicy
    accepted_uncertainty_kinds: tuple[UncertaintyKind, ...] = ()
    assay_required: bool = True
    required_extra_keys: tuple[str, ...] = ()
    required_provenance_source_version: bool = True
    required_provenance_raw_hash: bool = True
    limitations: tuple[str, ...] = Field(min_length=1)
    references: tuple[MeasurementReference, ...] = Field(min_length=1)
    latent_binding_status: Literal["not_registered"] = "not_registered"

    @model_validator(mode="after")
    def validate_measurement_definition(self) -> "MeasurementDefinition":
        if self.target_domain not in DOMAIN_NAMES:
            raise ValueError("target_domain must belong to the six-domain state vector")
        if self.canonical_unit not in self.accepted_units:
            raise ValueError("canonical_unit must be included in accepted_units")
        aliases = (self.canonical_feature,) + self.source_feature_aliases
        if len(set(aliases)) != len(aliases):
            raise ValueError("canonical_feature and source aliases must be unique")
        if self.uncertainty_policy == "not_applicable" and self.accepted_uncertainty_kinds:
            raise ValueError("not_applicable uncertainty cannot declare uncertainty kinds")
        if self.uncertainty_policy != "not_applicable" and not self.accepted_uncertainty_kinds:
            raise ValueError("uncertainty policy requires explicitly accepted uncertainty kinds")
        if self.value_type == "binary" and self.canonical_unit not in ("binary", "0/1"):
            raise ValueError("binary measurements must use a binary or 0/1 canonical unit")
        return self


class MeasurementEnvelope(BaseModel):
    """A canonical Observation plus measurement-specific uncertainty semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    measurement_id: str = Field(min_length=1)
    catalog_version: str = Field(min_length=1)
    observation: Observation
    uncertainty: MeasurementUncertainty | None = None
    transformation_id: str | None = None
    transformation_version: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def uncertainty_matches_observation(self) -> "MeasurementEnvelope":
        if self.uncertainty is not None:
            if self.observation.uncertainty is None:
                raise ValueError(
                    "typed uncertainty requires Observation.uncertainty to contain the same numeric value"
                )
            if abs(float(self.observation.uncertainty) - self.uncertainty.value) > 1e-12:
                raise ValueError("typed uncertainty does not match Observation.uncertainty")
        if (self.transformation_id is None) != (self.transformation_version is None):
            raise ValueError("transformation_id and transformation_version must be supplied together")
        return self


class MeasurementValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    measurement_id: str
    disease: Disease
    status: Literal["VALID_FOR_MEASUREMENT_LAYER", "BLOCKED", "MISSING"]
    latent_inference_eligible: Literal[False] = False
    blocking_reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    catalog_hash: str
    ontology_subconstruct: str


class MeasurementCatalog(BaseModel):
    """Versioned source-measurement inventory, separate from latent operators."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(min_length=1)
    catalog_status: Literal["measurement_inventory_only"] = "measurement_inventory_only"
    definitions: tuple[MeasurementDefinition, ...]

    @model_validator(mode="after")
    def unique_features_and_ids(self) -> "MeasurementCatalog":
        ids = [item.measurement_id for item in self.definitions]
        if len(ids) != len(set(ids)):
            raise ValueError("measurement_id values must be unique")
        features = [(item.disease, item.canonical_feature) for item in self.definitions]
        if len(features) != len(set(features)):
            raise ValueError("canonical feature must be unique within each disease")
        return self

    @property
    def catalog_hash(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get(self, measurement_id: str) -> MeasurementDefinition:
        for definition in self.definitions:
            if definition.measurement_id == measurement_id:
                return definition
        raise KeyError(f"unknown measurement_id: {measurement_id}")

    def definitions_for_disease(self, disease: Disease) -> tuple[MeasurementDefinition, ...]:
        return tuple(item for item in self.definitions if item.disease == disease)

    def validate_ontology(self, ontology: NeuroDomainOntology) -> None:
        """Verify every target subconstruct exists in the canonical ontology."""
        for definition in self.definitions:
            ontology_entry = ontology.get(definition.target_domain)
            valid_codes = {item.code for item in ontology_entry.subconstructs}
            if definition.target_subconstruct not in valid_codes:
                raise ValueError(
                    f"{definition.measurement_id} targets unknown subconstruct "
                    f"{definition.target_domain}.{definition.target_subconstruct}"
                )

    def validate_envelope(
        self,
        envelope: MeasurementEnvelope,
        *,
        disease: Disease,
    ) -> MeasurementValidationReport:
        """Validate data quality and metadata without creating a latent binding."""
        definition = self.get(envelope.measurement_id)
        observation = envelope.observation
        blocking: list[str] = []
        warnings: list[str] = []

        if envelope.catalog_version != self.schema_version:
            blocking.append("catalog version does not match the active measurement catalogue")
        if definition.disease != disease:
            blocking.append("measurement definition is not registered for the requested disease")
        if observation.feature != definition.canonical_feature:
            if observation.feature in definition.source_feature_aliases:
                blocking.append("source feature alias requires an explicit versioned harmonization step")
            else:
                blocking.append("observation feature does not match the canonical measurement feature")
        if observation.modality != definition.modality:
            blocking.append("observation modality does not match the measurement definition")
        if observation.observation_kind != definition.observation_kind:
            blocking.append("observation kind does not match the measurement definition")
        if observation.unit not in definition.accepted_units:
            blocking.append("observation unit is not accepted; harmonize explicitly before validation")
        if definition.assay_required and not observation.assay:
            blocking.append("assay identity is required")
        if not observation.quality.schema_valid:
            blocking.append("observation failed schema quality control")
        if not observation.quality.temporal_integrity:
            blocking.append("observation failed temporal-integrity quality control")
        if not observation.quality.unit_consistent:
            blocking.append("observation failed unit-consistency quality control")
        if not observation.quality.assay_consistent:
            blocking.append("observation failed assay-consistency quality control")
        if not observation.quality.provenance_complete:
            blocking.append("observation provenance is incomplete")
        if definition.required_provenance_source_version and not observation.provenance.source_version:
            blocking.append("source version is required")
        if definition.required_provenance_raw_hash and not observation.provenance.raw_hash:
            blocking.append("raw source hash is required")
        if observation.acquisition_time.tzinfo is None or observation.acquisition_time.utcoffset() is None:
            blocking.append("acquisition_time must be timezone-aware")
        if observation.ingest_time.tzinfo is None or observation.ingest_time.utcoffset() is None:
            blocking.append("ingest_time must be timezone-aware")
        missing_metadata = [
            key for key in definition.required_extra_keys
            if key not in observation.extra or observation.extra[key] in (None, "")
        ]
        if missing_metadata:
            blocking.append("missing required measurement metadata: " + ", ".join(missing_metadata))

        value = observation.value
        if isinstance(value, list):
            blocking.append("this measurement definition expects a scalar observation")
        elif definition.value_type == "binary" and float(value) not in (0.0, 1.0):
            blocking.append("binary measurement values must be exactly 0 or 1")

        if definition.uncertainty_policy == "required" and envelope.uncertainty is None:
            blocking.append("typed measurement uncertainty is required before quantitative inference")
        elif definition.uncertainty_policy == "recommended" and envelope.uncertainty is None:
            warnings.append("measurement uncertainty is not supplied")
        if envelope.uncertainty is not None:
            if envelope.uncertainty.kind not in definition.accepted_uncertainty_kinds:
                blocking.append("uncertainty kind is not accepted for this measurement definition")
            if envelope.uncertainty.kind in ("standard_deviation", "standard_error", "confidence_interval_half_width"):
                if envelope.uncertainty.unit != definition.canonical_unit:
                    blocking.append("uncertainty unit must match the canonical measurement unit")
            if envelope.uncertainty.kind == "coefficient_of_variation" and not (0.0 < envelope.uncertainty.value < 10.0):
                blocking.append("coefficient of variation must be a finite relative value in (0, 10)")

        if observation.quality.center_effect_flag:
            warnings.append("center effect is flagged and requires downstream harmonization/sensitivity analysis")
        if observation.quality.scanner_effect_flag:
            warnings.append("scanner effect is flagged and requires downstream harmonization/sensitivity analysis")
        if observation.quality.outlier_flag:
            warnings.append("observation is flagged as an outlier")
        if observation.quality.missingness_flag or observation.missing:
            warnings.append("observation is marked missing")
        if observation.uncertainty is not None and envelope.uncertainty is None:
            warnings.append("numeric uncertainty lacks a declared statistical interpretation")

        if observation.missing:
            status = "MISSING"
        elif blocking:
            status = "BLOCKED"
        else:
            status = "VALID_FOR_MEASUREMENT_LAYER"

        return MeasurementValidationReport(
            measurement_id=definition.measurement_id,
            disease=disease,
            status=status,
            latent_inference_eligible=False,
            blocking_reasons=tuple(blocking),
            warnings=tuple(warnings),
            catalog_hash=self.catalog_hash,
            ontology_subconstruct=f"{definition.target_domain}.{definition.target_subconstruct}",
        )


def normalize_timestamp_to_utc(value):
    """Small explicit helper for adapter pipelines; does not alter source data."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)
