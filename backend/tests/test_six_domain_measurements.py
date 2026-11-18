"""Tests for typed measurement metadata and the non-inference boundary."""
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from neuro_twin.data.schema import (
    AccessTier,
    Observation,
    ObservationKind,
    Provenance,
)
from neuro_twin.network import (
    MeasurementCatalog,
    MeasurementEnvelope,
    MeasurementUncertainty,
    NeuroDomainOntology,
)


BACKEND = Path(__file__).resolve().parents[1]
CATALOG_PATH = BACKEND / "neuro_twin" / "network" / "measurement_catalog_v1.json"
ONTOLOGY_PATH = BACKEND / "neuro_twin" / "network" / "domain_ontology_v1.json"


def _load():
    catalog = MeasurementCatalog.model_validate(
        json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    )
    ontology = NeuroDomainOntology.model_validate(
        json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8"))
    )
    catalog.validate_ontology(ontology)
    return catalog


def _observation(
    *,
    feature="plasma_pTau217",
    modality="blood_biomarker",
    kind=ObservationKind.BIOMARKER,
    unit="pg/mL",
    value=1.25,
    uncertainty=0.12,
    assay="validated_platform_assay",
    extra=None,
    source_version="source-v2",
    raw_hash="a" * 64,
    quality=None,
):
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    from neuro_twin.data.schema import QualityFlags

    return Observation(
        subject_id="SUBJECT-001",
        event_id="VISIT-001",
        acquisition_time=now,
        result_time=now,
        ingest_time=now,
        source="research-cohort",
        modality=modality,
        assay=assay,
        observation_kind=kind,
        feature=feature,
        value=value,
        unit=unit,
        uncertainty=uncertainty,
        quality=quality or QualityFlags(provenance_complete=True),
        provenance=Provenance(
            source_name="research-cohort",
            source_version=source_version,
            source_record_id="RECORD-001",
            retrieval_uri="https://example.org/record/001",
            raw_hash=raw_hash,
            processing_pipeline="measurement-test",
            processing_version="1.0.0",
            access_tier=AccessTier.PUBLIC_DATASET,
        ),
        extra={"assay_platform": "example-platform", **(extra or {})},
    )


def _typed_uncertainty(value=0.12, kind="standard_deviation", unit="pg/mL"):
    return MeasurementUncertainty(
        value=value,
        kind=kind,
        source="assay_validation",
        method_version="uncertainty-method-1.0",
        unit=unit,
    )


def test_catalog_and_ontology_are_valid_and_hash_stable():
    catalog = _load()
    second = _load()
    assert len(catalog.definitions) == 11
    assert catalog.catalog_hash == second.catalog_hash
    assert len(catalog.catalog_hash) == 64
    assert catalog.catalog_status == "measurement_inventory_only"


def test_catalog_contains_measurements_for_both_disease_scopes():
    catalog = _load()
    assert len(catalog.definitions_for_disease("alzheimer")) == 8
    assert len(catalog.definitions_for_disease("parkinson")) == 3


def test_measurement_validation_does_not_grant_latent_inference():
    catalog = _load()
    envelope = MeasurementEnvelope(
        measurement_id="ad_plasma_ptau217",
        catalog_version=catalog.schema_version,
        observation=_observation(),
        uncertainty=_typed_uncertainty(),
    )
    report = catalog.validate_envelope(envelope, disease="alzheimer")
    assert report.status == "VALID_FOR_MEASUREMENT_LAYER"
    assert report.latent_inference_eligible is False
    assert report.ontology_subconstruct == "P.pathology_burden"


def test_numeric_uncertainty_requires_explicit_statistical_semantics():
    catalog = _load()
    observation = _observation(uncertainty=0.12)
    envelope = MeasurementEnvelope(
        measurement_id="ad_plasma_ptau217",
        catalog_version=catalog.schema_version,
        observation=observation,
    )
    report = catalog.validate_envelope(envelope, disease="alzheimer")
    assert report.status == "BLOCKED"
    assert any("typed measurement uncertainty" in reason for reason in report.blocking_reasons)
    assert any("statistical interpretation" in warning for warning in report.warnings)


def test_typed_uncertainty_must_match_canonical_observation_value():
    with pytest.raises(ValidationError, match="does not match"):
        MeasurementEnvelope(
            measurement_id="ad_plasma_ptau217",
            catalog_version="1.0.0",
            observation=_observation(uncertainty=0.1),
            uncertainty=_typed_uncertainty(value=0.12),
        )


def test_source_unit_must_be_harmonized_to_canonical_unit():
    catalog = _load()
    envelope = MeasurementEnvelope(
        measurement_id="ad_plasma_ptau217",
        catalog_version=catalog.schema_version,
        observation=_observation(unit="ng/L", uncertainty=0.12),
        uncertainty=_typed_uncertainty(value=0.12, unit="pg/mL"),
    )
    report = catalog.validate_envelope(envelope, disease="alzheimer")
    assert report.status == "BLOCKED"
    assert any("canonical unit" in reason for reason in report.blocking_reasons)


def test_source_feature_alias_is_not_silently_relabelled():
    catalog = _load()
    envelope = MeasurementEnvelope(
        measurement_id="ad_plasma_ptau217",
        catalog_version=catalog.schema_version,
        observation=_observation(feature="pTau217"),
        uncertainty=_typed_uncertainty(),
    )
    report = catalog.validate_envelope(envelope, disease="alzheimer")
    assert report.status == "BLOCKED"
    assert any("harmonization" in reason for reason in report.blocking_reasons)


def test_assay_platform_and_provenance_are_mandatory_for_ptau():
    catalog = _load()
    observation = _observation(extra={})
    # Remove one required metadata field after building the otherwise valid record.
    observation = observation.model_copy(update={"extra": {}})
    envelope = MeasurementEnvelope(
        measurement_id="ad_plasma_ptau217",
        catalog_version=catalog.schema_version,
        observation=observation,
        uncertainty=_typed_uncertainty(),
    )
    report = catalog.validate_envelope(envelope, disease="alzheimer")
    assert report.status == "BLOCKED"
    assert any("assay_platform" in reason for reason in report.blocking_reasons)

    bad_provenance = observation.provenance.model_copy(update={"raw_hash": None, "source_version": None})
    bad_observation = observation.model_copy(update={"provenance": bad_provenance})
    bad_envelope = MeasurementEnvelope(
        measurement_id="ad_plasma_ptau217",
        catalog_version=catalog.schema_version,
        observation=bad_observation,
        uncertainty=_typed_uncertainty(),
    )
    bad_report = catalog.validate_envelope(bad_envelope, disease="alzheimer")
    assert any("source version" in reason for reason in bad_report.blocking_reasons)
    assert any("raw source hash" in reason for reason in bad_report.blocking_reasons)


def test_scanner_or_center_flags_are_warnings_not_silent_adjustments():
    catalog = _load()
    from neuro_twin.data.schema import QualityFlags

    observation = _observation(
        feature="FDG_PET_regional_SUVR",
        modality="fdg_pet",
        kind=ObservationKind.PET_FEATURE,
        unit="SUVR",
        value=0.8,
        uncertainty=0.03,
        assay=None,
        extra={
            "tracer": "18F-FDG",
            "reference_region": "declared-reference",
            "processing_pipeline_version": "pet-pipeline-1",
        },
        quality=QualityFlags(
            provenance_complete=True,
            scanner_effect_flag=True,
            center_effect_flag=True,
        ),
    )
    envelope = MeasurementEnvelope(
        measurement_id="ad_fdg_pet_regional_suvr",
        catalog_version=catalog.schema_version,
        observation=observation,
        uncertainty=MeasurementUncertainty(
            value=0.03,
            kind="standard_deviation",
            source="quality_control_estimate",
            method_version="pet-qc-1",
            unit="SUVR",
        ),
    )
    report = catalog.validate_envelope(envelope, disease="alzheimer")
    assert report.status == "VALID_FOR_MEASUREMENT_LAYER"
    assert any("scanner effect" in warning for warning in report.warnings)
    assert any("center effect" in warning for warning in report.warnings)


def test_alpha_synuclein_saa_is_binary_and_not_total_proteostasis():
    catalog = _load()
    envelope = MeasurementEnvelope(
        measurement_id="pd_csf_alpha_syn_saa",
        catalog_version=catalog.schema_version,
        observation=_observation(
            feature="CSF_alphaSyn_SAA_result",
            modality="csf_biomarker",
            kind=ObservationKind.BIOMARKER,
            unit="binary",
            value=1.0,
            uncertainty=None,
            assay="CSF-alphaSyn-SAA",
            extra={"assay_protocol_version": "protocol-1"},
        ),
    )
    report = catalog.validate_envelope(envelope, disease="parkinson")
    assert report.status == "VALID_FOR_MEASUREMENT_LAYER"
    assert report.ontology_subconstruct == "Pr.aggregation_trafficking"
    assert report.latent_inference_eligible is False


def test_alpha_synuclein_saa_rejects_non_binary_result():
    catalog = _load()
    envelope = MeasurementEnvelope(
        measurement_id="pd_csf_alpha_syn_saa",
        catalog_version=catalog.schema_version,
        observation=_observation(
            feature="CSF_alphaSyn_SAA_result",
            modality="csf_biomarker",
            kind=ObservationKind.BIOMARKER,
            unit="binary",
            value=2.0,
            uncertainty=None,
            assay="CSF-alphaSyn-SAA",
            extra={"assay_protocol_version": "protocol-1"},
        ),
    )
    report = catalog.validate_envelope(envelope, disease="parkinson")
    assert report.status == "BLOCKED"
    assert any("exactly 0 or 1" in reason for reason in report.blocking_reasons)


def test_measurement_definition_cannot_be_assigned_an_implicit_latent_binding():
    catalog = _load()
    definition = catalog.get("ad_plasma_ptau217")
    payload = definition.model_dump(mode="json")
    payload["latent_binding_status"] = "registered"
    with pytest.raises(ValidationError):
        type(definition).model_validate(payload)


def test_wrong_disease_is_blocked():
    catalog = _load()
    envelope = MeasurementEnvelope(
        measurement_id="ad_plasma_ptau217",
        catalog_version=catalog.schema_version,
        observation=_observation(),
        uncertainty=_typed_uncertainty(),
    )
    report = catalog.validate_envelope(envelope, disease="parkinson")
    assert report.status == "BLOCKED"
    assert any("not registered for the requested disease" in x for x in report.blocking_reasons)


def test_outdated_catalog_version_is_blocked():
    catalog = _load()
    envelope = MeasurementEnvelope(
        measurement_id="ad_plasma_ptau217",
        catalog_version="0.9.0",
        observation=_observation(),
        uncertainty=_typed_uncertainty(),
    )
    report = catalog.validate_envelope(envelope, disease="alzheimer")
    assert report.status == "BLOCKED"
    assert any("catalog version" in x for x in report.blocking_reasons)


def test_naive_timestamps_are_blocked_even_if_canonical_schema_accepts_them():
    catalog = _load()
    observation = _observation()
    naive = datetime(2026, 1, 1, 12, 0)
    observation = observation.model_copy(update={"acquisition_time": naive})
    envelope = MeasurementEnvelope(
        measurement_id="ad_plasma_ptau217",
        catalog_version=catalog.schema_version,
        observation=observation,
        uncertainty=_typed_uncertainty(),
    )
    report = catalog.validate_envelope(envelope, disease="alzheimer")
    assert report.status == "BLOCKED"
    assert any("timezone-aware" in reason for reason in report.blocking_reasons)
