"""Regression tests for the disease-specific literature evidence register."""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from neuro_twin.network import (
    MechanisticEvidenceRegistry,
    MechanisticInteractionRecord,
)


def _registry_payload():
    manifest = (
        Path(__file__).resolve().parents[1]
        / "neuro_twin"
        / "network"
        / "edge_evidence_v1.json"
    )
    return json.loads(manifest.read_text(encoding="utf-8"))


def test_literature_registry_loads_and_is_hashable():
    registry = MechanisticEvidenceRegistry.model_validate(_registry_payload())
    assert len(registry.records) == 7
    assert len(registry.registry_hash) == 64
    assert registry.registry_hash == MechanisticEvidenceRegistry.model_validate(
        _registry_payload()
    ).registry_hash


def test_alzheimer_unresolved_interaction_does_not_create_directed_edge():
    registry = MechanisticEvidenceRegistry.model_validate(_registry_payload())
    records = registry.records_for_disease("alzheimer")
    assert len(records) == 2
    # M -> Pr is only a directional hypothesis; I-Pr remains unresolved.
    assert registry.directional_candidates_for_disease("alzheimer") == (("M", "Pr"),)
    inflammation_record = next(
        record for record in records if record.interaction_id == "ad_inflammation_proteostasis"
    )
    assert inflammation_record.direction_class == "undirected_interaction"
    assert inflammation_record.candidate_directions == ()


def test_parkinson_mitochondria_proteostasis_feedback_is_explicitly_reciprocal():
    registry = MechanisticEvidenceRegistry.model_validate(_registry_payload())
    edges = registry.directional_candidates_for_disease("parkinson")
    assert ("M", "Pr") in edges
    assert ("Pr", "M") in edges


def test_literature_records_are_never_eligible_for_parameter_fitting():
    registry = MechanisticEvidenceRegistry.model_validate(_registry_payload())
    registry.assert_not_parameterized()
    assert all(record.model_eligible is False for record in registry.records)
    assert all(record.sign_constraint == "none" for record in registry.records)
    assert all(
        record.record_status == "literature_candidate_only"
        for record in registry.records
    )


def test_registry_rejects_unresolved_record_with_a_direction():
    payload = _registry_payload()["records"][1].copy()
    payload["candidate_directions"] = [["I", "Pr"]]
    with pytest.raises(ValidationError, match="undirected interactions"):
        MechanisticInteractionRecord.model_validate(payload)


def test_registry_rejects_a_reciprocal_record_with_only_one_direction():
    payload = _registry_payload()["records"][2].copy()
    payload["candidate_directions"] = [["M", "Pr"]]
    with pytest.raises(ValidationError, match="both directions"):
        MechanisticInteractionRecord.model_validate(payload)


def test_registry_rejects_missing_bibliographic_identifier():
    payload = _registry_payload()["records"][0].copy()
    payload["references"] = [{
        "title": "A review without an identifier",
        "publication_year": 2025,
        "url": "https://example.org/review",
    }]
    with pytest.raises(ValidationError, match="DOI or PMID"):
        MechanisticInteractionRecord.model_validate(payload)
