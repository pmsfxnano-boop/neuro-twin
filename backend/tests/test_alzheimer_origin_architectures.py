"""Tests for competing Alzheimer origin architecture hypotheses."""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from neuro_twin.network import (
    AlzheimerMechanismCatalog,
    AlzheimerOriginArchitectureRegistry,
)


BASE = Path(__file__).resolve().parents[1] / "neuro_twin" / "network"


def _json(name):
    return json.loads((BASE / name).read_text(encoding="utf-8"))


def test_origin_architecture_registry_validates_against_mechanism_map():
    catalog = AlzheimerMechanismCatalog.model_validate(
        _json("alzheimer_mechanism_map_v1.json")
    )
    registry = AlzheimerOriginArchitectureRegistry.model_validate(
        _json("alzheimer_origin_architectures_v1.json")
    )
    registry.validate_against(catalog)
    assert len(registry.architectures) == 7
    assert len({item.architecture_id for item in registry.architectures}) == 7


def test_origin_hypotheses_are_unranked_and_have_discriminating_predictions():
    registry = AlzheimerOriginArchitectureRegistry.model_validate(
        _json("alzheimer_origin_architectures_v1.json")
    )
    assert registry.registry_status == "competing_falsifiable_hypotheses_not_ranked"
    assert all(len(item.discriminating_prediction) >= 60 for item in registry.architectures)
    assert all(item.falsification_criteria for item in registry.architectures)


def test_registry_rejects_unknown_mechanism_edge():
    catalog = AlzheimerMechanismCatalog.model_validate(
        _json("alzheimer_mechanism_map_v1.json")
    )
    payload = _json("alzheimer_origin_architectures_v1.json")
    payload["architectures"][0]["candidate_edge_ids"].append("unknown_candidate_edge")
    registry = AlzheimerOriginArchitectureRegistry.model_validate(payload)
    with pytest.raises(ValueError, match="unknown edges"):
        registry.validate_against(catalog)


def test_registry_rejects_duplicate_architecture_ids():
    payload = _json("alzheimer_origin_architectures_v1.json")
    payload["architectures"][1]["architecture_id"] = payload["architectures"][0]["architecture_id"]
    with pytest.raises(ValidationError, match="unique"):
        AlzheimerOriginArchitectureRegistry.model_validate(payload)
