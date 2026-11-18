"""Tests for the evidence-graded Alzheimer mechanism catalogue."""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from neuro_twin.network import AlzheimerMechanismCatalog, MechanismEdge


def _payload():
    path = (
        Path(__file__).resolve().parents[1]
        / "neuro_twin"
        / "network"
        / "alzheimer_mechanism_map_v1.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_alzheimer_catalog_validates_and_has_multiscale_coverage():
    catalog = AlzheimerMechanismCatalog.model_validate(_payload())
    assert catalog.disease == "alzheimer"
    assert len(catalog.nodes) == 28
    assert len(catalog.edges) == 30
    scales = {node.biological_scale for node in catalog.nodes}
    assert {"genetic", "molecular", "organelle", "cellular", "circuit", "tissue", "systemic", "clinical"} <= scales


def test_mechanism_map_contains_separate_damage_and_reserve_constructs():
    catalog = AlzheimerMechanismCatalog.model_validate(_payload())
    node_ids = {node.node_id for node in catalog.nodes}
    assert "persistent_accumulated_damage" in node_ids
    assert "cognitive_reserve_compensation" in node_ids
    assert "clinical_cognitive_decline" in node_ids


def test_mechanism_edges_reference_known_nodes_and_preserve_causal_uncertainty():
    catalog = AlzheimerMechanismCatalog.model_validate(_payload())
    assert all(edge.source_node != edge.target_node for edge in catalog.edges)
    assert any(edge.causal_status == "hypothesis_only" for edge in catalog.edges)
    assert any(edge.causal_status == "genetically_supported" for edge in catalog.edges)
    assert catalog.node("familial_app_psen_variants").evidence_grade == "established"


def test_genetically_supported_edge_requires_human_genetic_evidence():
    with pytest.raises(ValidationError, match="human_genetics"):
        MechanismEdge(
            edge_id="bad_edge",
            source_node="source_a",
            target_node="target_b",
            relationship="candidate relationship",
            directionality="directed_candidate",
            causal_status="genetically_supported",
            evidence_grade="strong",
            evidence_types=("review_synthesis",),
            context="Test-only invalid edge used to verify the causal evidence guard.",
            limitations=("The test edge is deliberately invalid.",),
            source_refs=("10.0000/test",),
        )
