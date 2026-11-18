"""Tests for the six-domain hierarchical construct inventory."""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from neuro_twin.network import NeuroDomainOntology


def _payload():
    path = (
        Path(__file__).resolve().parents[1]
        / "neuro_twin"
        / "network"
        / "domain_ontology_v1.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_ontology_defines_all_six_domains_in_canonical_order():
    ontology = NeuroDomainOntology.model_validate(_payload())
    assert ontology.state_vector_order == ("P", "I", "M", "N", "Q", "Pr")
    assert tuple(entry.domain for entry in ontology.entries) == ontology.state_vector_order


def test_metabolism_and_proteostasis_are_hierarchical_not_single_biomarkers():
    ontology = NeuroDomainOntology.model_validate(_payload())
    metabolism = ontology.get("M")
    proteostasis = ontology.get("Pr")
    assert {
        "mitochondrial_bioenergetics",
        "glucose_lactate",
        "lipid_metabolism",
        "redox_homeostasis",
    }.issubset({item.code for item in metabolism.subconstructs})
    assert {
        "folding_chaperones",
        "ubiquitin_proteasome",
        "autophagy_lysosome",
        "aggregation_trafficking",
    }.issubset({item.code for item in proteostasis.subconstructs})


def test_pathology_biomarkers_are_not_equated_with_total_proteostasis():
    ontology = NeuroDomainOntology.model_validate(_payload())
    proteostasis = ontology.get("Pr")
    guarded_terms = " ".join(proteostasis.not_equivalent_to).lower()
    assert "tau concentration alone" in guarded_terms
    assert "alpha-synuclein seeding assay alone" in guarded_terms


def test_progression_is_a_state_and_velocity_is_its_derivative():
    ontology = NeuroDomainOntology.model_validate(_payload())
    definition = ontology.get("P").operational_definition.lower()
    assert "p is a state" in definition
    assert "dp/dt" in definition
    assert "not its derivative" in definition


def test_domain_aggregation_is_not_silently_defined():
    ontology = NeuroDomainOntology.model_validate(_payload())
    assert all(
        entry.aggregation_status == "not_operationalized"
        for entry in ontology.entries
    )


def test_ontology_rejects_a_missing_or_duplicate_domain():
    payload = _payload()
    payload["entries"] = payload["entries"][:-1]
    with pytest.raises(ValidationError, match="exactly P, I, M, N, Q and Pr"):
        NeuroDomainOntology.model_validate(payload)


def test_ontology_rejects_automatic_domain_averaging_status():
    payload = _payload()
    payload["entries"][2]["aggregation_status"] = "equal_weight_average"
    with pytest.raises(ValidationError):
        NeuroDomainOntology.model_validate(payload)
