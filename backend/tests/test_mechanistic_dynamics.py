"""Tests for the nonlinear mechanism simulator and virtual interventions."""
import json
from pathlib import Path

import numpy as np
import pytest

from neuro_twin.network import (
    AlzheimerMechanismCatalog,
    AlzheimerOriginArchitectureRegistry,
    DamageAccumulationModel,
    EdgeKinetics,
    InterventionProtocol,
    MechanisticArchitectureModel,
    NodeKinetics,
    compare_intervention,
    summarize_parameter_ensemble,
)


BASE = Path(__file__).resolve().parents[1] / "neuro_twin" / "network"


def _json(name):
    return json.loads((BASE / name).read_text(encoding="utf-8"))


def _catalog_and_registry():
    catalog = AlzheimerMechanismCatalog.model_validate(
        _json("alzheimer_mechanism_map_v1.json")
    )
    registry = AlzheimerOriginArchitectureRegistry.model_validate(
        _json("alzheimer_origin_architectures_v1.json")
    )
    registry.validate_against(catalog)
    return catalog, registry


def _model(coefficient=0.9):
    catalog, registry = _catalog_and_registry()
    damage = DamageAccumulationModel(
        disease="alzheimer",
        time_unit="year",
        driver_names=("synaptic_circuit_dysfunction",),
        repair_names=(),
        damage_weights=np.ones((4, 1)),
        repair_weights=np.zeros((4, 0)),
        rate_scale=np.ones(4),
    )
    model = MechanisticArchitectureModel(
        architecture_id="tau_neuronal_vulnerability_dominant",
        time_unit="year",
        nodes=(
            NodeKinetics("tau_misprocessing", 0.9, 0.05),
            NodeKinetics("tau_seeded_spread", 0.05, 0.08),
            NodeKinetics("synaptic_circuit_dysfunction", 0.05, 0.08),
        ),
        edges=(
            EdgeKinetics("tau_to_spread", coefficient, "activation", 0.5, 1.0),
            EdgeKinetics("tau_to_synapse", coefficient, "activation", 0.5, 1.0),
        ),
        damage_model=damage,
        parameter_status="hypothesis",
    )
    model.validate_against(catalog, registry)
    return catalog, model


def test_mechanistic_model_is_bound_to_evidence_catalog_and_architecture():
    catalog, registry = _catalog_and_registry()
    _, model = _model()
    assert model.validate_against(catalog, registry) is model
    assert len(model.config_hash) == 64


def test_intervention_changes_mechanism_and_damage_trajectory_in_toy_fixture():
    catalog, model = _model()
    time = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    initial = np.asarray([0.8, 0.1, 0.1])
    initial_damage = np.zeros(4)
    intervention = InterventionProtocol(
        intervention_id="virtual_tau_suppression",
        target_node="tau_misprocessing",
        mode="clamp",
        start_time=1.0,
        end_time=4.0,
        intensity=0.0,
    )
    result = compare_intervention(
        model,
        catalog=catalog,
        time=time,
        initial_activity=initial,
        initial_damage=initial_damage,
        interventions=(intervention,),
    )

    assert result.baseline.damage.cumulative_damage.shape == (5, 4)
    assert result.intervention.damage.cumulative_damage.shape == (5, 4)
    assert np.all(result.effect.damage_prevented_at_final > 0.0)
    assert np.all(result.effect.damage_burden_auc_prevented > 0.0)
    assert (
        result.effect.interpretation
        == "model_conditional_counterfactual_not_empirical_causal_effect"
    )
    assert result.intervention.interpretation == "hypothesis_scenario_only"


def test_additive_drive_can_reduce_downstream_damage_without_changing_initial_state():
    catalog, model = _model()
    result = compare_intervention(
        model,
        catalog=catalog,
        time=np.asarray([0.0, 1.0, 2.0, 3.0]),
        initial_activity=np.asarray([0.8, 0.1, 0.1]),
        initial_damage=np.zeros(4),
        interventions=(
            InterventionProtocol(
                intervention_id="virtual_synaptic_stress_reduction",
                target_node="synaptic_circuit_dysfunction",
                mode="additive_drive",
                start_time=1.0,
                end_time=3.0,
                intensity=-0.5,
            ),
        ),
    )
    assert np.all(result.effect.damage_prevented_at_final > 0.0)


def test_simulator_rejects_edges_with_unresolved_or_bidirectional_direction():
    catalog, registry = _catalog_and_registry()
    damage = DamageAccumulationModel(
        disease="alzheimer",
        time_unit="year",
        driver_names=("mitochondrial_bioenergetic_failure",),
        repair_names=(),
        damage_weights=np.ones((4, 1)),
        repair_weights=np.zeros((4, 0)),
        rate_scale=np.ones(4),
    )
    model = MechanisticArchitectureModel(
        architecture_id="metabolic_mitochondrial_dominant",
        time_unit="year",
        nodes=(
            NodeKinetics("glucose_lipid_dysregulation", 0.5, 0.1),
            NodeKinetics("mitochondrial_bioenergetic_failure", 0.5, 0.1),
        ),
        edges=(
            EdgeKinetics("glucose_lipid_to_mito", 0.2, "activation", 0.5, 1.0),
        ),
        damage_model=damage,
    )
    with pytest.raises(ValueError, match="not a directed candidate"):
        model.validate_against(catalog, registry)


def test_intervention_must_target_a_modeled_mechanism_node():
    catalog, model = _model()
    with pytest.raises(ValueError, match="not modeled"):
        model.simulate(
            catalog=catalog,
            time=np.asarray([0.0, 1.0]),
            initial_activity=np.asarray([0.8, 0.1, 0.1]),
            interventions=(
                InterventionProtocol(
                    intervention_id="invalid",
                    target_node="missing_node",
                    mode="additive_drive",
                    start_time=0.0,
                    end_time=1.0,
                    intensity=-0.1,
                ),
            ),
        )


def test_parameter_ensemble_reports_effect_interval():
    catalog, model_a = _model(0.7)
    _, model_b = _model(1.0)
    kwargs = {
        "catalog": catalog,
        "time": np.asarray([0.0, 1.0, 2.0, 3.0]),
        "initial_activity": np.asarray([0.8, 0.1, 0.1]),
        "initial_damage": np.zeros(4),
        "interventions": (
            InterventionProtocol(
                intervention_id="virtual_tau_suppression",
                target_node="tau_misprocessing",
                mode="clamp",
                start_time=1.0,
                end_time=3.0,
                intensity=0.0,
            ),
        ),
    }
    comparisons = (
        compare_intervention(model_a, **kwargs),
        compare_intervention(model_b, **kwargs),
    )
    summary = summarize_parameter_ensemble(comparisons)
    assert summary["damage_prevented_median"].shape == (4,)
    assert summary["damage_prevented_interval"].shape == (2, 4)
    assert np.all(summary["damage_prevented_interval"][0] <= summary["damage_prevented_interval"][1])
