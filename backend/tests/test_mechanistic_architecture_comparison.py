"""Integration tests for end-to-end competing architecture comparison."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from neuro_twin.network import (
    AlzheimerMechanismCatalog,
    AlzheimerOriginArchitectureRegistry,
    ArchitectureCandidateRun,
    DamageAccumulationModel,
    EdgeKinetics,
    MechanismObservationModel,
    MechanisticArchitectureModel,
    NodeKinetics,
    compare_mechanistic_architectures,
)


BASE = Path(__file__).resolve().parents[1] / "neuro_twin" / "network"


def _catalog_registry():
    catalog = AlzheimerMechanismCatalog.model_validate(
        json.loads((BASE / "alzheimer_mechanism_map_v1.json").read_text(encoding="utf-8"))
    )
    registry = AlzheimerOriginArchitectureRegistry.model_validate(
        json.loads((BASE / "alzheimer_origin_architectures_v1.json").read_text(encoding="utf-8"))
    )
    return catalog, registry


def _model(architecture_id, nodes, edges, coefficient):
    damage = DamageAccumulationModel(
        disease="alzheimer",
        time_unit="year",
        driver_names=("synaptic_circuit_dysfunction",),
        repair_names=(),
        damage_weights=np.ones((4, 1)),
        repair_weights=np.zeros((4, 0)),
        rate_scale=np.ones(4),
    )
    return MechanisticArchitectureModel(
        architecture_id=architecture_id,
        time_unit="year",
        nodes=tuple(NodeKinetics(node, baseline, rate) for node, baseline, rate in nodes),
        edges=tuple(
            EdgeKinetics(edge_id, coefficient, "activation", 0.5, 1.0)
            for edge_id in edges
        ),
        damage_model=damage,
        parameter_status="hypothesis",
    )


def _operator(model, weights, *, features=("synaptic_signal",), units=("zscore",)):
    return MechanismObservationModel(
        model_id=f"operator-{model.architecture_id}",
        version="1.0",
        mechanism_node_ids=model.node_ids,
        feature_names=features,
        feature_units=units,
        H=np.asarray([weights], dtype=float),
        bias=np.zeros(len(features)),
        R=np.asarray([[0.1]]),
        source_catalog_hash="d" * 64,
        transformation_id="test-common-zscore",
        transformation_version="v1",
    )


def _candidates():
    tau_low = _model(
        "tau_neuronal_vulnerability_dominant",
        (
            ("tau_misprocessing", 0.9, 0.05),
            ("tau_seeded_spread", 0.05, 0.08),
            ("synaptic_circuit_dysfunction", 0.05, 0.08),
        ),
        ("tau_to_spread", "tau_to_synapse"),
        0.45,
    )
    tau_high = _model(
        "tau_neuronal_vulnerability_dominant",
        (
            ("tau_misprocessing", 0.9, 0.05),
            ("tau_seeded_spread", 0.05, 0.08),
            ("synaptic_circuit_dysfunction", 0.05, 0.08),
        ),
        ("tau_to_spread", "tau_to_synapse"),
        0.55,
    )
    immune_low = _model(
        "immune_glial_dominant",
        (
            ("microglial_state_shift", 0.9, 0.05),
            ("complement_synapse_elimination", 0.05, 0.08),
            ("synaptic_circuit_dysfunction", 0.05, 0.08),
        ),
        ("microglia_to_complement", "complement_to_synapse"),
        0.45,
    )
    immune_high = _model(
        "immune_glial_dominant",
        (
            ("microglial_state_shift", 0.9, 0.05),
            ("complement_synapse_elimination", 0.05, 0.08),
            ("synaptic_circuit_dysfunction", 0.05, 0.08),
        ),
        ("microglia_to_complement", "complement_to_synapse"),
        0.55,
    )
    return (
        ArchitectureCandidateRun(
            architecture_id=tau_low.architecture_id,
            parameter_draws=(tau_low, tau_high),
            initial_activity_draws=(
                np.asarray([0.8, 0.1, 0.1]),
                np.asarray([0.8, 0.1, 0.1]),
            ),
            observation_model=_operator(tau_low, [0.0, 0.0, 1.0]),
        ),
        ArchitectureCandidateRun(
            architecture_id=immune_low.architecture_id,
            parameter_draws=(immune_low, immune_high),
            initial_activity_draws=(
                np.asarray([0.8, 0.1, 0.1]),
                np.asarray([0.8, 0.1, 0.1]),
            ),
            observation_model=_operator(immune_low, [0.0, 0.0, 1.0]),
        ),
    )


def test_competing_architectures_are_simulated_and_scored_on_identical_holdout():
    catalog, registry = _catalog_registry()
    candidates = _candidates()
    time = np.asarray([0.0, 1.0, 2.0, 3.0])
    observed = np.asarray([[0.1], [0.16], [0.23], [0.29]])
    report = compare_mechanistic_architectures(
        evaluation_id="synthetic-common-holdout-v1",
        holdout_kind="temporal_holdout",
        catalog=catalog,
        registry=registry,
        time=time,
        observed=observed,
        measurement_sd=np.full_like(observed, 0.1),
        available_mask=np.ones_like(observed, dtype=bool),
        candidates=candidates,
    )
    assert report.report.evaluation_id == "synthetic-common-holdout-v1"
    assert report.report.interpretation == "held_out_predictive_comparison_not_causal_model_confirmation"
    assert len(report.forecasts) == 2
    assert {item.architecture_id for item in report.forecasts} == {
        "tau_neuronal_vulnerability_dominant",
        "immune_glial_dominant",
    }
    assert all(item.predictive_mean.shape == (4, 1) for item in report.forecasts)
    assert all(item.predictive_sd.shape == (4, 1) for item in report.forecasts)
    assert all(item.parameter_draw_count == 2 for item in report.forecasts)
    assert report.report.scores[0].rank == 1
    assert report.report.scores[1].rank == 2


def test_architecture_comparison_rejects_non_common_observation_contract():
    catalog, registry = _catalog_registry()
    candidates = list(_candidates())
    second = candidates[1]
    wrong_operator = MechanismObservationModel(
        model_id="bad-unit",
        version="1.0",
        mechanism_node_ids=second.observation_model.mechanism_node_ids,
        feature_names=("synaptic_signal",),
        feature_units=("percent",),
        H=second.observation_model.H,
        bias=second.observation_model.bias,
        R=second.observation_model.R,
        source_catalog_hash=second.observation_model.source_catalog_hash,
        transformation_id=second.observation_model.transformation_id,
        transformation_version=second.observation_model.transformation_version,
    )
    candidates[1] = ArchitectureCandidateRun(
        architecture_id=second.architecture_id,
        parameter_draws=second.parameter_draws,
        initial_activity_draws=second.initial_activity_draws,
        observation_model=wrong_operator,
    )
    with pytest.raises(ValueError, match="same units"):
        compare_mechanistic_architectures(
            evaluation_id="bad-common-contract",
            holdout_kind="external_cohort",
            catalog=catalog,
            registry=registry,
            time=np.asarray([0.0, 1.0, 2.0]),
            observed=np.asarray([[0.1], [0.2], [0.3]]),
            measurement_sd=np.full((3, 1), 0.1),
            available_mask=np.ones((3, 1), dtype=bool),
            candidates=tuple(candidates),
        )


def test_architecture_comparison_rejects_parameter_draw_topology_changes():
    catalog, registry = _catalog_registry()
    candidates = list(_candidates())
    first = candidates[0]
    changed = _model(
        first.architecture_id,
        (
            ("tau_misprocessing", 0.9, 0.05),
            ("tau_seeded_spread", 0.05, 0.08),
            ("synaptic_circuit_dysfunction", 0.05, 0.08),
        ),
        ("tau_to_spread",),
        0.5,
    )
    candidates[0] = ArchitectureCandidateRun(
        architecture_id=first.architecture_id,
        parameter_draws=(first.parameter_draws[0], changed),
        initial_activity_draws=first.initial_activity_draws,
        observation_model=first.observation_model,
    )
    with pytest.raises(ValueError, match="preserve node and edge topology"):
        compare_mechanistic_architectures(
            evaluation_id="bad-topology",
            holdout_kind="subject_holdout",
            catalog=catalog,
            registry=registry,
            time=np.asarray([0.0, 1.0, 2.0]),
            observed=np.asarray([[0.1], [0.2], [0.3]]),
            measurement_sd=np.full((3, 1), 0.1),
            available_mask=np.ones((3, 1), dtype=bool),
            candidates=tuple(candidates),
        )


def test_parameter_draws_cannot_change_structural_edge_sign():
    catalog, registry = _catalog_registry()
    first = _candidates()[0]
    base = first.parameter_draws[0]
    changed_edge = replace(base.edges[0], sign="inhibition")
    changed_model = replace(
        base,
        edges=(changed_edge, *base.edges[1:]),
    )
    altered = ArchitectureCandidateRun(
        architecture_id=first.architecture_id,
        parameter_draws=(base, changed_model),
        initial_activity_draws=first.initial_activity_draws,
        observation_model=first.observation_model,
    )
    with pytest.raises(ValueError, match="preserve edge direction/sign"):
        compare_mechanistic_architectures(
            evaluation_id="edge-sign-invariance",
            holdout_kind="temporal_holdout",
            catalog=catalog,
            registry=registry,
            time=np.asarray([0.0, 1.0, 2.0]),
            observed=np.asarray([[0.1], [0.2], [0.3]]),
            measurement_sd=np.full((3, 1), 0.1),
            available_mask=np.ones((3, 1), dtype=bool),
            candidates=(altered, _candidates()[1]),
        )
