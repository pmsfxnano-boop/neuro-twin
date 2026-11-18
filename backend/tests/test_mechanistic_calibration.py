"""Tests for constrained mechanism parameter calibration and identifiability."""
import json
from pathlib import Path

import numpy as np
import pytest

from neuro_twin.network import (
    AlzheimerMechanismCatalog,
    AlzheimerOriginArchitectureRegistry,
    DamageAccumulationModel,
    EdgeKinetics,
    MechanismObservationModel,
    MechanisticArchitectureModel,
    NodeKinetics,
    ParameterBound,
    fit_mechanistic_parameters,
)


BASE = Path(__file__).resolve().parents[1] / "neuro_twin" / "network"


def _catalog_registry():
    catalog = AlzheimerMechanismCatalog.model_validate(
        json.loads((BASE / "alzheimer_mechanism_map_v1.json").read_text(encoding="utf-8"))
    )
    registry = AlzheimerOriginArchitectureRegistry.model_validate(
        json.loads((BASE / "alzheimer_origin_architectures_v1.json").read_text(encoding="utf-8"))
    )
    registry.validate_against(catalog)
    return catalog, registry


def _model(coefficient):
    catalog, registry = _catalog_registry()
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
    )
    model.validate_against(catalog, registry)
    return catalog, registry, model


def _observation_model(model, H):
    return MechanismObservationModel(
        model_id="calibration-test-operator",
        version="1.0",
        mechanism_node_ids=model.node_ids,
        feature_names=("tau_seeded_spread_signal",),
        feature_units=("zscore",),
        H=np.asarray(H, dtype=float).reshape(1, len(model.node_ids)),
        bias=np.asarray([0.0]),
        R=np.asarray([[0.01]]),
        source_catalog_hash="c" * 64,
        transformation_id="synthetic-test-transform",
        transformation_version="1",
    )


def test_calibrator_recovers_one_selected_edge_parameter_from_synthetic_observations():
    catalog, registry, truth_model = _model(0.9)
    _, _, starting_model = _model(0.4)
    time = np.linspace(0.0, 4.0, 17)
    initial = np.asarray([0.8, 0.1, 0.1])
    operator = _observation_model(starting_model, [0.0, 1.0, 0.0])
    truth_trajectory = truth_model.simulate(
        catalog=catalog,
        registry=registry,
        time=time,
        initial_activity=initial,
    )
    observed = operator.predict(truth_trajectory).predictive_mean
    result = fit_mechanistic_parameters(
        starting_model,
        catalog=catalog,
        registry=registry,
        observation_model=operator,
        time=time,
        observed=observed,
        available_mask=np.ones_like(observed, dtype=bool),
        initial_activity=initial,
        parameter_bounds=(
            ParameterBound(
                name="edge:tau_to_spread:coefficient",
                initial=0.4,
                lower=0.05,
                upper=2.0,
            ),
        ),
    )
    assert result.fit_status == "LOCALLY_FULL_RANK_CANDIDATE_FIT"
    assert result.identifiable_rank == 1
    assert result.parameter_names == ("edge:tau_to_spread:coefficient",)
    assert result.parameter_estimates[0] == pytest.approx(0.9, abs=2e-4)
    assert result.fitted_model.parameter_status == "hypothesis"
    assert result.interpretation == "candidate_parameter_fit_not_biological_validation"


def test_calibrator_reports_rank_deficiency_when_observation_has_no_sensitivity():
    catalog, registry, model = _model(0.4)
    time = np.linspace(0.0, 3.0, 8)
    initial = np.asarray([0.8, 0.1, 0.1])
    operator = _observation_model(model, [0.0, 0.0, 0.0])
    observed = np.zeros((len(time), 1))
    result = fit_mechanistic_parameters(
        model,
        catalog=catalog,
        registry=registry,
        observation_model=operator,
        time=time,
        observed=observed,
        available_mask=np.ones_like(observed, dtype=bool),
        initial_activity=initial,
        parameter_bounds=(
            ParameterBound(
                name="edge:tau_to_spread:coefficient",
                initial=0.4,
                lower=0.05,
                upper=2.0,
            ),
        ),
    )
    assert result.fit_status == "RANK_DEFICIENT_CANDIDATE_FIT"
    assert result.identifiable_rank == 0
    assert np.isinf(result.condition_number)


def test_parameter_bounds_reject_invalid_or_unidentified_parameter_names():
    catalog, registry, model = _model(0.4)
    time = np.asarray([0.0, 1.0])
    initial = np.asarray([0.8, 0.1, 0.1])
    operator = _observation_model(model, [0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="unsupported or unknown"):
        fit_mechanistic_parameters(
            model,
            catalog=catalog,
            registry=registry,
            observation_model=operator,
            time=time,
            observed=np.zeros((2, 1)),
            available_mask=np.ones((2, 1), dtype=bool),
            initial_activity=initial,
            parameter_bounds=(
                ParameterBound("edge:missing_edge:coefficient", 0.2, 0.0, 1.0),
            ),
        )
