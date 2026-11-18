"""Tests for shared-kinetic calibration over multiple subject trajectories."""
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
    MechanisticSubjectSeries,
    NodeKinetics,
    ParameterBound,
    SubjectInitialStateBound,
    fit_mechanistic_cohort_parameters,
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


def _model(coefficient):
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


def _operator(model, visible_node):
    weights = np.zeros(len(model.node_ids))
    weights[model.node_ids.index(visible_node)] = 1.0
    return MechanismObservationModel(
        model_id="multi-subject-calibration",
        version="1",
        mechanism_node_ids=model.node_ids,
        feature_names=("tau_spread_feature",),
        feature_units=("zscore",),
        H=weights.reshape(1, -1),
        bias=np.zeros(1),
        R=np.asarray([[0.01]]),
        source_catalog_hash="e" * 64,
        transformation_id="synthetic-cohort-transform",
        transformation_version="1",
    )


def _subject(subject_id, truth_model, catalog, registry, operator, time, initial):
    trajectory = truth_model.simulate(
        catalog=catalog,
        registry=registry,
        time=time,
        initial_activity=initial,
    )
    observed = operator.predict(trajectory).predictive_mean
    return MechanisticSubjectSeries(
        subject_id=subject_id,
        time=time,
        observed=observed,
        available_mask=np.ones_like(observed, dtype=bool),
        initial_activity=initial,
    )


def test_shared_kinetic_calibration_recovers_parameter_across_irregular_subject_series():
    catalog, registry = _catalog_registry()
    truth = _model(0.9)
    start = _model(0.4)
    operator = _operator(start, "tau_seeded_spread")
    subjects = (
        _subject(
            "subject_a",
            truth,
            catalog,
            registry,
            operator,
            np.linspace(0.0, 4.0, 17),
            np.asarray([0.8, 0.1, 0.1]),
        ),
        _subject(
            "subject_b",
            truth,
            catalog,
            registry,
            operator,
            np.asarray([0.0, 0.3, 1.0, 1.9, 2.8, 4.0]),
            np.asarray([0.65, 0.15, 0.08]),
        ),
        _subject(
            "subject_c",
            truth,
            catalog,
            registry,
            operator,
            np.asarray([0.0, 0.2, 0.7, 1.4, 2.2, 3.0, 3.6, 4.0]),
            np.asarray([0.9, 0.05, 0.05]),
        ),
    )
    result = fit_mechanistic_cohort_parameters(
        start,
        catalog=catalog,
        registry=registry,
        observation_model=operator,
        subjects=subjects,
        shared_parameter_bounds=(
            ParameterBound(
                "edge:tau_to_spread:coefficient",
                initial=0.4,
                lower=0.05,
                upper=2.0,
            ),
        ),
    )
    assert result.subject_count == 3
    assert result.observation_count == sum(len(item.time) for item in subjects)
    assert result.fit_status == "COHORT_LOCALLY_FULL_RANK_CANDIDATE_FIT"
    assert result.identifiable_rank == 1
    assert result.parameter_estimates[0] == pytest.approx(0.9, abs=2e-4)
    assert result.fitted_model.parameter_status == "hypothesis"
    assert {x.subject_id for x in result.subject_predictions} == {
        "subject_a",
        "subject_b",
        "subject_c",
    }
    assert result.interpretation == "shared_kinetic_cohort_fit_not_causal_or_external_validation"


def test_cohort_calibration_can_fit_subject_specific_initial_states():
    catalog, registry = _catalog_registry()
    truth = _model(0.8)
    start = _model(0.8)
    operator = _operator(start, "tau_seeded_spread")
    time = np.linspace(0.0, 3.0, 13)
    subjects = (
        _subject(
            "subject_a", truth, catalog, registry, operator, time,
            np.asarray([0.8, 0.1, 0.1]),
        ),
        _subject(
            "subject_b", truth, catalog, registry, operator, time,
            np.asarray([0.7, 0.2, 0.1]),
        ),
    )
    result = fit_mechanistic_cohort_parameters(
        start,
        catalog=catalog,
        registry=registry,
        observation_model=operator,
        subjects=subjects,
        shared_parameter_bounds=(),
        subject_initial_state_bounds=(
            SubjectInitialStateBound("subject_a", "tau_seeded_spread", 0.08, 0.0, 0.4),
            SubjectInitialStateBound("subject_b", "tau_seeded_spread", 0.18, 0.0, 0.4),
        ),
    )
    assert result.parameter_count == 2
    assert result.subject_count == 2
    assert np.all(np.isfinite(result.parameter_estimates))
    assert all(x.initial_activity.shape == (3,) for x in result.subject_predictions)


def test_cohort_calibration_rejects_duplicate_ids_unknown_subject_and_single_series():
    catalog, registry = _catalog_registry()
    model = _model(0.5)
    operator = _operator(model, "tau_seeded_spread")
    subject = MechanisticSubjectSeries(
        subject_id="same",
        time=np.asarray([0.0, 1.0]),
        observed=np.zeros((2, 1)),
        available_mask=np.ones((2, 1), dtype=bool),
        initial_activity=np.asarray([0.8, 0.1, 0.1]),
    )
    kwargs = {
        "catalog": catalog,
        "registry": registry,
        "observation_model": operator,
        "shared_parameter_bounds": (
            ParameterBound("edge:tau_to_spread:coefficient", 0.5, 0.1, 1.5),
        ),
    }
    with pytest.raises(ValueError, match="at least two subjects"):
        fit_mechanistic_cohort_parameters(
            model, subjects=(subject,), **kwargs
        )
    with pytest.raises(ValueError, match="unique"):
        fit_mechanistic_cohort_parameters(
            model, subjects=(subject, subject), **kwargs
        )
    with pytest.raises(ValueError, match="unknown subject"):
        fit_mechanistic_cohort_parameters(
            model,
            subjects=(subject, MechanisticSubjectSeries(
                subject_id="other",
                time=np.asarray([0.0, 1.0]),
                observed=np.zeros((2, 1)),
                available_mask=np.ones((2, 1), dtype=bool),
                initial_activity=np.asarray([0.8, 0.1, 0.1]),
            )),
            subject_initial_state_bounds=(
                SubjectInitialStateBound("missing", "tau_misprocessing", 0.5, 0.0, 1.0),
            ),
            shared_parameter_bounds=(),
            catalog=catalog,
            registry=registry,
            observation_model=operator,
        )
