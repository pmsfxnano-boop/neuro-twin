"""Tests for time-window sweeps and parameter-draw robustness."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from neuro_twin.network import (
    AlzheimerMechanismCatalog,
    AlzheimerOriginArchitectureRegistry,
    DamageAccumulationModel,
    EdgeKinetics,
    InterventionTimingWindow,
    MechanisticArchitectureModel,
    NodeKinetics,
    sweep_intervention_timing,
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
        parameter_status="hypothesis",
    )


def test_timing_sweep_reports_damage_reduction_and_draw_robustness():
    catalog, registry = _catalog_registry()
    models = (_model(0.7), _model(0.9))
    result = sweep_intervention_timing(
        models,
        catalog=catalog,
        registry=registry,
        time=np.asarray([0.0, 1.0, 2.0, 3.0, 4.0]),
        initial_activity_draws=(
            np.asarray([0.8, 0.1, 0.1]),
            np.asarray([0.8, 0.1, 0.1]),
        ),
        initial_damage=np.zeros(4),
        target_node="tau_misprocessing",
        mode="clamp",
        intensity=0.0,
        windows=(
            InterventionTimingWindow(0.5, 1.5),
            InterventionTimingWindow(2.5, 3.5),
        ),
    )
    assert result.architecture_id == "tau_neuronal_vulnerability_dominant"
    assert result.parameter_draw_count == 2
    assert len(result.effects) == 2
    for effect in result.effects:
        assert np.all(effect.damage_prevented_median > 0.0)
        assert np.all(effect.fraction_parameter_draws_beneficial == 1.0)
        assert np.all(effect.fraction_parameter_draws_harmful == 0.0)
        assert effect.damage_prevented_interval.shape == (2, 4)
        assert effect.damage_burden_auc_prevented_interval.shape == (2, 4)
    assert result.interpretation == "model_conditional_timing_sweep_not_empirical_treatment_window"


def test_timing_sweep_rejects_inconsistent_model_draws_and_out_of_range_windows():
    catalog, registry = _catalog_registry()
    models = (_model(0.7), _model(0.9))
    with pytest.raises(ValueError, match="within simulation time"):
        sweep_intervention_timing(
            models,
            catalog=catalog,
            registry=registry,
            time=np.asarray([0.0, 1.0, 2.0]),
            initial_activity_draws=(
                np.asarray([0.8, 0.1, 0.1]),
                np.asarray([0.8, 0.1, 0.1]),
            ),
            initial_damage=np.zeros(4),
            target_node="tau_misprocessing",
            mode="clamp",
            intensity=0.0,
            windows=(InterventionTimingWindow(1.0, 2.1),),
        )
    different_architecture = replace(_model(0.8), architecture_id="immune_glial_dominant")
    with pytest.raises(ValueError):
        sweep_intervention_timing(
            (models[0], different_architecture),
            catalog=catalog,
            registry=registry,
            time=np.asarray([0.0, 1.0, 2.0]),
            initial_activity_draws=(
                np.asarray([0.8, 0.1, 0.1]),
                np.asarray([0.8, 0.1, 0.1]),
            ),
            initial_damage=np.zeros(4),
            target_node="tau_misprocessing",
            mode="clamp",
            intensity=0.0,
            windows=(InterventionTimingWindow(0.5, 1.5),),
        )
