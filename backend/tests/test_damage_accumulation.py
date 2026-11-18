"""Tests for the cumulative persistent-damage sidecar module."""
import numpy as np
import pytest

from neuro_twin.network.damage import (
    DAMAGE_COMPONENTS,
    DamageAccumulationModel,
)


def _model(*, repair_weight=0.0, disease="alzheimer"):
    return DamageAccumulationModel(
        disease=disease,
        time_unit="year",
        driver_names=("injury_pressure",),
        repair_names=("repair_capacity",),
        damage_weights=np.ones((4, 1)),
        repair_weights=np.ones((4, 1)) * repair_weight,
        rate_scale=np.ones(4),
    )


def test_damage_integrator_uses_irregular_time_and_trapezoidal_exposure():
    model = _model()
    result = model.accumulate(
        time=np.asarray([0.0, 1.0, 3.0]),
        damaging_exposures=np.asarray([[0.2], [0.4], [0.8]]),
        repair_capacity=np.zeros((3, 1)),
        initial_damage=np.asarray([1.0, 2.0, 3.0, 4.0]),
    )

    expected = np.asarray([2.5, 3.5, 4.5, 5.5])
    np.testing.assert_allclose(result.cumulative_damage[-1], expected)
    np.testing.assert_allclose(
        result.damage_rate,
        np.repeat(np.asarray([[0.2], [0.4], [0.8]]), 4, axis=1),
    )


def test_repair_capacity_prevents_additional_damage_but_never_erases_past_damage():
    model = _model(repair_weight=1.0)
    result = model.accumulate(
        time=np.asarray([0.0, 1.0, 2.0]),
        damaging_exposures=np.asarray([[0.8], [0.8], [0.2]]),
        repair_capacity=np.asarray([[0.2], [0.2], [0.8]]),
        initial_damage=np.asarray([0.5, 0.4, 0.3, 0.2]),
    )

    # First interval: excess pressure is 0.6 at both ends. Second:
    # the rate decreases from 0.6 to zero; accumulated damage stays flat
    # or increases, but never declines.
    assert np.all(np.diff(result.cumulative_damage, axis=0) >= 0.0)
    np.testing.assert_allclose(result.cumulative_damage[0], [0.5, 0.4, 0.3, 0.2])
    np.testing.assert_allclose(result.damage_rate[-1], np.zeros(4))
    np.testing.assert_allclose(result.cumulative_damage[-1], [1.4, 1.3, 1.2, 1.1])


def test_damage_model_requires_explicit_nonnegative_coefficients():
    with pytest.raises(ValueError, match="non-negative"):
        DamageAccumulationModel(
            disease="alzheimer",
            time_unit="year",
            driver_names=("stress",),
            repair_names=("repair",),
            damage_weights=-np.ones((4, 1)),
            repair_weights=np.zeros((4, 1)),
            rate_scale=np.ones(4),
        )


def test_damage_model_rejects_duplicate_or_overlapping_input_names():
    with pytest.raises(ValueError, match="unique"):
        DamageAccumulationModel(
            disease="alzheimer",
            time_unit="year",
            driver_names=("stress", "stress"),
            repair_names=("repair",),
            damage_weights=np.ones((4, 2)),
            repair_weights=np.zeros((4, 1)),
            rate_scale=np.ones(4),
        )
    with pytest.raises(ValueError, match="distinct"):
        DamageAccumulationModel(
            disease="alzheimer",
            time_unit="year",
            driver_names=("stress",),
            repair_names=("stress",),
            damage_weights=np.ones((4, 1)),
            repair_weights=np.zeros((4, 1)),
            rate_scale=np.ones(4),
        )


def test_damage_model_rejects_non_monotonic_time_and_unscaled_inputs():
    model = _model()
    common = {
        "damaging_exposures": np.asarray([[0.2], [0.3]]),
        "repair_capacity": np.zeros((2, 1)),
        "initial_damage": np.zeros(4),
    }
    with pytest.raises(ValueError, match="strictly increasing"):
        model.accumulate(time=np.asarray([1.0, 1.0]), **common)

    invalid = dict(common)
    invalid["damaging_exposures"] = np.asarray([[0.2], [1.2]])
    with pytest.raises(ValueError, match="normalized to"):
        model.accumulate(time=np.asarray([0.0, 1.0]), **invalid)


def test_damage_hash_is_deterministic_and_disease_scoped():
    first = _model()
    second = _model()
    parkinson = _model(disease="parkinson")
    assert first.config_hash == second.config_hash
    assert first.config_hash != parkinson.config_hash
    assert len(first.config_hash) == 64
    assert first.component_names == DAMAGE_COMPONENTS


def test_damage_model_can_explicitly_assign_no_repair_credit():
    model = DamageAccumulationModel(
        disease="alzheimer",
        time_unit="year",
        driver_names=("stress",),
        repair_names=(),
        damage_weights=np.ones((4, 1)),
        repair_weights=np.zeros((4, 0)),
        rate_scale=np.ones(4),
    )
    result = model.accumulate(
        time=np.asarray([0.0, 1.0]),
        damaging_exposures=np.asarray([[0.4], [0.4]]),
        repair_capacity=np.zeros((2, 0)),
        initial_damage=np.zeros(4),
    )
    np.testing.assert_allclose(result.cumulative_damage[-1], np.full(4, 0.4))
