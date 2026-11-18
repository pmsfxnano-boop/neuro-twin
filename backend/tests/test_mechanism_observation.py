"""Tests for explicit observation operators on mechanism-level states."""
import numpy as np
import pytest

from neuro_twin.network import (
    MechanismObservationModel,
    MechanismTrajectory,
    summarize_mechanism_forecast_ensemble,
)


def _operator():
    return MechanismObservationModel(
        model_id="alzheimer-mechanism-observation-test",
        version="1.0.0",
        mechanism_node_ids=("tau_misprocessing", "tau_seeded_spread"),
        feature_names=("tau_feature_a", "tau_feature_b"),
        feature_units=("zscore", "zscore"),
        H=np.asarray([[1.0, 0.0], [0.5, 0.5]]),
        bias=np.asarray([0.1, -0.1]),
        R=np.asarray([[0.2, 0.05], [0.05, 0.3]]),
        source_catalog_hash="a" * 64,
        transformation_id="test-standardization",
        transformation_version="1.0",
    )


def test_mechanism_observation_operator_requires_explicit_node_order_and_units():
    operator = _operator()
    trajectory = MechanismTrajectory(
        time=np.asarray([0.0, 1.0]),
        node_ids=("tau_misprocessing", "tau_seeded_spread"),
        activity=np.asarray([[0.2, 0.4], [0.5, 0.7]]),
    )
    prediction = operator.predict(trajectory)
    np.testing.assert_allclose(
        prediction.predictive_mean,
        np.asarray([[0.3, 0.2], [0.6, 0.5]]),
    )
    assert prediction.feature_names == ("tau_feature_a", "tau_feature_b")
    assert prediction.observation_covariance.shape == (2, 2)
    assert len(prediction.operator_hash) == 64


def test_mechanism_observation_model_rejects_order_mismatch_and_non_pd_noise():
    operator = _operator()
    wrong_order = MechanismTrajectory(
        time=np.asarray([0.0, 1.0]),
        node_ids=("tau_seeded_spread", "tau_misprocessing"),
        activity=np.asarray([[0.4, 0.2], [0.7, 0.5]]),
    )
    with pytest.raises(ValueError, match="exactly match"):
        operator.predict(wrong_order)

    with pytest.raises(ValueError, match="positive definite"):
        MechanismObservationModel(
            model_id="bad",
            version="1",
            mechanism_node_ids=("a",),
            feature_names=("f",),
            feature_units=("unit",),
            H=np.asarray([[1.0]]),
            bias=np.asarray([0.0]),
            R=np.asarray([[0.0]]),
            source_catalog_hash="hash",
            transformation_id="transform",
            transformation_version="1",
        )


def test_parameter_draw_ensemble_produces_prediction_mean_and_sd():
    operator = _operator()
    first = MechanismTrajectory(
        time=np.asarray([0.0, 1.0]),
        node_ids=("tau_misprocessing", "tau_seeded_spread"),
        activity=np.asarray([[0.2, 0.4], [0.5, 0.7]]),
    )
    second = MechanismTrajectory(
        time=np.asarray([0.0, 1.0]),
        node_ids=("tau_misprocessing", "tau_seeded_spread"),
        activity=np.asarray([[0.4, 0.4], [0.7, 0.7]]),
    )
    ensemble = summarize_mechanism_forecast_ensemble(
        architecture_id="tau_architecture",
        trajectories=(first, second),
        observation_model=operator,
    )
    assert ensemble.parameter_draw_count == 2
    assert ensemble.predictive_mean.shape == (2, 2)
    assert ensemble.predictive_sd.shape == (2, 2)
    assert np.all(ensemble.predictive_sd >= 0.0)
