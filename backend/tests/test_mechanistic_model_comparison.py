"""Tests for held-out predictive comparison of rival model architectures."""
import numpy as np
import pytest

from neuro_twin.network import ArchitectureForecast, compare_architecture_forecasts


def test_architecture_forecasts_are_ranked_by_held_out_log_score():
    observed = np.asarray([0.1, 1.0, 2.1, 3.0])
    measurement_sd = np.full(4, 0.1)
    mask = np.asarray([True, True, True, True])
    forecasts = (
        ArchitectureForecast(
            architecture_id="amyloid_origin",
            predictive_mean=np.asarray([0.0, 1.1, 2.0, 3.1]),
            predictive_sd=np.full(4, 0.15),
        ),
        ArchitectureForecast(
            architecture_id="vascular_origin",
            predictive_mean=np.asarray([1.0, 0.0, 3.0, 2.0]),
            predictive_sd=np.full(4, 0.15),
        ),
    )
    report = compare_architecture_forecasts(
        evaluation_id="adni_temporal_holdout_v1",
        holdout_kind="temporal_holdout",
        observed=observed,
        measurement_sd=measurement_sd,
        available_mask=mask,
        forecasts=forecasts,
    )
    assert report.scores[0].architecture_id == "amyloid_origin"
    assert report.scores[0].rank == 1
    assert report.scores[0].mean_log_predictive_density > report.scores[1].mean_log_predictive_density
    assert report.scores[0].n_scored == 4
    assert report.interpretation == "held_out_predictive_comparison_not_causal_model_confirmation"


def test_architecture_comparison_requires_common_held_out_data_and_uncertainty():
    observed = np.asarray([1.0, 2.0])
    forecasts = (
        ArchitectureForecast("a", np.asarray([1.0, 2.0]), np.asarray([0.2, 0.2])),
        ArchitectureForecast("b", np.asarray([1.1, 2.1]), np.asarray([0.2, 0.2])),
    )
    with pytest.raises(ValueError, match="evaluation_id"):
        compare_architecture_forecasts(
            evaluation_id="",
            holdout_kind="temporal_holdout",
            observed=observed,
            measurement_sd=np.asarray([0.1, 0.1]),
            available_mask=np.asarray([True, True]),
            forecasts=forecasts,
        )
    with pytest.raises(ValueError, match="held-out"):
        compare_architecture_forecasts(
            evaluation_id="bad",
            holdout_kind="not_a_holdout",
            observed=observed,
            measurement_sd=np.asarray([0.1, 0.1]),
            available_mask=np.asarray([True, True]),
            forecasts=forecasts,
        )
    bad = (
        forecasts[0],
        ArchitectureForecast("b", np.asarray([1.1, 2.1]), np.asarray([0.2, 0.0])),
    )
    with pytest.raises(ValueError, match="predictive_sd must be > 0"):
        compare_architecture_forecasts(
            evaluation_id="test",
            holdout_kind="external_cohort",
            observed=observed,
            measurement_sd=np.asarray([0.1, 0.1]),
            available_mask=np.asarray([True, True]),
            forecasts=bad,
        )


def test_zero_parameter_spread_is_valid_when_measurement_uncertainty_is_nonzero():
    observed = np.asarray([1.0, 2.0])
    zero_spread = np.zeros(2)
    forecasts = (
        ArchitectureForecast("a", np.asarray([1.0, 2.0]), zero_spread),
        ArchitectureForecast("b", np.asarray([1.1, 2.1]), zero_spread),
    )
    report = compare_architecture_forecasts(
        evaluation_id="fixed_parameter_holdout",
        holdout_kind="subject_holdout",
        observed=observed,
        measurement_sd=np.asarray([0.1, 0.1]),
        available_mask=np.asarray([True, True]),
        forecasts=forecasts,
    )
    assert report.scores[0].architecture_id == "a"
