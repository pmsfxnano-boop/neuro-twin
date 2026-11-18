"""Regression tests for the additive six-domain network research package.

All inputs are mathematical fixtures, not clinical or biological evidence.
"""
import numpy as np
import pytest
from scipy.linalg import expm

from neuro_twin.network import (
    DOMAIN_NAMES,
    DOMAIN_POLARITY,
    LinearNetworkObservationModel,
    SixDomainNetwork,
    exact_affine_transition,
    exact_network_transition,
    kalman_filter_network,
    rts_smoother_network,
    simulate_network,
)


def _network(*, disease="alzheimer", drift=None, coupling=None, candidate_edges=()):
    return SixDomainNetwork(
        disease=disease,
        time_unit="month",
        drift=np.zeros(6) if drift is None else drift,
        coupling_matrix=np.zeros((6, 6)) if coupling is None else coupling,
        candidate_edges=candidate_edges,
    )


def _observation_model():
    H = np.eye(6)
    R = np.eye(6) * 0.1
    return LinearNetworkObservationModel(
        H=H,
        R=R,
        feature_names=DOMAIN_NAMES,
        model_name="explicit-six-state-test",
        model_version="1.0.0",
    )


def test_domain_order_and_explicit_polarity_contract():
    assert DOMAIN_NAMES == ("P", "I", "M", "N", "Q", "Pr")
    assert "metabolic dysregulation" in DOMAIN_POLARITY["M"]


def test_exact_affine_transition_matches_matrix_exponential():
    coupling = np.diag([-0.1, -0.2, -0.3, -0.4, -0.5, -0.6])
    drift = np.arange(1.0, 7.0) / 10.0
    model = _network(
        drift=drift,
        coupling=coupling,
        candidate_edges=tuple((name, name) for name in DOMAIN_NAMES),
    )
    phi, offset = exact_affine_transition(model, 2.5)

    augmented = np.zeros((7, 7))
    augmented[:6, :6] = coupling
    augmented[:6, 6] = drift
    expected = expm(augmented * 2.5)
    np.testing.assert_allclose(phi, expected[:6, :6], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(offset, expected[:6, 6], rtol=1e-12, atol=1e-12)


def test_simulation_supports_irregular_visits_and_constant_drift():
    drift = np.asarray([0.1, 0.2, 0.0, -0.1, 0.0, 0.3])
    model = _network(drift=drift)
    initial = np.zeros(6)
    trajectory = simulate_network(model, initial, np.asarray([0.0, 1.0, 3.5]))
    np.testing.assert_allclose(
        trajectory.state,
        np.asarray([initial, drift, 3.5 * drift]),
        rtol=1e-12,
        atol=1e-12,
    )


def test_network_rejects_nonfinite_or_wrong_dimension_parameters():
    with pytest.raises(ValueError, match="coupling_matrix"):
        _network(coupling=np.zeros((4, 4)))
    with pytest.raises(ValueError, match="drift"):
        _network(drift=np.asarray([0.0, 0.0, np.nan, 0.0, 0.0, 0.0]))


def test_disease_must_be_explicit_and_supported():
    with pytest.raises(ValueError, match="disease"):
        SixDomainNetwork(
            disease="ftd",
            time_unit="month",
            drift=np.zeros(6),
            coupling_matrix=np.zeros((6, 6)),
            candidate_edges=(),
        )


def test_observation_operator_requires_explicit_six_state_mapping():
    with pytest.raises(ValueError, match="shape"):
        LinearNetworkObservationModel(
            H=np.eye(4),
            R=np.eye(4),
            feature_names=("a", "b", "c", "d"),
            model_name="bad",
            model_version="1",
        )


def test_observation_operator_rejects_non_positive_definite_noise():
    with pytest.raises(ValueError, match="positive definite"):
        LinearNetworkObservationModel(
            H=np.eye(6),
            R=np.zeros((6, 6)),
            feature_names=DOMAIN_NAMES,
            model_name="bad",
            model_version="1",
        )


def test_observation_configuration_hash_is_deterministic():
    first = _observation_model()
    second = _observation_model()
    assert first.config_hash == second.config_hash
    assert len(first.config_hash) == 64


def test_process_noise_discretization_is_positive_semidefinite():
    model = _network(
        coupling=np.diag([-0.1] * 6),
        candidate_edges=tuple((name, name) for name in DOMAIN_NAMES),
    )
    q = np.eye(6) * 0.05
    transition = exact_network_transition(model, 2.0, q)
    assert transition.phi.shape == (6, 6)
    assert np.linalg.eigvalsh(transition.q).min() >= -1e-10


def test_filter_handles_missing_observations_and_preserves_covariance():
    model = _network(
        drift=np.ones(6) * 0.01,
        coupling=np.diag([-0.01] * 6),
        candidate_edges=tuple((name, name) for name in DOMAIN_NAMES),
    )
    observation_model = _observation_model()
    observations = np.asarray([
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        [0.1, np.nan, 0.2, 0.3, 0.4, 0.5],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    ])
    result = kalman_filter_network(
        time=np.asarray([0.0, 1.0, 3.0]),
        observations=observations,
        observation_model=observation_model,
        initial_mean=np.zeros(6),
        initial_covariance=np.eye(6),
        network=model,
        process_noise_spectral_density=np.eye(6) * 0.001,
    )

    assert result.filtered_state.shape == (3, 6)
    assert result.filtered_covariance.shape == (3, 6, 6)
    assert result.innovations[2] is None
    assert len(result.observed_channel_indices[1]) == 5
    for covariance in result.filtered_covariance:
        assert np.linalg.eigvalsh(covariance).min() >= -1e-9


def test_rts_smoother_is_separate_from_forward_filter():
    model = _network(
        coupling=np.diag([-0.05] * 6),
        candidate_edges=tuple((name, name) for name in DOMAIN_NAMES),
    )
    observations = np.asarray([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.2, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.3, 0.2, 0.1, 0.0, 0.0, 0.0],
    ])
    filtered = kalman_filter_network(
        np.asarray([0.0, 1.0, 2.0]),
        observations,
        _observation_model(),
        np.zeros(6),
        np.eye(6),
        model,
    )
    smoothed = rts_smoother_network(filtered)
    assert smoothed.smoothed_state.shape == (3, 6)
    assert smoothed.smoothed_covariance.shape == (3, 6, 6)
    assert np.isfinite(filtered.log_likelihood)
    assert smoothed.smoothing_gains.shape == (2, 6, 6)


def test_filter_rejects_infinite_observation_values():
    with pytest.raises(ValueError, match="infinity"):
        kalman_filter_network(
            np.asarray([0.0]),
            np.asarray([[np.inf] * 6]),
            _observation_model(),
            np.zeros(6),
            np.eye(6),
            _network(),
        )


def test_coupling_matrix_cannot_activate_an_undeclared_edge():
    coupling = np.zeros((6, 6))
    coupling[1, 0] = 0.25  # I influences dI/dt? Entry [target=I, source=P].
    with pytest.raises(ValueError, match="outside candidate_edges"):
        _network(coupling=coupling, candidate_edges=())


def test_parameter_vector_round_trip_preserves_explicit_graph():
    coupling = np.zeros((6, 6))
    coupling[1, 0] = 0.25  # P -> I
    coupling[4, 3] = -0.1  # N -> Q
    edges = (("P", "I"), ("N", "Q"))
    model = _network(drift=np.arange(6, dtype=float), coupling=coupling, candidate_edges=edges)
    vector = model.as_parameter_vector()
    rebuilt = model.with_parameter_vector(vector)
    np.testing.assert_allclose(rebuilt.drift, model.drift)
    np.testing.assert_allclose(rebuilt.coupling_matrix, model.coupling_matrix)
    assert rebuilt.candidate_edges == model.candidate_edges
    assert rebuilt.parameter_count == 8
