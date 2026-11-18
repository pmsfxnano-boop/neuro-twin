"""Tests for six-domain observability and local identifiability diagnostics.

Fixtures are mathematical counterexamples and controls, not clinical data.
"""
import numpy as np
import pytest

from neuro_twin.network import (
    DOMAIN_NAMES,
    LinearNetworkObservationModel,
    SixDomainNetwork,
    finite_horizon_observability_report,
    local_parameter_identifiability_report,
    structural_observability_report,
)


def _network(*, coupling=None, candidate_edges=()):
    return SixDomainNetwork(
        disease="alzheimer",
        time_unit="month",
        drift=np.zeros(6),
        coupling_matrix=np.zeros((6, 6)) if coupling is None else coupling,
        candidate_edges=candidate_edges,
    )


def _observation(H, R=None, names=None):
    H = np.asarray(H, dtype=float)
    n_features = H.shape[0]
    return LinearNetworkObservationModel(
        H=H,
        R=np.eye(n_features) if R is None else np.asarray(R, dtype=float),
        feature_names=tuple(names or (f"channel_{i}" for i in range(n_features))),
        model_name="explicit-diagnostics-test",
        model_version="1.0.0",
    )


def test_structural_observability_full_rank_when_all_states_are_observed():
    report = structural_observability_report(_network(), _observation(np.eye(6)))
    assert report.subspace.status == "FULL_RANK"
    assert report.subspace.rank == 6
    assert report.subspace.unidentifiable_basis.shape == (6, 0)
    assert report.subspace.condition_number == pytest.approx(1.0)


def test_structural_observability_detects_unobserved_states():
    H = np.zeros((1, 6))
    H[0, 4] = 1.0  # only Q is observed; no cross-domain dynamics
    report = structural_observability_report(_network(), _observation(H))
    assert report.subspace.status == "PARTIAL"
    assert report.subspace.rank == 1
    assert report.subspace.unidentifiable_basis.shape == (6, 5)


def test_structural_observability_uses_declared_dynamic_coupling():
    coupling = np.zeros((6, 6))
    coupling[4, 0] = 0.2  # P -> Q; A[target=Q, source=P]
    network = _network(coupling=coupling, candidate_edges=(("P", "Q"),))
    H = np.zeros((1, 6))
    H[0, 4] = 1.0
    report = structural_observability_report(network, _observation(H))
    assert report.subspace.rank == 2
    assert report.subspace.status == "PARTIAL"


def test_finite_horizon_observability_accounts_for_irregular_schedule():
    coupling = np.zeros((6, 6))
    coupling[4, 0] = 0.2
    network = _network(coupling=coupling, candidate_edges=(("P", "Q"),))
    H = np.zeros((1, 6))
    H[0, 4] = 1.0
    model = _observation(H)
    times = np.asarray([0.0, 0.5, 2.0])
    available = np.ones((3, 1), dtype=bool)

    full = finite_horizon_observability_report(
        times, model, network, available_mask=available
    )
    first_visit_only = finite_horizon_observability_report(
        times,
        model,
        network,
        available_mask=np.asarray([[True], [False], [False]]),
    )

    assert full.subspace.rank == 2
    assert first_visit_only.subspace.rank == 1
    assert full.observation_counts_by_visit == (1, 1, 1)
    assert first_visit_only.observation_counts_by_visit == (1, 0, 0)
    assert np.linalg.eigvalsh(full.information_matrix).min() >= -1e-10


def test_finite_horizon_observability_rejects_malformed_masks():
    model = _observation(np.eye(6))
    with pytest.raises(ValueError, match="available_mask"):
        finite_horizon_observability_report(
            np.asarray([0.0, 1.0]),
            model,
            _network(),
            available_mask=np.ones((2, 5), dtype=bool),
        )
    with pytest.raises(ValueError, match="boolean"):
        finite_horizon_observability_report(
            np.asarray([0.0, 1.0]),
            model,
            _network(),
            available_mask=np.ones((2, 6), dtype=int),
        )


def test_parameter_identifiability_profiles_out_initial_state_and_finds_confounding():
    # With P constant at 1, the effects of drift_I and A[P -> I] are both
    # proportional to elapsed time. The observations cannot distinguish them.
    coupling = np.zeros((6, 6))
    network = _network(coupling=coupling, candidate_edges=(("P", "I"),))
    H = np.zeros((1, 6))
    H[0, 1] = 1.0
    model = _observation(H, R=np.asarray([[0.25]]), names=("I_channel",))
    report = local_parameter_identifiability_report(
        np.asarray([0.0, 1.0, 2.0, 3.0]),
        model,
        network,
        np.asarray([1.0, 0.2, 0.0, 0.0, 0.0, 0.0]),
    )

    assert report.status == "RANK_DEFICIENT"
    assert report.parameter_names[6] == "A:P->I"
    assert report.rank == 1
    assert report.unidentifiable_parameter_combinations.shape == (7, 6)
    assert report.initial_state_rank == 1


def test_parameter_identifiability_is_scale_explicit_and_deterministic():
    network = _network(candidate_edges=(("P", "I"),))
    H = np.eye(6)
    model = _observation(H)
    args = (
        np.asarray([0.0, 1.0, 2.0]),
        model,
        network,
        np.ones(6),
    )
    first = local_parameter_identifiability_report(
        *args,
        initial_state_scales=np.ones(6),
        parameter_scales=np.ones(network.parameter_count),
    )
    second = local_parameter_identifiability_report(
        *args,
        initial_state_scales=np.ones(6),
        parameter_scales=np.ones(network.parameter_count),
    )
    np.testing.assert_allclose(
        first.profiled_parameter_sensitivity,
        second.profiled_parameter_sensitivity,
    )
    np.testing.assert_allclose(first.parameter_scales, np.ones(network.parameter_count))


def test_parameter_identifiability_rejects_an_empty_observation_schedule():
    network = _network()
    model = _observation(np.eye(6))
    with pytest.raises(ValueError, match="at least one observed channel"):
        local_parameter_identifiability_report(
            np.asarray([0.0, 1.0]),
            model,
            network,
            np.zeros(6),
            available_mask=np.zeros((2, 6), dtype=bool),
        )
