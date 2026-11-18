"""Observability and local-identifiability diagnostics for six-domain networks.

These diagnostics are mathematical gates, not evidence that a clinical cohort
actually measures all six biological domains. All rank statements are local
to the supplied model, observation operator, schedule, masks, and scaling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from neuro_twin.network.model import SixDomainNetwork, simulate_network
from neuro_twin.network.observation import LinearNetworkObservationModel
from neuro_twin.network.state_space import exact_network_transition


RankStatus = Literal["FULL_RANK", "PARTIAL", "ZERO_INFORMATION"]


@dataclass(frozen=True)
class SubspaceDiagnostics:
    """Rank and identifiable/unidentifiable subspaces for a supplied matrix."""

    status: RankStatus
    rank: int
    dimension: int
    singular_values: np.ndarray
    relative_tolerance: float
    absolute_tolerance: float
    condition_number: float
    identifiable_basis: np.ndarray
    unidentifiable_basis: np.ndarray


@dataclass(frozen=True)
class StructuralObservabilityReport:
    observability_matrix: np.ndarray
    subspace: SubspaceDiagnostics


@dataclass(frozen=True)
class FiniteHorizonObservabilityReport:
    information_matrix: np.ndarray
    subspace: SubspaceDiagnostics
    observation_counts_by_visit: tuple[int, ...]
    time_origin: float


@dataclass(frozen=True)
class LocalParameterIdentifiabilityReport:
    parameter_names: tuple[str, ...]
    status: Literal[
        "LOCALLY_FULL_COLUMN_RANK",
        "RANK_DEFICIENT",
        "NO_LOCAL_INFORMATION",
    ]
    rank: int
    parameter_count: int
    singular_values: np.ndarray
    relative_tolerance: float
    absolute_tolerance: float
    condition_number: float
    identifiable_parameter_combinations: np.ndarray
    unidentifiable_parameter_combinations: np.ndarray
    profiled_parameter_sensitivity: np.ndarray
    initial_state_rank: int
    initial_state_scales: np.ndarray
    parameter_scales: np.ndarray
    relative_step: float


def _validate_relative_tolerance(value: float) -> float:
    tolerance = float(value)
    if not np.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise ValueError("relative_tolerance must be finite and between 0 and 1")
    return tolerance


def _validate_time(time: np.ndarray, *, minimum_points: int = 1) -> np.ndarray:
    values = np.asarray(time, dtype=float)
    if (
        values.ndim != 1
        or len(values) < minimum_points
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(
            f"time must be a finite 1-D array with at least {minimum_points} point(s)"
        )
    if len(values) > 1 and np.any(np.diff(values) <= 0.0):
        raise ValueError("time must be strictly increasing")
    return values


def _validate_mask(
    available_mask: np.ndarray | None,
    *,
    n_time: int,
    n_features: int,
) -> np.ndarray:
    if available_mask is None:
        return np.ones((n_time, n_features), dtype=bool)
    mask = np.asarray(available_mask)
    if mask.shape != (n_time, n_features):
        raise ValueError(
            f"available_mask must have shape ({n_time}, {n_features})"
        )
    if mask.dtype != np.bool_:
        raise ValueError("available_mask must be boolean")
    return mask


def _svd_diagnostics(
    matrix: np.ndarray,
    *,
    dimension: int,
    relative_tolerance: float,
) -> SubspaceDiagnostics:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[1] != dimension:
        raise ValueError("diagnostic matrix has an invalid shape")
    _, singular_values, vh = np.linalg.svd(values, full_matrices=True)
    leading = float(singular_values[0]) if len(singular_values) else 0.0
    absolute_tolerance = relative_tolerance * leading
    rank = int(np.sum(singular_values > absolute_tolerance)) if leading > 0 else 0
    # With full_matrices=True, Vh always spans the complete parameter/state space.
    identifiable = vh[:rank, :].T.copy()
    unidentifiable = vh[rank:, :].T.copy()
    if rank == 0:
        status: RankStatus = "ZERO_INFORMATION"
    elif rank == dimension:
        status = "FULL_RANK"
    else:
        status = "PARTIAL"
    condition = (
        float(singular_values[0] / singular_values[dimension - 1])
        if rank == dimension and dimension > 0
        else float("inf")
    )
    return SubspaceDiagnostics(
        status=status,
        rank=rank,
        dimension=dimension,
        singular_values=singular_values.copy(),
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        condition_number=condition,
        identifiable_basis=identifiable,
        unidentifiable_basis=unidentifiable,
    )


def _eigen_subspace_diagnostics(
    information: np.ndarray,
    *,
    dimension: int,
    relative_tolerance: float,
) -> SubspaceDiagnostics:
    matrix = 0.5 * (np.asarray(information, dtype=float) + np.asarray(information, dtype=float).T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if float(np.min(eigenvalues)) < -1e-8:
        raise RuntimeError("observability information matrix is not positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    leading = float(eigenvalues[0]) if len(eigenvalues) else 0.0
    absolute_tolerance = relative_tolerance * leading
    rank = int(np.sum(eigenvalues > absolute_tolerance)) if leading > 0 else 0
    identifiable = eigenvectors[:, :rank].copy()
    unidentifiable = eigenvectors[:, rank:].copy()
    singular_values = np.sqrt(eigenvalues)
    if rank == 0:
        status: RankStatus = "ZERO_INFORMATION"
    elif rank == dimension:
        status = "FULL_RANK"
    else:
        status = "PARTIAL"
    condition = (
        float(np.sqrt(eigenvalues[0] / eigenvalues[rank - 1]))
        if rank == dimension and dimension > 0
        else float("inf")
    )
    return SubspaceDiagnostics(
        status=status,
        rank=rank,
        dimension=dimension,
        singular_values=singular_values,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        condition_number=condition,
        identifiable_basis=identifiable,
        unidentifiable_basis=unidentifiable,
    )


def structural_observability_report(
    network: SixDomainNetwork,
    observation_model: LinearNetworkObservationModel,
    *,
    relative_tolerance: float = 1e-9,
) -> StructuralObservabilityReport:
    """Evaluate the classical LTI observability matrix [H; HA; ...; HA^5].

    The affine drift does not affect state observability. The result is
    conditional on the exact supplied A and H; it is not cohort-level proof.
    """
    tolerance = _validate_relative_tolerance(relative_tolerance)
    A = network.coupling_matrix
    power = np.eye(6)
    blocks = []
    for _ in range(6):
        blocks.append(observation_model.H @ power)
        power = power @ A
    matrix = np.vstack(blocks)
    return StructuralObservabilityReport(
        observability_matrix=matrix,
        subspace=_svd_diagnostics(
            matrix,
            dimension=6,
            relative_tolerance=tolerance,
        ),
    )


def finite_horizon_observability_report(
    time: np.ndarray,
    observation_model: LinearNetworkObservationModel,
    network: SixDomainNetwork,
    *,
    available_mask: np.ndarray | None = None,
    relative_tolerance: float = 1e-9,
) -> FiniteHorizonObservabilityReport:
    """Compute a schedule- and missingness-aware finite-horizon Gramian.

    For each visit i, let Phi_i map the initial state to x(t_i). The weighted
    Gramian is sum_i Phi_i.T H_i.T R_i^-1 H_i Phi_i. It is a deterministic
    transition/local measurement-noise diagnostic; process-noise-induced
    cross-visit covariance is not included in this v1 report.
    """
    tolerance = _validate_relative_tolerance(relative_tolerance)
    t = _validate_time(time)
    mask = _validate_mask(
        available_mask,
        n_time=len(t),
        n_features=observation_model.n_observed,
    )

    information = np.zeros((6, 6), dtype=float)
    phi_from_origin = np.eye(6)
    counts: list[int] = []

    for index in range(len(t)):
        if index > 0:
            transition = exact_network_transition(
                network,
                float(t[index] - t[index - 1]),
            )
            phi_from_origin = transition.phi @ phi_from_origin

        channels = np.flatnonzero(mask[index])
        counts.append(int(len(channels)))
        if len(channels) == 0:
            continue

        H_i = observation_model.H[channels] @ phi_from_origin
        R_i = observation_model.R[np.ix_(channels, channels)]
        weighted = np.linalg.solve(R_i, H_i)
        information += H_i.T @ weighted

    information = 0.5 * (information + information.T)
    return FiniteHorizonObservabilityReport(
        information_matrix=information,
        subspace=_eigen_subspace_diagnostics(
            information,
            dimension=6,
            relative_tolerance=tolerance,
        ),
        observation_counts_by_visit=tuple(counts),
        time_origin=float(t[0]),
    )


def _validate_positive_scales(
    values: np.ndarray | None,
    *,
    dimension: int,
    name: str,
    default: np.ndarray,
) -> np.ndarray:
    scales = np.asarray(default if values is None else values, dtype=float)
    if scales.shape != (dimension,):
        raise ValueError(f"{name} must have shape ({dimension},)")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError(f"{name} must contain finite positive values")
    return scales


def local_parameter_identifiability_report(
    time: np.ndarray,
    observation_model: LinearNetworkObservationModel,
    network: SixDomainNetwork,
    initial_state: np.ndarray,
    *,
    available_mask: np.ndarray | None = None,
    initial_state_scales: np.ndarray | None = None,
    parameter_scales: np.ndarray | None = None,
    relative_step: float = 1e-5,
    relative_tolerance: float = 1e-8,
) -> LocalParameterIdentifiabilityReport:
    """Assess local parameter sensitivity after profiling out unknown x(0).

    A central-difference Jacobian is formed from noise-whitened predicted
    observations. Initial-state sensitivities are treated as nuisance
    directions; the remaining parameter Jacobian is projected orthogonally
    away from their span before its numerical rank is assessed.

    This is a local, scale-dependent practical-identifiability diagnostic.
    It is not a proof of global or structural parameter identifiability.
    """
    tolerance = _validate_relative_tolerance(relative_tolerance)
    step_fraction = float(relative_step)
    if not np.isfinite(step_fraction) or not 0.0 < step_fraction < 1e-2:
        raise ValueError("relative_step must be finite and between 0 and 1e-2")

    t = _validate_time(time, minimum_points=2)
    mask = _validate_mask(
        available_mask,
        n_time=len(t),
        n_features=observation_model.n_observed,
    )
    if not np.any(mask):
        raise ValueError("at least one observed channel is required")

    x0 = np.asarray(initial_state, dtype=float)
    if x0.shape != (6,) or not np.all(np.isfinite(x0)):
        raise ValueError("initial_state must be a finite vector with shape (6,)")

    model_parameters = network.as_parameter_vector()
    default_x_scales = np.maximum(1.0, np.abs(x0))
    default_parameter_scales = np.maximum(1.0, np.abs(model_parameters))
    x_scales = _validate_positive_scales(
        initial_state_scales,
        dimension=6,
        name="initial_state_scales",
        default=default_x_scales,
    )
    theta_scales = _validate_positive_scales(
        parameter_scales,
        dimension=len(model_parameters),
        name="parameter_scales",
        default=default_parameter_scales,
    )

    z0 = np.concatenate([x0, model_parameters])
    scales = np.concatenate([x_scales, theta_scales])
    relative_times = t - t[0]

    def predicted_measurements(z: np.ndarray) -> np.ndarray:
        local_network = network.with_parameter_vector(z[6:])
        trajectory = simulate_network(local_network, z[:6], relative_times)
        flattened: list[float] = []
        for index in range(len(t)):
            channels = np.flatnonzero(mask[index])
            if len(channels) == 0:
                continue
            prediction = (
                observation_model.H[channels] @ trajectory.state[index]
                + observation_model.bias[channels]
            )
            covariance = observation_model.R[np.ix_(channels, channels)]
            chol = np.linalg.cholesky(covariance)
            flattened.extend(np.linalg.solve(chol, prediction).tolist())
        return np.asarray(flattened, dtype=float)

    n_outputs = int(np.sum(mask))
    jacobian = np.empty((n_outputs, len(z0)), dtype=float)
    for column in range(len(z0)):
        perturbation = step_fraction * scales[column]
        plus = z0.copy()
        minus = z0.copy()
        plus[column] += perturbation
        minus[column] -= perturbation
        jacobian[:, column] = (
            predicted_measurements(plus) - predicted_measurements(minus)
        ) / (2.0 * step_fraction)

    initial_jacobian = jacobian[:, :6]
    parameter_jacobian = jacobian[:, 6:]
    u, singular_x, _ = np.linalg.svd(initial_jacobian, full_matrices=False)
    leading_x = float(singular_x[0]) if len(singular_x) else 0.0
    tolerance_x = tolerance * leading_x
    initial_rank = (
        int(np.sum(singular_x > tolerance_x)) if leading_x > 0.0 else 0
    )
    nuisance_basis = u[:, :initial_rank]
    profiled = (
        parameter_jacobian - nuisance_basis @ (nuisance_basis.T @ parameter_jacobian)
        if initial_rank
        else parameter_jacobian.copy()
    )

    # Work in a dimensionless local coordinate system defined by the explicit
    # scale vectors above; SVD directions are combinations of scaled parameters.
    _, singular_values, vh = np.linalg.svd(profiled, full_matrices=True)
    leading = float(singular_values[0]) if len(singular_values) else 0.0
    absolute_tolerance = tolerance * leading
    rank = int(np.sum(singular_values > absolute_tolerance)) if leading > 0.0 else 0
    parameter_count = len(model_parameters)
    identifiable = vh[:rank, :].T.copy()
    unidentifiable = vh[rank:, :].T.copy()

    if rank == 0:
        status = "NO_LOCAL_INFORMATION"
    elif rank < parameter_count:
        status = "RANK_DEFICIENT"
    else:
        status = "LOCALLY_FULL_COLUMN_RANK"
    condition = (
        float(singular_values[0] / singular_values[parameter_count - 1])
        if rank == parameter_count and parameter_count > 0
        else float("inf")
    )
    return LocalParameterIdentifiabilityReport(
        parameter_names=network.parameter_names,
        status=status,
        rank=rank,
        parameter_count=parameter_count,
        singular_values=singular_values.copy(),
        relative_tolerance=tolerance,
        absolute_tolerance=absolute_tolerance,
        condition_number=condition,
        identifiable_parameter_combinations=identifiable,
        unidentifiable_parameter_combinations=unidentifiable,
        profiled_parameter_sensitivity=profiled,
        initial_state_rank=initial_rank,
        initial_state_scales=x_scales.copy(),
        parameter_scales=theta_scales.copy(),
        relative_step=step_fraction,
    )
