"""Shared-parameter calibration across multiple longitudinal Alzheimer series.

This is a weighted nonlinear least-squares cohort prototype with shared
mechanism kinetics and optional subject-specific initial latent states. It is
not a hierarchical Bayesian model, a causal estimator, or a substitute for
external validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import least_squares

from neuro_twin.network.mechanistic_calibration import (
    ParameterBound,
    _apply_parameters,
    _get_parameter,
    _validate_parameter_domain,
)
from neuro_twin.network.mechanistic_dynamics import MechanisticArchitectureModel
from neuro_twin.network.mechanisms import (
    AlzheimerMechanismCatalog,
    AlzheimerOriginArchitectureRegistry,
)
from neuro_twin.network.mechanism_observation import MechanismObservationModel


CohortFitStatus = Literal[
    "COHORT_LOCALLY_FULL_RANK_CANDIDATE_FIT",
    "COHORT_RANK_DEFICIENT_CANDIDATE_FIT",
]


@dataclass(frozen=True)
class MechanisticSubjectSeries:
    """One pseudonymized subject's irregular longitudinal observation series."""

    subject_id: str
    time: np.ndarray
    observed: np.ndarray
    available_mask: np.ndarray
    initial_activity: np.ndarray


@dataclass(frozen=True)
class SubjectInitialStateBound:
    """Optional subject-specific fit interval for one initial mechanism state."""

    subject_id: str
    node_id: str
    initial: float
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not self.subject_id.strip() or not self.node_id.strip():
            raise ValueError("subject_id and node_id must be non-empty")
        values = np.asarray(
            [self.initial, self.lower, self.upper], dtype=float
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("subject initial-state values and bounds must be finite")
        if self.lower < 0.0 or self.upper > 1.0 or self.lower >= self.upper:
            raise ValueError("subject initial-state bounds must satisfy 0 <= lower < upper <= 1")
        if self.initial < self.lower or self.initial > self.upper:
            raise ValueError("subject initial state must lie within its bounds")


@dataclass(frozen=True)
class CohortSubjectPrediction:
    subject_id: str
    time: np.ndarray
    predictive_mean: np.ndarray
    initial_activity: np.ndarray
    observation_mask: np.ndarray


@dataclass(frozen=True)
class MechanisticCohortCalibrationResult:
    architecture_id: str
    fit_status: CohortFitStatus
    parameter_names: tuple[str, ...]
    parameter_estimates: np.ndarray
    parameter_standard_errors: np.ndarray
    parameter_covariance: np.ndarray
    singular_values: np.ndarray
    identifiable_rank: int
    parameter_count: int
    condition_number: float
    weighted_sum_squares: float
    residual_degrees_of_freedom: int
    residual_rms: float
    fitted_model: MechanisticArchitectureModel
    subject_predictions: tuple[CohortSubjectPrediction, ...]
    subject_count: int
    observation_count: int
    interpretation: Literal[
        "shared_kinetic_cohort_fit_not_causal_or_external_validation"
    ] = "shared_kinetic_cohort_fit_not_causal_or_external_validation"


def _validate_subject(
    subject: MechanisticSubjectSeries,
    *,
    feature_count: int,
    node_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not subject.subject_id.strip():
        raise ValueError("subject_id must be non-empty")
    time = np.asarray(subject.time, dtype=float)
    observed = np.asarray(subject.observed, dtype=float)
    mask = np.asarray(subject.available_mask)
    initial = np.asarray(subject.initial_activity, dtype=float)
    if time.ndim != 1 or len(time) < 2 or not np.all(np.isfinite(time)):
        raise ValueError(f"{subject.subject_id}: time must contain at least two finite points")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError(f"{subject.subject_id}: time must be strictly increasing")
    if observed.shape != (len(time), feature_count):
        raise ValueError(f"{subject.subject_id}: observed has incorrect shape")
    if mask.shape != observed.shape or mask.dtype != np.bool_:
        raise ValueError(f"{subject.subject_id}: available_mask must be boolean and match observed")
    if not np.any(mask):
        raise ValueError(f"{subject.subject_id}: at least one observation is required")
    if not np.all(np.isfinite(observed[mask])):
        raise ValueError(f"{subject.subject_id}: available observed values must be finite")
    if initial.shape != (node_count,) or not np.all(np.isfinite(initial)):
        raise ValueError(f"{subject.subject_id}: initial_activity has incorrect shape or values")
    if np.any((initial < 0.0) | (initial > 1.0)):
        raise ValueError(f"{subject.subject_id}: initial_activity must lie in [0, 1]")
    return time, observed, mask, initial


def fit_mechanistic_cohort_parameters(
    model: MechanisticArchitectureModel,
    *,
    catalog: AlzheimerMechanismCatalog,
    registry: AlzheimerOriginArchitectureRegistry,
    observation_model: MechanismObservationModel,
    subjects: tuple[MechanisticSubjectSeries, ...],
    shared_parameter_bounds: tuple[ParameterBound, ...],
    subject_initial_state_bounds: tuple[SubjectInitialStateBound, ...] = (),
    relative_rank_tolerance: float = 1e-8,
    max_nfev: int = 500,
) -> MechanisticCohortCalibrationResult:
    """Fit shared kinetic parameters over multiple subjects with individual starts.

    The same graph, node/edge structure and kinetic parameters are shared across
    subjects. Initial mechanism activities may be fixed per subject or selected
    subject/node coordinates may be estimated within explicit [0,1] bounds.

    Observation residuals are whitened by the shared versioned covariance R at
    each visit, using only available channels. Subject series may have different
    irregular visit schedules. All parameter bounds, graph signs and topology
    remain explicit; this routine does not infer causal edge direction.
    """
    model.validate_against(catalog, registry)
    if observation_model.mechanism_node_ids != model.node_ids:
        raise ValueError("observation model node order must match the mechanism model")
    if len(subjects) < 2:
        raise ValueError("cohort calibration requires at least two subjects")
    subject_ids = tuple(item.subject_id for item in subjects)
    if any(not value.strip() for value in subject_ids) or len(set(subject_ids)) != len(subject_ids):
        raise ValueError("subject identifiers must be unique and non-empty")
    if not shared_parameter_bounds and not subject_initial_state_bounds:
        raise ValueError("at least one shared or subject-specific parameter must be fitted")
    if int(max_nfev) < 1:
        raise ValueError("max_nfev must be >= 1")
    rank_tolerance = float(relative_rank_tolerance)
    if not np.isfinite(rank_tolerance) or not 0.0 < rank_tolerance < 1.0:
        raise ValueError("relative_rank_tolerance must lie in (0, 1)")

    validated = tuple(
        _validate_subject(
            subject,
            feature_count=len(observation_model.feature_names),
            node_count=len(model.nodes),
        )
        for subject in subjects
    )
    validated_by_id = {
        subject.subject_id: data for subject, data in zip(subjects, validated)
    }
    observation_count = sum(int(np.sum(item[2])) for item in validated)

    shared_names = tuple(item.name for item in shared_parameter_bounds)
    if len(shared_names) != len(set(shared_names)):
        raise ValueError("shared parameter names must be unique")
    for item in shared_parameter_bounds:
        if item.name.startswith("initial:"):
            raise ValueError(
                "cohort shared parameters cannot include initial states; "
                "use subject_initial_state_bounds"
            )
        _get_parameter(item.name, model, validated[0][3])
        _validate_parameter_domain(item.name, item.lower, item.upper)

    local_keys = tuple(
        (item.subject_id, item.node_id) for item in subject_initial_state_bounds
    )
    if len(local_keys) != len(set(local_keys)):
        raise ValueError("subject initial-state bounds must be unique per subject and node")
    subject_index = {value: index for index, value in enumerate(subject_ids)}
    node_index = {value: index for index, value in enumerate(model.node_ids)}
    for bound in subject_initial_state_bounds:
        if bound.subject_id not in subject_index:
            raise ValueError(f"unknown subject in initial-state bound: {bound.subject_id}")
        if bound.node_id not in node_index:
            raise ValueError(f"unknown mechanism node in initial-state bound: {bound.node_id}")

    parameter_names = shared_names + tuple(
        f"subject:{item.subject_id}:initial:{item.node_id}"
        for item in subject_initial_state_bounds
    )
    if len(parameter_names) != len(set(parameter_names)):
        raise ValueError("combined parameter names must be unique")

    shared_lower = np.asarray([item.lower for item in shared_parameter_bounds], dtype=float)
    shared_upper = np.asarray([item.upper for item in shared_parameter_bounds], dtype=float)
    shared_start = np.asarray([item.initial for item in shared_parameter_bounds], dtype=float)
    local_lower = np.asarray([item.lower for item in subject_initial_state_bounds], dtype=float)
    local_upper = np.asarray([item.upper for item in subject_initial_state_bounds], dtype=float)
    local_start = np.asarray([item.initial for item in subject_initial_state_bounds], dtype=float)
    lower = np.concatenate([shared_lower, local_lower])
    upper = np.concatenate([shared_upper, local_upper])
    start = np.concatenate([shared_start, local_start])
    scales = upper - lower

    residual_rows: dict[str, tuple[tuple[int, np.ndarray, np.ndarray], ...]] = {}
    for subject, (time, observed, mask, _initial) in zip(subjects, validated):
        rows = []
        for row in range(len(time)):
            channels = np.flatnonzero(mask[row])
            if len(channels) == 0:
                continue
            covariance = observation_model.R[np.ix_(channels, channels)]
            chol = np.linalg.cholesky(covariance)
            rows.append((row, channels, chol))
        residual_rows[subject.subject_id] = tuple(rows)

    def calculate(values: np.ndarray):
        global_values = values[:len(shared_names)]
        candidate_model, _ = _apply_parameters(
            model,
            validated[0][3],
            shared_names,
            global_values,
        )
        subject_initials = {
            subject.subject_id: np.array(data[3], dtype=float, copy=True)
            for subject, data in zip(subjects, validated)
        }
        local_values = values[len(shared_names):]
        for bound, fitted_value in zip(subject_initial_state_bounds, local_values):
            subject_initials[bound.subject_id][node_index[bound.node_id]] = float(fitted_value)

        predictions: list[CohortSubjectPrediction] = []
        residual_chunks: list[np.ndarray] = []
        for subject in subjects:
            time, observed, mask, _initial = validated_by_id[subject.subject_id]
            trajectory = candidate_model.simulate(
                catalog=catalog,
                registry=registry,
                time=time,
                initial_activity=subject_initials[subject.subject_id],
            )
            predicted = observation_model.predict(trajectory).predictive_mean
            for row, channels, chol in residual_rows[subject.subject_id]:
                residual = predicted[row, channels] - observed[row, channels]
                residual_chunks.append(np.linalg.solve(chol, residual))
            predictions.append(
                CohortSubjectPrediction(
                    subject_id=subject.subject_id,
                    time=time.copy(),
                    predictive_mean=predicted,
                    initial_activity=subject_initials[subject.subject_id].copy(),
                    observation_mask=mask.copy(),
                )
            )
        return candidate_model, tuple(predictions), np.concatenate(residual_chunks)

    def residual_function(values: np.ndarray) -> np.ndarray:
        return calculate(values)[2]

    optimum = least_squares(
        residual_function,
        start,
        bounds=(lower, upper),
        method="trf",
        x_scale=scales,
        max_nfev=int(max_nfev),
    )
    fitted_model, fitted_predictions, residuals = calculate(optimum.x)
    jacobian = np.asarray(optimum.jac, dtype=float)
    scaled_jacobian = jacobian * scales[np.newaxis, :]
    singular_values = np.linalg.svd(scaled_jacobian, compute_uv=False)
    leading = float(singular_values[0]) if len(singular_values) else 0.0
    rank = int(np.sum(singular_values > rank_tolerance * leading)) if leading > 0.0 else 0
    parameter_count = len(parameter_names)
    status: CohortFitStatus = (
        "COHORT_LOCALLY_FULL_RANK_CANDIDATE_FIT"
        if rank == parameter_count
        else "COHORT_RANK_DEFICIENT_CANDIDATE_FIT"
    )
    dof = max(1, len(residuals) - parameter_count)
    weighted_sse = float(np.dot(residuals, residuals))
    residual_variance = weighted_sse / dof
    covariance = (
        np.linalg.pinv(jacobian.T @ jacobian, rcond=rank_tolerance)
        * residual_variance
    )
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    condition = (
        float(singular_values[0] / singular_values[rank - 1])
        if rank == parameter_count and parameter_count > 0
        else float("inf")
    )
    return MechanisticCohortCalibrationResult(
        architecture_id=model.architecture_id,
        fit_status=status,
        parameter_names=parameter_names,
        parameter_estimates=optimum.x.copy(),
        parameter_standard_errors=standard_errors,
        parameter_covariance=covariance,
        singular_values=singular_values,
        identifiable_rank=rank,
        parameter_count=parameter_count,
        condition_number=condition,
        weighted_sum_squares=weighted_sse,
        residual_degrees_of_freedom=dof,
        residual_rms=float(np.sqrt(np.mean(residuals ** 2))),
        fitted_model=fitted_model,
        subject_predictions=fitted_predictions,
        subject_count=len(subjects),
        observation_count=observation_count,
    )
