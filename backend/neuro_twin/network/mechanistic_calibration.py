"""Constrained candidate calibration for mechanism-level Alzheimer models.

This is weighted nonlinear least squares, not Bayesian causal discovery.
Only explicitly selected parameters are fitted. Results remain candidate
estimates; they are not validated biological coefficients.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import least_squares

from neuro_twin.network.mechanistic_dynamics import (
    EdgeKinetics,
    MechanisticArchitectureModel,
    NodeKinetics,
)
from neuro_twin.network.mechanisms import (
    AlzheimerMechanismCatalog,
    AlzheimerOriginArchitectureRegistry,
)
from neuro_twin.network.mechanism_observation import MechanismObservationModel


FitStatus = Literal["LOCALLY_FULL_RANK_CANDIDATE_FIT", "RANK_DEFICIENT_CANDIDATE_FIT"]


@dataclass(frozen=True)
class ParameterBound:
    """Caller-supplied fit interval and starting value for one model parameter."""

    name: str
    initial: float
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("parameter name must be non-empty")
        values = np.asarray([self.initial, self.lower, self.upper], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("parameter fit values and bounds must be finite")
        if self.lower >= self.upper:
            raise ValueError("parameter lower bound must be less than upper bound")
        if self.initial < self.lower or self.initial > self.upper:
            raise ValueError("parameter initial value must lie within its bounds")


@dataclass(frozen=True)
class MechanisticCalibrationResult:
    architecture_id: str
    fit_status: FitStatus
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
    fitted_initial_activity: np.ndarray
    fitted_model: MechanisticArchitectureModel
    fitted_predictive_mean: np.ndarray
    observation_mask: np.ndarray
    interpretation: Literal["candidate_parameter_fit_not_biological_validation"] = (
        "candidate_parameter_fit_not_biological_validation"
    )


def _validate_time(time: np.ndarray) -> np.ndarray:
    values = np.asarray(time, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("time must be a finite 1-D array with at least two points")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("time must be strictly increasing")
    return values


def _get_parameter(
    name: str,
    model: MechanisticArchitectureModel,
    initial_activity: np.ndarray,
) -> float:
    parts = name.split(":")
    if len(parts) == 3 and parts[0] == "node":
        _, node_id, field = parts
        for node in model.nodes:
            if node.node_id == node_id and field in ("baseline_activity", "relaxation_rate"):
                return float(getattr(node, field))
    elif len(parts) == 3 and parts[0] == "edge":
        _, edge_id, field = parts
        if field == "coefficient":
            for edge in model.edges:
                if edge.edge_id == edge_id:
                    return float(edge.coefficient)
    elif len(parts) == 2 and parts[0] == "initial":
        try:
            return float(initial_activity[model.node_ids.index(parts[1])])
        except ValueError:
            pass
    raise ValueError(f"unsupported or unknown fit parameter name: {name}")


def _validate_parameter_domain(name: str, lower: float, upper: float) -> None:
    parts = name.split(":")
    if parts[0] == "node" and len(parts) == 3 and parts[2] == "baseline_activity":
        if lower < 0.0 or upper > 1.0:
            raise ValueError("baseline_activity bounds must lie within [0, 1]")
    elif parts[0] == "node" and len(parts) == 3 and parts[2] == "relaxation_rate":
        if lower <= 0.0:
            raise ValueError("relaxation_rate lower bound must be > 0")
    elif parts[0] == "edge" and len(parts) == 3 and parts[2] == "coefficient":
        if lower < 0.0:
            raise ValueError("edge coefficient bounds must be non-negative")
    elif parts[0] == "initial" and len(parts) == 2:
        if lower < 0.0 or upper > 1.0:
            raise ValueError("initial-state bounds must lie within [0, 1]")
    else:
        raise ValueError(f"unsupported fit parameter name: {name}")


def _apply_parameters(
    model: MechanisticArchitectureModel,
    initial_activity: np.ndarray,
    names: tuple[str, ...],
    values: np.ndarray,
) -> tuple[MechanisticArchitectureModel, np.ndarray]:
    node_values = {
        item.node_id: {
            "baseline_activity": item.baseline_activity,
            "relaxation_rate": item.relaxation_rate,
        }
        for item in model.nodes
    }
    edge_values = {item.edge_id: item.coefficient for item in model.edges}
    initial = np.array(initial_activity, dtype=float, copy=True)

    for name, value in zip(names, values):
        parts = name.split(":")
        if parts[0] == "node":
            node_values[parts[1]][parts[2]] = float(value)
        elif parts[0] == "edge":
            edge_values[parts[1]] = float(value)
        elif parts[0] == "initial":
            initial[model.node_ids.index(parts[1])] = float(value)

    nodes = tuple(
        NodeKinetics(
            node_id=item.node_id,
            baseline_activity=node_values[item.node_id]["baseline_activity"],
            relaxation_rate=node_values[item.node_id]["relaxation_rate"],
        )
        for item in model.nodes
    )
    edges = tuple(
        EdgeKinetics(
            edge_id=item.edge_id,
            coefficient=edge_values[item.edge_id],
            sign=item.sign,
            hill_half_saturation=item.hill_half_saturation,
            hill_cooperativity=item.hill_cooperativity,
        )
        for item in model.edges
    )
    updated = MechanisticArchitectureModel(
        architecture_id=model.architecture_id,
        time_unit=model.time_unit,
        nodes=nodes,
        edges=edges,
        damage_model=model.damage_model,
        version=model.version,
        parameter_status="hypothesis",
    )
    return updated, initial


def fit_mechanistic_parameters(
    model: MechanisticArchitectureModel,
    *,
    catalog: AlzheimerMechanismCatalog,
    registry: AlzheimerOriginArchitectureRegistry,
    observation_model: MechanismObservationModel,
    time: np.ndarray,
    observed: np.ndarray,
    available_mask: np.ndarray,
    initial_activity: np.ndarray,
    parameter_bounds: tuple[ParameterBound, ...],
    relative_rank_tolerance: float = 1e-8,
    max_nfev: int = 300,
) -> MechanisticCalibrationResult:
    """Fit only caller-selected parameters to one longitudinal observation set.

    Measurement residuals are whitened by the supplied observation covariance
    R, respecting missing channels. The model, graph, observation operator,
    initial state and fit bounds are all explicit. No edge is added, no sign is
    learned, and no unselected parameter is changed.

    The local rank test is scale-aware using the declared fit interval widths.
    A rank-deficient fit is returned with that status, never promoted silently
    to a validated or causal parameter set.
    """
    model.validate_against(catalog, registry)
    if observation_model.mechanism_node_ids != model.node_ids:
        raise ValueError("observation model node order must match the mechanism model")
    t = _validate_time(time)
    y = np.asarray(observed, dtype=float)
    mask = np.asarray(available_mask)
    x0 = np.asarray(initial_activity, dtype=float)
    if y.shape != (len(t), len(observation_model.feature_names)):
        raise ValueError("observed must have shape (n_time, n_features)")
    if mask.shape != y.shape or mask.dtype != np.bool_:
        raise ValueError("available_mask must be boolean and match observed shape")
    if not np.any(mask):
        raise ValueError("at least one observation must be available")
    if not np.all(np.isfinite(y[mask])):
        raise ValueError("available observed values must be finite")
    if x0.shape != (len(model.nodes),) or not np.all(np.isfinite(x0)):
        raise ValueError("initial_activity must be finite and match model nodes")
    if np.any((x0 < 0.0) | (x0 > 1.0)):
        raise ValueError("initial_activity must lie in [0, 1]")
    if not parameter_bounds:
        raise ValueError("at least one explicit parameter bound is required")
    if int(max_nfev) < 1:
        raise ValueError("max_nfev must be >= 1")
    rank_tolerance = float(relative_rank_tolerance)
    if not np.isfinite(rank_tolerance) or not 0.0 < rank_tolerance < 1.0:
        raise ValueError("relative_rank_tolerance must lie in (0, 1)")

    names = tuple(item.name for item in parameter_bounds)
    if len(set(names)) != len(names):
        raise ValueError("parameter bounds must have unique parameter names")
    lower = np.asarray([item.lower for item in parameter_bounds], dtype=float)
    upper = np.asarray([item.upper for item in parameter_bounds], dtype=float)
    start = np.asarray([item.initial for item in parameter_bounds], dtype=float)
    for item in parameter_bounds:
        _get_parameter(item.name, model, x0)
        _validate_parameter_domain(item.name, item.lower, item.upper)
    scales = upper - lower

    residual_rows = []
    for row in range(len(t)):
        channels = np.flatnonzero(mask[row])
        if not len(channels):
            continue
        covariance = observation_model.R[np.ix_(channels, channels)]
        chol = np.linalg.cholesky(covariance)
        residual_rows.append((row, channels, chol))

    def calculate(values: np.ndarray):
        candidate_model, candidate_initial = _apply_parameters(
            model, x0, names, values
        )
        trajectory = candidate_model.simulate(
            catalog=catalog,
            registry=registry,
            time=t,
            initial_activity=candidate_initial,
        )
        prediction = observation_model.predict(trajectory).predictive_mean
        residual_chunks = []
        for row, channels, chol in residual_rows:
            residual = prediction[row, channels] - y[row, channels]
            residual_chunks.append(np.linalg.solve(chol, residual))
        return candidate_model, candidate_initial, prediction, np.concatenate(residual_chunks)

    def residual_function(values: np.ndarray) -> np.ndarray:
        return calculate(values)[3]

    optimum = least_squares(
        residual_function,
        start,
        bounds=(lower, upper),
        method="trf",
        x_scale=scales,
        max_nfev=int(max_nfev),
    )
    fitted_model, fitted_initial, prediction, residuals = calculate(optimum.x)
    jacobian = np.asarray(optimum.jac, dtype=float)
    scaled_jacobian = jacobian * scales[np.newaxis, :]
    singular_values = np.linalg.svd(scaled_jacobian, compute_uv=False)
    leading = float(singular_values[0]) if len(singular_values) else 0.0
    rank = int(np.sum(singular_values > rank_tolerance * leading)) if leading > 0.0 else 0
    parameter_count = len(names)
    status: FitStatus = (
        "LOCALLY_FULL_RANK_CANDIDATE_FIT"
        if rank == parameter_count
        else "RANK_DEFICIENT_CANDIDATE_FIT"
    )
    dof = max(1, len(residuals) - parameter_count)
    weighted_sse = float(np.dot(residuals, residuals))
    residual_variance = weighted_sse / dof
    covariance = np.linalg.pinv(jacobian.T @ jacobian, rcond=rank_tolerance) * residual_variance
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    condition = (
        float(singular_values[0] / singular_values[rank - 1])
        if rank == parameter_count and parameter_count > 0
        else float("inf")
    )
    return MechanisticCalibrationResult(
        architecture_id=model.architecture_id,
        fit_status=status,
        parameter_names=names,
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
        fitted_initial_activity=fitted_initial,
        fitted_model=fitted_model,
        fitted_predictive_mean=prediction,
        observation_mask=mask.copy(),
    )
