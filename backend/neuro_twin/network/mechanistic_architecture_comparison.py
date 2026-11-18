"""End-to-end comparison of competing Alzheimer mechanism architectures.

For each architecture, supplied parameter/initial-state draws are simulated,
mapped to common held-out measurement features with an explicit operator,
and scored on the same observations. Predictive preference is not causal proof.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neuro_twin.network.mechanistic_dynamics import MechanisticArchitectureModel
from neuro_twin.network.mechanisms import (
    AlzheimerMechanismCatalog,
    AlzheimerOriginArchitectureRegistry,
)
from neuro_twin.network.mechanism_observation import (
    MechanismObservationModel,
    summarize_mechanism_forecast_ensemble,
)
from neuro_twin.network.model_comparison import (
    ArchitectureComparisonReport,
    ArchitectureForecast,
    compare_architecture_forecasts,
)


@dataclass(frozen=True)
class ArchitectureCandidateRun:
    """All explicit inputs needed to produce forecasts for one origin hypothesis."""

    architecture_id: str
    parameter_draws: tuple[MechanisticArchitectureModel, ...]
    initial_activity_draws: tuple[np.ndarray, ...]
    observation_model: MechanismObservationModel


@dataclass(frozen=True)
class MechanisticArchitectureForecast:
    architecture_id: str
    predictive_mean: np.ndarray
    predictive_sd: np.ndarray
    feature_names: tuple[str, ...]
    feature_units: tuple[str, ...]
    operator_hash: str
    parameter_draw_count: int


@dataclass(frozen=True)
class MechanisticArchitectureComparison:
    report: ArchitectureComparisonReport
    forecasts: tuple[MechanisticArchitectureForecast, ...]
    time: np.ndarray
    evaluation_id: str


def _same_array(left: np.ndarray, right: np.ndarray) -> bool:
    return left.shape == right.shape and np.allclose(left, right, rtol=0.0, atol=1e-12)


def _validate_candidate(
    candidate: ArchitectureCandidateRun,
    *,
    catalog: AlzheimerMechanismCatalog,
    registry: AlzheimerOriginArchitectureRegistry,
    time: np.ndarray,
) -> None:
    if not candidate.architecture_id.strip():
        raise ValueError("architecture_id must be non-empty")
    if not candidate.parameter_draws:
        raise ValueError(f"{candidate.architecture_id}: at least one parameter draw is required")
    if len(candidate.parameter_draws) != len(candidate.initial_activity_draws):
        raise ValueError(
            f"{candidate.architecture_id}: one initial state is required per parameter draw"
        )
    if candidate.observation_model.feature_names == ():
        raise ValueError("observation model must declare evaluated features")

    first = candidate.parameter_draws[0]
    if first.architecture_id != candidate.architecture_id:
        raise ValueError("candidate architecture_id does not match its mechanism model")
    first.validate_against(catalog, registry)
    if candidate.observation_model.mechanism_node_ids != first.node_ids:
        raise ValueError("observation operator node order must match candidate mechanism model")

    for index, (model, initial) in enumerate(
        zip(candidate.parameter_draws, candidate.initial_activity_draws)
    ):
        model.validate_against(catalog, registry)
        if model.architecture_id != candidate.architecture_id:
            raise ValueError("all parameter draws must belong to the candidate architecture")
        if model.node_ids != first.node_ids or model.edge_ids != first.edge_ids:
            raise ValueError("parameter draws must preserve node and edge topology/order")
        if model.time_unit != first.time_unit:
            raise ValueError("parameter draws must use the same time unit")
        if model.damage_model.driver_names != first.damage_model.driver_names:
            raise ValueError("parameter draws must preserve damage-driver mappings")
        if model.damage_model.repair_names != first.damage_model.repair_names:
            raise ValueError("parameter draws must preserve repair mappings")
        initial_values = np.asarray(initial, dtype=float)
        if initial_values.shape != (len(model.nodes),):
            raise ValueError(
                f"{candidate.architecture_id}: initial state {index} has incorrect shape"
            )
        if not np.all(np.isfinite(initial_values)) or np.any(
            (initial_values < 0.0) | (initial_values > 1.0)
        ):
            raise ValueError("initial mechanism activities must lie in [0, 1]")


def compare_mechanistic_architectures(
    *,
    evaluation_id: str,
    holdout_kind: str,
    catalog: AlzheimerMechanismCatalog,
    registry: AlzheimerOriginArchitectureRegistry,
    time: np.ndarray,
    observed: np.ndarray,
    measurement_sd: np.ndarray,
    available_mask: np.ndarray,
    candidates: tuple[ArchitectureCandidateRun, ...],
) -> MechanisticArchitectureComparison:
    """Run competing architecture ensembles and score them on common held-out data.

    Different architectures may have different mechanism nodes and graph
    topology. They must nevertheless predict the same ordered observed
    features, in the same transformed units, with the same data provenance,
    transformation version, measurement covariance, evaluation time grid and
    available-data mask.

    Each candidate supplies one observation operator and paired parameter /
    initial-state draws. The operator H may differ by architecture, while the
    feature names, units, R, source-catalog hash and transformation identity
    must be common across all candidates.
    """
    registry.validate_against(catalog)
    t = np.asarray(time, dtype=float)
    if t.ndim != 1 or len(t) < 2 or not np.all(np.isfinite(t)):
        raise ValueError("time must contain at least two finite points")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("time must be strictly increasing")
    if len(candidates) < 2:
        raise ValueError("at least two competing architecture candidates are required")
    ids = tuple(item.architecture_id for item in candidates)
    if len(ids) != len(set(ids)):
        raise ValueError("architecture candidates must have unique IDs")

    for candidate in candidates:
        _validate_candidate(
            candidate, catalog=catalog, registry=registry, time=t
        )

    reference = candidates[0].observation_model
    for candidate in candidates[1:]:
        operator = candidate.observation_model
        if operator.feature_names != reference.feature_names:
            raise ValueError("all architectures must predict the same ordered features")
        if operator.feature_units != reference.feature_units:
            raise ValueError("all architectures must predict features in the same units")
        if operator.source_catalog_hash != reference.source_catalog_hash:
            raise ValueError("all architectures must use the same source catalog hash")
        if (
            operator.transformation_id != reference.transformation_id
            or operator.transformation_version != reference.transformation_version
        ):
            raise ValueError("all architectures must use the same feature transformation")
        if not _same_array(operator.R, reference.R):
            raise ValueError("all architectures must use the same measurement covariance")

    forecast_records: list[MechanisticArchitectureForecast] = []
    scoring_forecasts: list[ArchitectureForecast] = []
    for candidate in candidates:
        trajectories = tuple(
            model.simulate(
                catalog=catalog,
                registry=registry,
                time=t,
                initial_activity=np.asarray(initial, dtype=float),
            )
            for model, initial in zip(
                candidate.parameter_draws, candidate.initial_activity_draws
            )
        )
        ensemble = summarize_mechanism_forecast_ensemble(
            architecture_id=candidate.architecture_id,
            trajectories=trajectories,
            observation_model=candidate.observation_model,
        )
        record = MechanisticArchitectureForecast(
            architecture_id=ensemble.architecture_id,
            predictive_mean=ensemble.predictive_mean,
            predictive_sd=ensemble.predictive_sd,
            feature_names=ensemble.feature_names,
            feature_units=candidate.observation_model.feature_units,
            operator_hash=ensemble.operator_hash,
            parameter_draw_count=ensemble.parameter_draw_count,
        )
        forecast_records.append(record)
        scoring_forecasts.append(
            ArchitectureForecast(
                architecture_id=record.architecture_id,
                predictive_mean=record.predictive_mean,
                predictive_sd=record.predictive_sd,
            )
        )

    report = compare_architecture_forecasts(
        evaluation_id=evaluation_id,
        holdout_kind=holdout_kind,
        observed=observed,
        measurement_sd=measurement_sd,
        available_mask=available_mask,
        forecasts=tuple(scoring_forecasts),
    )
    return MechanisticArchitectureComparison(
        report=report,
        forecasts=tuple(forecast_records),
        time=t.copy(),
        evaluation_id=evaluation_id,
    )
