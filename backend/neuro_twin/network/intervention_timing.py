"""Timing sweeps for virtual interventions against persistent damage.

This module evaluates when a declared model perturbation changes the damage
trajectory most strongly. It does not infer clinical treatment windows or
pharmacological dose/time relationships.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from neuro_twin.network.damage import DAMAGE_COMPONENTS
from neuro_twin.network.mechanistic_dynamics import (
    InterventionMode,
    InterventionProtocol,
    MechanisticArchitectureModel,
    compare_intervention,
)
from neuro_twin.network.mechanisms import (
    AlzheimerMechanismCatalog,
    AlzheimerOriginArchitectureRegistry,
)


@dataclass(frozen=True)
class InterventionTimingWindow:
    start_time: float
    end_time: float


@dataclass(frozen=True)
class InterventionTimingEffect:
    window: InterventionTimingWindow
    damage_prevented_median: np.ndarray
    damage_prevented_interval: np.ndarray
    fraction_parameter_draws_beneficial: np.ndarray
    fraction_parameter_draws_harmful: np.ndarray
    damage_burden_auc_prevented_median: np.ndarray
    damage_burden_auc_prevented_interval: np.ndarray
    fraction_parameter_draws_auc_beneficial: np.ndarray


@dataclass(frozen=True)
class InterventionTimingSweepResult:
    architecture_id: str
    target_node: str
    mode: InterventionMode
    intensity: float
    time_unit: str
    parameter_draw_count: int
    component_names: tuple[str, ...]
    effects: tuple[InterventionTimingEffect, ...]
    interpretation: Literal[
        "model_conditional_timing_sweep_not_empirical_treatment_window"
    ] = "model_conditional_timing_sweep_not_empirical_treatment_window"


def sweep_intervention_timing(
    models: tuple[MechanisticArchitectureModel, ...],
    *,
    catalog: AlzheimerMechanismCatalog,
    registry: AlzheimerOriginArchitectureRegistry,
    time: np.ndarray,
    initial_activity_draws: tuple[np.ndarray, ...],
    initial_damage: np.ndarray,
    target_node: str,
    mode: InterventionMode,
    intensity: float,
    windows: tuple[InterventionTimingWindow, ...],
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> InterventionTimingSweepResult:
    """Compare a fixed perturbation over caller-selected start/end windows.

    Each model is one caller-supplied parameter draw for the same architecture.
    Every window is evaluated against a paired baseline with the same model,
    initial mechanism state and initial damage. Positive prevented-damage
    values mean less accumulated damage in the perturbed trajectory.

    The reported fraction of parameter draws with benefit is a descriptive
    fraction over supplied draws; it is not a posterior probability unless
    the draws are independently justified as posterior samples.
    """
    if not models:
        raise ValueError("at least one parameterized mechanism model is required")
    if len(models) != len(initial_activity_draws):
        raise ValueError("one initial mechanism state is required per model draw")
    if not windows:
        raise ValueError("at least one intervention timing window is required")
    if len({(item.start_time, item.end_time) for item in windows}) != len(windows):
        raise ValueError("intervention timing windows must be unique")

    t = np.asarray(time, dtype=float)
    if t.ndim != 1 or len(t) < 2 or not np.all(np.isfinite(t)):
        raise ValueError("time must be a finite 1-D array with at least two points")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("time must be strictly increasing")
    lower = float(lower_quantile)
    upper = float(upper_quantile)
    if not np.isfinite(lower) or not np.isfinite(upper) or not 0.0 <= lower < upper <= 1.0:
        raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")

    first = models[0]
    registry.validate_against(catalog)
    first.validate_against(catalog, registry)
    architecture_id = first.architecture_id
    node_ids = first.node_ids
    edge_ids = first.edge_ids
    driver_names = first.damage_model.driver_names
    repair_names = first.damage_model.repair_names
    if target_node not in node_ids:
        raise ValueError(f"intervention target {target_node} is not modeled")
    if mode not in ("clamp", "additive_drive"):
        raise ValueError("unsupported intervention mode")
    for model, initial in zip(models, initial_activity_draws):
        model.validate_against(catalog, registry)
        if model.architecture_id != architecture_id:
            raise ValueError("all parameter draws must belong to one architecture")
        if model.node_ids != node_ids or model.edge_ids != edge_ids:
            raise ValueError("parameter draws must preserve node and edge topology/order")
        if model.time_unit != first.time_unit:
            raise ValueError("parameter draws must share a time unit")
        if model.damage_model.driver_names != driver_names:
            raise ValueError("parameter draws must preserve damage drivers")
        if model.damage_model.repair_names != repair_names:
            raise ValueError("parameter draws must preserve repair mappings")
        initial = np.asarray(initial, dtype=float)
        if initial.shape != (len(node_ids),):
            raise ValueError("initial mechanism state has incorrect shape")
        if not np.all(np.isfinite(initial)) or np.any((initial < 0.0) | (initial > 1.0)):
            raise ValueError("initial mechanism activities must lie in [0, 1]")

    initial_damage_values = np.asarray(initial_damage, dtype=float)
    if initial_damage_values.shape != (len(DAMAGE_COMPONENTS),):
        raise ValueError("initial_damage has incorrect shape")
    if not np.all(np.isfinite(initial_damage_values)) or np.any(initial_damage_values < 0.0):
        raise ValueError("initial_damage must be finite and non-negative")

    effects: list[InterventionTimingEffect] = []
    for index, window in enumerate(windows):
        start = float(window.start_time)
        end = float(window.end_time)
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            raise ValueError("each timing window must have finite end_time > start_time")
        if start < t[0] or end > t[-1]:
            raise ValueError("each intervention window must lie within simulation time")

        final_effects = []
        auc_effects = []
        for draw_index, (model, initial) in enumerate(zip(models, initial_activity_draws)):
            protocol = InterventionProtocol(
                intervention_id=f"timing_sweep_{index}_draw_{draw_index}",
                target_node=target_node,
                mode=mode,
                start_time=start,
                end_time=end,
                intensity=float(intensity),
            )
            comparison = compare_intervention(
                model,
                catalog=catalog,
                registry=registry,
                time=t,
                initial_activity=np.asarray(initial, dtype=float),
                initial_damage=initial_damage_values,
                interventions=(protocol,),
            )
            final_effects.append(comparison.effect.damage_prevented_at_final)
            auc_effects.append(comparison.effect.damage_burden_auc_prevented)

        final_array = np.stack(final_effects, axis=0)
        auc_array = np.stack(auc_effects, axis=0)
        effects.append(
            InterventionTimingEffect(
                window=window,
                damage_prevented_median=np.quantile(final_array, 0.5, axis=0),
                damage_prevented_interval=np.quantile(
                    final_array, [lower, upper], axis=0
                ),
                fraction_parameter_draws_beneficial=np.mean(final_array > 0.0, axis=0),
                fraction_parameter_draws_harmful=np.mean(final_array < 0.0, axis=0),
                damage_burden_auc_prevented_median=np.quantile(
                    auc_array, 0.5, axis=0
                ),
                damage_burden_auc_prevented_interval=np.quantile(
                    auc_array, [lower, upper], axis=0
                ),
                fraction_parameter_draws_auc_beneficial=np.mean(
                    auc_array > 0.0, axis=0
                ),
            )
        )

    return InterventionTimingSweepResult(
        architecture_id=architecture_id,
        target_node=target_node,
        mode=mode,
        intensity=float(intensity),
        time_unit=first.time_unit,
        parameter_draw_count=len(models),
        component_names=DAMAGE_COMPONENTS,
        effects=tuple(effects),
    )
