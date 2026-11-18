"""Mechanism-level dynamic simulator and conditional intervention engine.

This module turns explicitly selected evidence-catalogue edges into a
parameterized nonlinear dynamical system. It does not infer edge signs,
coefficients, initial states, damage mappings, or treatment effects.
Simulated interventions are model-conditional counterfactual scenarios,
not empirical causal effects.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Literal

import numpy as np
from scipy.integrate import solve_ivp

from neuro_twin.network.damage import DAMAGE_COMPONENTS, DamageAccumulationModel, DamageTrajectory
from neuro_twin.network.mechanisms import AlzheimerMechanismCatalog, AlzheimerOriginArchitectureRegistry


EdgeSign = Literal["activation", "inhibition"]
InterventionMode = Literal["clamp", "additive_drive"]
ParameterStatus = Literal["hypothesis", "estimated", "validated"]


def _finite_scalar(value: float, *, name: str, minimum: float | None = None) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result <= minimum:
        raise ValueError(f"{name} must be > {minimum}")
    return result


def _validate_time(time: np.ndarray) -> np.ndarray:
    values = np.asarray(time, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("time must be a finite 1-D array with at least two points")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("time must be strictly increasing")
    return values


@dataclass(frozen=True)
class NodeKinetics:
    """Explicit normalized activity set-point and recovery rate for one node."""

    node_id: str
    baseline_activity: float
    relaxation_rate: float

    def __post_init__(self) -> None:
        if not self.node_id.strip():
            raise ValueError("node_id must be non-empty")
        baseline = _finite_scalar(self.baseline_activity, name="baseline_activity")
        if not 0.0 <= baseline <= 1.0:
            raise ValueError("baseline_activity must lie in [0, 1]")
        rate = _finite_scalar(self.relaxation_rate, name="relaxation_rate", minimum=0.0)
        object.__setattr__(self, "baseline_activity", baseline)
        object.__setattr__(self, "relaxation_rate", rate)


@dataclass(frozen=True)
class EdgeKinetics:
    """Explicit effect assumption for one literature-candidate interaction."""

    edge_id: str
    coefficient: float
    sign: EdgeSign
    hill_half_saturation: float
    hill_cooperativity: float

    def __post_init__(self) -> None:
        if not self.edge_id.strip():
            raise ValueError("edge_id must be non-empty")
        coefficient = _finite_scalar(self.coefficient, name="coefficient")
        half = _finite_scalar(self.hill_half_saturation, name="hill_half_saturation")
        cooperativity = _finite_scalar(self.hill_cooperativity, name="hill_cooperativity")
        if self.sign not in ("activation", "inhibition"):
            raise ValueError("sign must be activation or inhibition")
        if coefficient < 0.0:
            raise ValueError("coefficient must be non-negative")
        if half > 1.0:
            raise ValueError("hill_half_saturation must lie in (0, 1]")
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "hill_half_saturation", half)
        object.__setattr__(self, "hill_cooperativity", cooperativity)


@dataclass(frozen=True)
class InterventionProtocol:
    """Time-bounded virtual perturbation of one mechanism node.

    clamp sets the node to intensity at onset and holds it.
    additive_drive adds a signed, saturating rate input.
    These are abstract mechanism perturbations, not pharmacological dose models.
    """

    intervention_id: str
    target_node: str
    mode: InterventionMode
    start_time: float
    end_time: float
    intensity: float

    def __post_init__(self) -> None:
        if not self.intervention_id.strip() or not self.target_node.strip():
            raise ValueError("intervention_id and target_node must be non-empty")
        start = _finite_scalar(self.start_time, name="start_time")
        end = _finite_scalar(self.end_time, name="end_time")
        intensity = _finite_scalar(self.intensity, name="intensity")
        if end <= start:
            raise ValueError("end_time must be greater than start_time")
        if self.mode not in ("clamp", "additive_drive"):
            raise ValueError("unsupported intervention mode")
        if self.mode == "clamp" and not 0.0 <= intensity <= 1.0:
            raise ValueError("clamp intensity must lie in [0, 1]")
        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "end_time", end)
        object.__setattr__(self, "intensity", intensity)


@dataclass(frozen=True)
class MechanismTrajectory:
    time: np.ndarray
    node_ids: tuple[str, ...]
    activity: np.ndarray

    def node(self, node_id: str) -> np.ndarray:
        try:
            index = self.node_ids.index(node_id)
        except ValueError as exc:
            raise KeyError(node_id) from exc
        return self.activity[:, index].copy()


@dataclass(frozen=True)
class MechanisticSimulationResult:
    architecture_id: str
    model_hash: str
    parameter_status: ParameterStatus
    mechanism: MechanismTrajectory
    damage: DamageTrajectory
    interpretation: Literal[
        "hypothesis_scenario_only",
        "parameter_estimate_conditional_simulation",
        "validated_parameter_conditional_simulation",
    ]


@dataclass(frozen=True)
class DamageInterventionEffect:
    """Positive prevented_damage values mean less modeled damage under intervention."""

    damage_prevented_at_final: np.ndarray
    damage_burden_auc_prevented: np.ndarray
    component_names: tuple[str, ...]
    interpretation: Literal[
        "model_conditional_counterfactual_not_empirical_causal_effect"
    ] = "model_conditional_counterfactual_not_empirical_causal_effect"


@dataclass(frozen=True)
class InterventionComparison:
    baseline: MechanisticSimulationResult
    intervention: MechanisticSimulationResult
    effect: DamageInterventionEffect


@dataclass(frozen=True)
class MechanisticArchitectureModel:
    """Nonlinear, bounded mechanism network linked to persistent-damage maps.

    For node i:
      dz_i/dt = lambda_i * (baseline_i - z_i)
                + sum(activation) beta_e * Hill(z_source) * (1-z_i)
                - sum(inhibition) beta_e * Hill(z_source) * z_i

    Every biological set-point, rate, edge sign, edge magnitude, Hill parameter,
    initial state and damage map must be supplied by the caller.
    """

    architecture_id: str
    time_unit: str
    nodes: tuple[NodeKinetics, ...]
    edges: tuple[EdgeKinetics, ...]
    damage_model: DamageAccumulationModel
    version: str = "1.0.0"
    parameter_status: ParameterStatus = "hypothesis"

    def __post_init__(self) -> None:
        if not self.architecture_id.strip() or not self.time_unit.strip():
            raise ValueError("architecture_id and time_unit must be non-empty")
        if not self.nodes:
            raise ValueError("at least one node kinetic is required")
        if self.parameter_status not in ("hypothesis", "estimated", "validated"):
            raise ValueError("invalid parameter_status")
        node_ids = tuple(item.node_id for item in self.nodes)
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("node kinetics must have unique node_id values")
        edge_ids = tuple(item.edge_id for item in self.edges)
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("edge kinetics must have unique edge_id values")
        damage_inputs = set(self.damage_model.driver_names) | set(self.damage_model.repair_names)
        unknown = damage_inputs - set(node_ids)
        if unknown:
            raise ValueError(f"damage model references nodes outside model: {sorted(unknown)}")
        if self.damage_model.disease != "alzheimer":
            raise ValueError("this mechanism simulator currently supports Alzheimer only")

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(item.node_id for item in self.nodes)

    @property
    def edge_ids(self) -> tuple[str, ...]:
        return tuple(item.edge_id for item in self.edges)

    @property
    def config_hash(self) -> str:
        payload = {
            "architecture_id": self.architecture_id,
            "time_unit": self.time_unit,
            "nodes": [[x.node_id, x.baseline_activity, x.relaxation_rate] for x in self.nodes],
            "edges": [[x.edge_id, x.coefficient, x.sign, x.hill_half_saturation,
                       x.hill_cooperativity] for x in self.edges],
            "damage_model_hash": self.damage_model.config_hash,
            "version": self.version,
            "parameter_status": self.parameter_status,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def validate_against(
        self,
        catalog: AlzheimerMechanismCatalog,
        registry: AlzheimerOriginArchitectureRegistry,
    ) -> "MechanisticArchitectureModel":
        registry.validate_against(catalog)
        architecture = next(
            (x for x in registry.architectures if x.architecture_id == self.architecture_id),
            None,
        )
        if architecture is None:
            raise ValueError(f"unknown origin architecture: {self.architecture_id}")
        catalog_nodes = {x.node_id for x in catalog.nodes}
        if not set(self.node_ids).issubset(catalog_nodes):
            raise ValueError("model contains nodes absent from the mechanism catalog")
        if not set(self.edge_ids).issubset(set(architecture.candidate_edge_ids)):
            raise ValueError("model uses edges outside its selected origin architecture")
        edge_by_id = {x.edge_id: x for x in catalog.edges}
        nodes = set(self.node_ids)
        for parameter in self.edges:
            candidate = edge_by_id[parameter.edge_id]
            if candidate.directionality != "directed_candidate":
                raise ValueError(
                    f"edge {parameter.edge_id} is not a directed candidate; resolve direction before simulation"
                )
            if candidate.source_node not in nodes or candidate.target_node not in nodes:
                raise ValueError(f"edge {parameter.edge_id} endpoints must both be explicitly modeled")
        return self

    def _edge_endpoints(self, catalog: AlzheimerMechanismCatalog):
        node_index = {name: index for index, name in enumerate(self.node_ids)}
        edge_by_id = {x.edge_id: x for x in catalog.edges}
        return tuple(
            (node_index[edge_by_id[p.edge_id].source_node],
             node_index[edge_by_id[p.edge_id].target_node], p)
            for p in self.edges
        )

    def simulate(
        self,
        *,
        catalog: AlzheimerMechanismCatalog,
        registry: AlzheimerOriginArchitectureRegistry,
        time: np.ndarray,
        initial_activity: np.ndarray,
        interventions: tuple[InterventionProtocol, ...] = (),
        relative_tolerance: float = 1e-7,
        absolute_tolerance: float = 1e-9,
    ) -> MechanismTrajectory:
        self.validate_against(catalog, registry)
        t = _validate_time(time)
        initial = np.asarray(initial_activity, dtype=float)
        if initial.shape != (len(self.nodes),):
            raise ValueError(f"initial_activity must have shape ({len(self.nodes)},)")
        if not np.all(np.isfinite(initial)) or np.any((initial < 0.0) | (initial > 1.0)):
            raise ValueError("initial_activity must be finite and lie in [0, 1]")
        rtol = _finite_scalar(relative_tolerance, name="relative_tolerance")
        atol = _finite_scalar(absolute_tolerance, name="absolute_tolerance")
        if rtol >= 1.0 or atol >= 1.0:
            raise ValueError("solver tolerances must be less than 1")
        node_index = {name: index for index, name in enumerate(self.node_ids)}
        for intervention in interventions:
            if intervention.target_node not in node_index:
                raise ValueError(f"intervention target {intervention.target_node} is not modeled")
            if intervention.start_time < t[0] or intervention.end_time > t[-1]:
                raise ValueError("intervention interval must lie within simulation time")
        clamps = [x for x in interventions if x.mode == "clamp"]
        for i, left in enumerate(clamps):
            for right in clamps[i + 1:]:
                overlap = max(left.start_time, right.start_time) < min(left.end_time, right.end_time)
                if overlap and left.target_node == right.target_node:
                    raise ValueError("overlapping clamps on the same node are ambiguous")

        endpoints = self._edge_endpoints(catalog)
        event_times = sorted(
            set(float(x) for x in t)
            | {float(x.start_time) for x in interventions}
            | {float(x.end_time) for x in interventions}
        )
        requested = {float(value): index for index, value in enumerate(t)}
        output = np.empty((len(t), len(self.nodes)), dtype=float)
        state = initial.copy()

        for event_index, event_time in enumerate(event_times):
            if event_time in requested:
                output[requested[event_time]] = state
            if event_index == len(event_times) - 1:
                break
            next_time = event_times[event_index + 1]
            midpoint = 0.5 * (event_time + next_time)
            active = tuple(
                x for x in interventions
                if x.start_time <= midpoint < x.end_time
            )
            for item in active:
                if item.mode == "clamp" and item.start_time == event_time:
                    state[node_index[item.target_node]] = item.intensity
            clamped = {
                node_index[x.target_node] for x in active if x.mode == "clamp"
            }

            def rhs(_time: float, values: np.ndarray) -> np.ndarray:
                derivative = np.asarray([
                    spec.relaxation_rate * (spec.baseline_activity - values[i])
                    for i, spec in enumerate(self.nodes)
                ], dtype=float)
                for source_i, target_i, edge in endpoints:
                    source_value = max(float(values[source_i]), 0.0)
                    source_power = source_value ** edge.hill_cooperativity
                    half_power = edge.hill_half_saturation ** edge.hill_cooperativity
                    signal = source_power / (half_power + source_power)
                    if edge.sign == "activation":
                        derivative[target_i] += edge.coefficient * signal * (1.0 - values[target_i])
                    else:
                        derivative[target_i] -= edge.coefficient * signal * values[target_i]
                for item in active:
                    if item.mode != "additive_drive":
                        continue
                    target_i = node_index[item.target_node]
                    if item.intensity >= 0.0:
                        derivative[target_i] += item.intensity * (1.0 - values[target_i])
                    else:
                        derivative[target_i] += item.intensity * values[target_i]
                for index in clamped:
                    derivative[index] = 0.0
                return derivative

            solution = solve_ivp(
                rhs, (event_time, next_time), state, method="RK45",
                rtol=rtol, atol=atol, t_eval=[next_time],
            )
            if not solution.success or solution.y.shape[1] != 1:
                raise RuntimeError(f"mechanistic integration failed: {solution.message}")
            state = solution.y[:, -1]
            if np.any(state < -1e-6) or np.any(state > 1.0 + 1e-6):
                raise RuntimeError("mechanistic state escaped [0, 1]; review parameter stability")
            state = np.clip(state, 0.0, 1.0)

        return MechanismTrajectory(time=t.copy(), node_ids=self.node_ids, activity=output)

    def simulate_with_damage(
        self,
        *,
        catalog: AlzheimerMechanismCatalog,
        registry: AlzheimerOriginArchitectureRegistry,
        time: np.ndarray,
        initial_activity: np.ndarray,
        initial_damage: np.ndarray,
        interventions: tuple[InterventionProtocol, ...] = (),
    ) -> MechanisticSimulationResult:
        mechanism = self.simulate(
            catalog=catalog, registry=registry, time=time, initial_activity=initial_activity,
            interventions=interventions,
        )
        index = {name: i for i, name in enumerate(mechanism.node_ids)}
        damaging = mechanism.activity[:, [index[name] for name in self.damage_model.driver_names]]
        repair = mechanism.activity[:, [index[name] for name in self.damage_model.repair_names]]
        damage = self.damage_model.accumulate(
            time=mechanism.time, damaging_exposures=damaging,
            repair_capacity=repair, initial_damage=initial_damage,
        )
        interpretation = {
            "hypothesis": "hypothesis_scenario_only",
            "estimated": "parameter_estimate_conditional_simulation",
            "validated": "validated_parameter_conditional_simulation",
        }[self.parameter_status]
        return MechanisticSimulationResult(
            architecture_id=self.architecture_id, model_hash=self.config_hash,
            parameter_status=self.parameter_status, mechanism=mechanism,
            damage=damage, interpretation=interpretation,
        )


def compare_intervention(
    model: MechanisticArchitectureModel,
    *,
    catalog: AlzheimerMechanismCatalog,
    registry: AlzheimerOriginArchitectureRegistry,
    time: np.ndarray,
    initial_activity: np.ndarray,
    initial_damage: np.ndarray,
    interventions: tuple[InterventionProtocol, ...],
) -> InterventionComparison:
    """Compare baseline and perturbed trajectories under identical assumptions."""
    if not interventions:
        raise ValueError("at least one intervention protocol is required")
    baseline = model.simulate_with_damage(
        catalog=catalog, registry=registry, time=time, initial_activity=initial_activity,
        initial_damage=initial_damage,
    )
    treated = model.simulate_with_damage(
        catalog=catalog, registry=registry, time=time, initial_activity=initial_activity,
        initial_damage=initial_damage, interventions=interventions,
    )
    prevented_final = baseline.damage.cumulative_damage[-1] - treated.damage.cumulative_damage[-1]
    baseline_auc = np.trapezoid(baseline.damage.cumulative_damage, baseline.damage.time, axis=0)
    treated_auc = np.trapezoid(treated.damage.cumulative_damage, treated.damage.time, axis=0)
    return InterventionComparison(
        baseline=baseline, intervention=treated,
        effect=DamageInterventionEffect(
            damage_prevented_at_final=prevented_final,
            damage_burden_auc_prevented=baseline_auc - treated_auc,
            component_names=DAMAGE_COMPONENTS,
        ),
    )


def summarize_parameter_ensemble(
    comparisons: tuple[InterventionComparison, ...],
    *,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> dict[str, np.ndarray]:
    """Summarize effect variation across caller-supplied parameter draws only."""
    if not comparisons:
        raise ValueError("at least one intervention comparison is required")
    lower = _finite_scalar(lower_quantile, name="lower_quantile")
    upper = _finite_scalar(upper_quantile, name="upper_quantile")
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
    values = np.stack([x.effect.damage_prevented_at_final for x in comparisons], axis=0)
    auc = np.stack([x.effect.damage_burden_auc_prevented for x in comparisons], axis=0)
    return {
        "damage_prevented_median": np.quantile(values, 0.5, axis=0),
        "damage_prevented_interval": np.quantile(values, [lower, upper], axis=0),
        "damage_burden_auc_prevented_median": np.quantile(auc, 0.5, axis=0),
        "damage_burden_auc_prevented_interval": np.quantile(auc, [lower, upper], axis=0),
    }
