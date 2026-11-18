"""Explicit cumulative persistent-damage model for Neuro-Twin.

This is a sidecar research module, not a seventh coordinate silently appended
to the existing six-domain state vector. It integrates excess damaging
exposures over modeled repair capacity into persistent damage compartments.
All mappings and coefficients must be supplied explicitly and remain
hypotheses until calibrated and independently validated.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Literal

import numpy as np


Disease = Literal["alzheimer", "parkinson"]
ParameterStatus = Literal["hypothesis", "estimated", "validated"]

DAMAGE_COMPONENTS = (
    "synaptic_structural",
    "axonal_neuritic",
    "neuronal_somatic",
    "neurovascular",
)


def _readonly_array(value: np.ndarray, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative")
    array.setflags(write=False)
    return array


def _validate_names(
    values: tuple[str, ...], *, name: str, allow_empty: bool = False
) -> tuple[str, ...]:
    normalized = tuple(values)
    if not normalized and not allow_empty:
        raise ValueError(f"{name} must contain at least one name")
    if any(not isinstance(item, str) or not item.strip() for item in normalized):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must contain unique names")
    return normalized


@dataclass(frozen=True)
class DamageTrajectory:
    """Irregular-time persistent-damage trajectory.

    Arrays use rows as visit times and columns in DAMAGE_COMPONENTS order.
    Damage values are cumulative model units and are not clinical scores.
    """

    time: np.ndarray
    cumulative_damage: np.ndarray
    damage_rate: np.ndarray
    excess_pressure: np.ndarray


@dataclass(frozen=True)
class DamageAccumulationModel:
    """Disease-specific cumulative-damage integrator.

    damage_weights maps explicitly named normalized damaging exposures
    to each persistent-damage compartment. repair_weights maps explicitly
    named normalized repair/resilience capacities to the same compartments.
    Positive excess pressure is multiplied by rate_scale and integrated
    over time. Existing damage never decreases in this module: repair
    capacity can prevent additional damage, but does not erase prior
    neuronal loss or structural injury.

    Driver and repair inputs must be normalized to [0, 1] by a documented,
    versioned upstream measurement/latent model. This class does not normalize
    raw biomarkers or infer biological mappings from names.
    """

    disease: Disease
    time_unit: str
    driver_names: tuple[str, ...]
    repair_names: tuple[str, ...]
    damage_weights: np.ndarray
    repair_weights: np.ndarray
    rate_scale: np.ndarray
    version: str = "1.0.0"
    parameter_status: ParameterStatus = "hypothesis"

    def __post_init__(self) -> None:
        if self.disease not in ("alzheimer", "parkinson"):
            raise ValueError("disease must be 'alzheimer' or 'parkinson'")
        if not isinstance(self.time_unit, str) or not self.time_unit.strip():
            raise ValueError("time_unit must be a non-empty string")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("version must be a non-empty string")
        if self.parameter_status not in ("hypothesis", "estimated", "validated"):
            raise ValueError("invalid parameter_status")

        drivers = _validate_names(self.driver_names, name="driver_names")
        repairs = _validate_names(self.repair_names, name="repair_names", allow_empty=True)
        if set(drivers) & set(repairs):
            raise ValueError("driver_names and repair_names must be distinct")

        n_components = len(DAMAGE_COMPONENTS)
        driver_weights = _readonly_array(
            self.damage_weights,
            name="damage_weights",
            shape=(n_components, len(drivers)),
        )
        repair_weights = _readonly_array(
            self.repair_weights,
            name="repair_weights",
            shape=(n_components, len(repairs)),
        )
        scales = _readonly_array(
            self.rate_scale,
            name="rate_scale",
            shape=(n_components,),
        )
        if np.any(scales <= 0.0):
            raise ValueError("rate_scale must be strictly positive")
        if np.any(np.sum(driver_weights, axis=1) <= 0.0):
            raise ValueError("each damage component must have at least one explicit driver")

        object.__setattr__(self, "driver_names", drivers)
        object.__setattr__(self, "repair_names", repairs)
        object.__setattr__(self, "damage_weights", driver_weights)
        object.__setattr__(self, "repair_weights", repair_weights)
        object.__setattr__(self, "rate_scale", scales)

    @property
    def component_names(self) -> tuple[str, ...]:
        return DAMAGE_COMPONENTS

    @property
    def config_hash(self) -> str:
        payload = {
            "disease": self.disease,
            "time_unit": self.time_unit,
            "driver_names": list(self.driver_names),
            "repair_names": list(self.repair_names),
            "damage_weights": self.damage_weights.tolist(),
            "repair_weights": self.repair_weights.tolist(),
            "rate_scale": self.rate_scale.tolist(),
            "version": self.version,
            "parameter_status": self.parameter_status,
            "component_names": list(DAMAGE_COMPONENTS),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def instantaneous_rate(
        self,
        damaging_exposures: np.ndarray,
        repair_capacity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return non-negative damage rate and excess pressure at one time point."""
        exposures = np.asarray(damaging_exposures, dtype=float)
        repair = np.asarray(repair_capacity, dtype=float)
        if exposures.shape != (len(self.driver_names),):
            raise ValueError(f"damaging_exposures must have shape ({len(self.driver_names)},)")
        if repair.shape != (len(self.repair_names),):
            raise ValueError(f"repair_capacity must have shape ({len(self.repair_names)},)")
        if not np.all(np.isfinite(exposures)) or not np.all(np.isfinite(repair)):
            raise ValueError("exposure and repair inputs must be finite")
        if np.any((exposures < 0.0) | (exposures > 1.0)):
            raise ValueError("damaging_exposures must be normalized to [0, 1]")
        if np.any((repair < 0.0) | (repair > 1.0)):
            raise ValueError("repair_capacity must be normalized to [0, 1]")

        pressure = self.damage_weights @ exposures
        repair_effect = self.repair_weights @ repair
        excess = np.maximum(pressure - repair_effect, 0.0)
        return self.rate_scale * excess, excess

    def accumulate(
        self,
        *,
        time: np.ndarray,
        damaging_exposures: np.ndarray,
        repair_capacity: np.ndarray,
        initial_damage: np.ndarray,
    ) -> DamageTrajectory:
        """Integrate persistent damage using trapezoidal integration.

        Time points may be irregular but must be strictly increasing. The
        trapezoidal rule assumes linear interpolation of the supplied
        instantaneous rates between adjacent observations; it does not imply
        that biological dynamics are truly linear between visits.
        """
        t = np.asarray(time, dtype=float)
        exposures = np.asarray(damaging_exposures, dtype=float)
        repair = np.asarray(repair_capacity, dtype=float)
        initial = np.asarray(initial_damage, dtype=float)

        if t.ndim != 1 or len(t) < 1 or not np.all(np.isfinite(t)):
            raise ValueError("time must be a finite 1-D array with at least one point")
        if len(t) > 1 and np.any(np.diff(t) <= 0.0):
            raise ValueError("time must be strictly increasing")
        if exposures.shape != (len(t), len(self.driver_names)):
            raise ValueError(
                f"damaging_exposures must have shape ({len(t)}, {len(self.driver_names)})"
            )
        if repair.shape != (len(t), len(self.repair_names)):
            raise ValueError(
                f"repair_capacity must have shape ({len(t)}, {len(self.repair_names)})"
            )
        if initial.shape != (len(DAMAGE_COMPONENTS),):
            raise ValueError(f"initial_damage must have shape ({len(DAMAGE_COMPONENTS)},)")
        if not np.all(np.isfinite(initial)) or np.any(initial < 0.0):
            raise ValueError("initial_damage must be finite and non-negative")

        rates = np.empty((len(t), len(DAMAGE_COMPONENTS)), dtype=float)
        excess = np.empty_like(rates)
        for index in range(len(t)):
            rates[index], excess[index] = self.instantaneous_rate(
                exposures[index], repair[index]
            )

        cumulative = np.empty_like(rates)
        cumulative[0] = initial
        for index, dt in enumerate(np.diff(t), start=1):
            increment = 0.5 * (rates[index - 1] + rates[index]) * float(dt)
            cumulative[index] = cumulative[index - 1] + increment

        return DamageTrajectory(
            time=t.copy(),
            cumulative_damage=cumulative,
            damage_rate=rates,
            excess_pressure=excess,
        )
