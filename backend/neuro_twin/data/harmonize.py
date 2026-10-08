"""Explicit, auditable observation harmonization.

No biological meaning is inferred here. Transformations are declared in a
registry and every transformation emits a record that can be persisted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import math

from neuro_twin.data.schema import Observation


@dataclass(frozen=True)
class TransformRecord:
    feature: str
    source_unit: str | None
    target_unit: str | None
    transform: str


@dataclass(frozen=True)
class FeatureRule:
    source_feature: str
    canonical_feature: str
    source_unit: str | None = None
    target_unit: str | None = None
    transform_name: str = "identity"


class UnitRegistry:
    """Minimal high-integrity unit conversion registry."""

    def __init__(self) -> None:
        self._rules: dict[tuple[str, str], Callable[[float], float]] = {
            ("ng/mL", "pg/mL"): lambda x: x * 1000.0,
            ("pg/mL", "ng/mL"): lambda x: x / 1000.0,
            ("ng/L", "pg/mL"): lambda x: x,
            ("pg/mL", "ng/L"): lambda x: x,
        }

    def convert(self, value: float, source: str | None, target: str | None) -> float:
        if source == target or target is None or source is None:
            return float(value)
        try:
            return float(self._rules[(source, target)](float(value)))
        except KeyError as exc:
            raise ValueError(f"no declared unit conversion: {source!r} -> {target!r}") from exc


def harmonize(observation: Observation, rule: FeatureRule, registry: UnitRegistry | None = None) -> tuple[Observation, TransformRecord]:
    registry = registry or UnitRegistry()
    if observation.feature != rule.source_feature:
        raise ValueError(f"rule/source mismatch: {observation.feature} != {rule.source_feature}")
    if isinstance(observation.value, list):
        values = [registry.convert(v, observation.unit, rule.target_unit) for v in observation.value]
    else:
        values = registry.convert(float(observation.value), observation.unit, rule.target_unit)
    transformed = observation.model_copy(update={
        "feature": rule.canonical_feature,
        "value": values,
        "unit": rule.target_unit,
    })
    return transformed, TransformRecord(
        feature=rule.canonical_feature,
        source_unit=observation.unit,
        target_unit=rule.target_unit,
        transform=rule.transform_name,
    )
