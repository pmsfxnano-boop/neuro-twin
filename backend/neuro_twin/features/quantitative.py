"""Versioned quantitative feature transforms for multimodal observations.

These functions operate on already extracted scalar/tabular quantities. They do
not infer P/I/N/Q identities. Every derived feature must record its parent
features and transform version in the returned metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping


@dataclass(frozen=True)
class DerivedFeature:
    name: str
    value: float
    unit: str | None
    parents: tuple[str, ...]
    transform: str
    version: str = "1.0.0"


def _finite(x: float) -> float:
    x = float(x)
    if not isfinite(x):
        raise ValueError("feature value must be finite")
    return x


def abeta_ratio(abeta42: float, abeta40: float) -> DerivedFeature:
    a42, a40 = _finite(abeta42), _finite(abeta40)
    if a40 <= 0:
        raise ValueError("Abeta40 must be > 0 for Abeta42/40")
    return DerivedFeature(
        name="Abeta42_40_ratio",
        value=a42 / a40,
        unit="ratio",
        parents=("Abeta42", "Abeta40"),
        transform="Abeta42 / Abeta40",
    )


def log_transform(feature: str, value: float, *, offset: float = 0.0) -> DerivedFeature:
    import math
    x = _finite(value) + float(offset)
    if x <= 0:
        raise ValueError("log-transform input must be > 0 after offset")
    return DerivedFeature(
        name=f"log_{feature}",
        value=math.log(x),
        unit="log-scale",
        parents=(feature,),
        transform=f"log({feature}+{offset:g})",
    )


def zscore(feature: str, value: float, mean: float, std: float, *, reference: str) -> DerivedFeature:
    x = _finite(value)
    mu = _finite(mean)
    sigma = _finite(std)
    if sigma <= 0:
        raise ValueError("reference standard deviation must be > 0")
    return DerivedFeature(
        name=f"z_{feature}",
        value=(x - mu) / sigma,
        unit="z-score",
        parents=(feature,),
        transform=f"({feature}-{mu:g})/{sigma:g}",
        version=f"reference:{reference}",
    )


def derive_abeta_ratio_from_mapping(values: Mapping[str, float]) -> DerivedFeature:
    return abeta_ratio(values["Abeta42"], values["Abeta40"])
