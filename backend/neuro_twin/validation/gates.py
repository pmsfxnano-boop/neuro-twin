"""Scientific quality gates for Batch 03."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neuro_twin.data.schema import Observation


@dataclass(frozen=True)
class GateReport:
    passed: bool
    checks: dict[str, bool]
    failures: tuple[str, ...]


def run_observation_gate(observations: list[Observation]) -> GateReport:
    checks: dict[str, bool] = {}
    checks["nonempty"] = bool(observations)
    checks["finite"] = all(
        all(np.isfinite(np.asarray(o.value, dtype=float))) for o in observations
    ) if observations else False
    checks["provenance"] = all(bool(o.provenance.source_name and o.provenance.access_tier) for o in observations) if observations else False
    checks["temporal"] = all(o.quality.temporal_integrity for o in observations) if observations else False
    checks["no_missing_values"] = all(not o.missing for o in observations) if observations else False
    failures = tuple(k for k, v in checks.items() if not v)
    return GateReport(passed=not failures, checks=checks, failures=failures)
