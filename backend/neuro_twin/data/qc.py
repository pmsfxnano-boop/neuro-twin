"""Source-independent quality gate."""
from __future__ import annotations

from dataclasses import dataclass

from neuro_twin.data.schema import Observation


@dataclass(frozen=True)
class QCResult:
    accepted: bool
    reasons: tuple[str, ...] = ()


def run_qc(observation: Observation) -> QCResult:
    reasons: list[str] = []
    if not observation.quality.schema_valid:
        reasons.append("schema_invalid")
    if not observation.quality.temporal_integrity:
        reasons.append("temporal_integrity")
    if not observation.quality.unit_consistent:
        reasons.append("unit_inconsistency")
    if not observation.quality.assay_consistent:
        reasons.append("assay_inconsistency")
    if not observation.quality.provenance_complete:
        reasons.append("incomplete_provenance")
    if observation.missing:
        reasons.append("missing_observation")
    return QCResult(accepted=not reasons, reasons=tuple(reasons))
