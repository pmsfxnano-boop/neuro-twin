'''Disease-module observability/readiness checks.

The purpose is to stop the pipeline before it produces a disease-specific
latent interpretation when the measurements required to observe the relevant
states are absent.
'''
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .modules import DiseaseModule


@dataclass(frozen=True)
class CoverageResult:
    disease: str
    module_version: str
    available_features: tuple[str, ...]
    covered_required_features: tuple[str, ...]
    missing_required_features: tuple[str, ...]
    observed_states: tuple[str, ...]
    unobserved_states: tuple[str, ...]
    status: str
    reasons: tuple[str, ...]


def evaluate_module_coverage(module: DiseaseModule, available_features: Iterable[str]) -> CoverageResult:
    available = {str(x) for x in available_features}
    required = [b for b in module.core_observations if b.required]
    covered = [b for b in required if b.feature in available]
    missing = [b for b in required if b.feature not in available]
    observed_states = sorted({b.target_state for b in covered})
    all_states = {'P', 'I', 'N', 'Q'}
    unobserved_states = sorted(all_states - set(observed_states))

    if not missing:
        status = 'READY_FOR_FULL_OBSERVABILITY'
        reasons = ('all required module observations are present; this is necessary but not sufficient for inference',)
    elif covered:
        status = 'PARTIAL_OBSERVABILITY'
        reasons = ('one or more required module observations are missing', 'latent interpretation must be restricted to observed state axes')
    else:
        status = 'NOT_READY'
        reasons = ('no required module observation is present', 'disease-specific latent inference is blocked')

    return CoverageResult(
        disease=module.disease.value,
        module_version=module.version,
        available_features=tuple(sorted(available)),
        covered_required_features=tuple(b.feature for b in covered),
        missing_required_features=tuple(b.feature for b in missing),
        observed_states=tuple(observed_states),
        unobserved_states=tuple(unobserved_states),
        status=status,
        reasons=reasons,
    )