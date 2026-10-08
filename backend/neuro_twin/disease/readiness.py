'''Dataset-level disease readiness contracts.

Diagnosis/group labels remain evaluation targets, never latent-state inputs.
'''
from __future__ import annotations

from dataclasses import dataclass

from .coverage import CoverageResult, evaluate_module_coverage
from .modules import Disease, get_disease_module


@dataclass(frozen=True)
class DatasetReadiness:
    dataset_id: str
    dataset_version: str
    disease: str
    available_features: tuple[str, ...]
    cohort_label_available: bool
    coverage: CoverageResult
    status: str
    note: str


PUBLIC_DATASET_FEATURES = {
    'openneuro-ds004504': ('EEG', 'MMSE', 'diagnosis_group'),
    'openneuro-ds005892': ('MRI', 'fMRI', 'diagnosis_group'),
    'uci-parkinsons-telemonitoring-189': ('motor_UPDRS', 'total_UPDRS', 'voice'),
}


def assess_dataset_readiness(*, dataset_id: str, dataset_version: str, disease: Disease | str, available_features: tuple[str, ...] | None = None, cohort_label_available: bool = True) -> DatasetReadiness:
    module = get_disease_module(disease)
    features = tuple(available_features if available_features is not None else PUBLIC_DATASET_FEATURES.get(dataset_id, ()))
    normalized = set(features)
    if 'MMSE' in normalized:
        normalized.add('cognition')
    if 'motor_UPDRS' in normalized or 'total_UPDRS' in normalized:
        normalized.add('MDS_UPDRS')
    coverage = evaluate_module_coverage(module, normalized)
    note = 'The disease label is retained as an external evaluation target.' if cohort_label_available else 'No disease label is available; only measurement coverage can be assessed.'
    return DatasetReadiness(
        dataset_id=dataset_id, dataset_version=dataset_version, disease=module.disease.value,
        available_features=tuple(sorted(normalized)), cohort_label_available=cohort_label_available,
        coverage=coverage, status=coverage.status, note=note,
    )