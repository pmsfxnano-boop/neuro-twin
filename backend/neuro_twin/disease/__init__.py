"""Disease-specific cohort and validation infrastructure.

Disease labels are kept outside the scientific observation vector. They are
evaluation targets or strata unless an explicit, versioned observation model
is registered for them.
"""
from .cohorts import CohortRecord, CohortSummary, parse_participants_tsv, summarize_cohort
from .public import PUBLIC_COHORTS, PublicCohortSpec, fetch_public_cohort
from .evaluation import DiseaseEvaluationResult, SubjectDiseaseRecord, evaluate_binary_disease_endpoint
from .registry import DiseaseCohortDefinition, PARKINSON_ANT, ALZHEIMER_EEG
from .coverage import CoverageResult, evaluate_module_coverage
from .modules import ALZHEIMER_V1, DISEASE_MODULES, PARKINSON_V1, Disease, DiseaseModule, ObservationBinding, get_disease_module
from .readiness import DatasetReadiness, assess_dataset_readiness
from .exports import ExportRecord, parse_authorized_export
from .sources import ADNI_PLASMA_V1, CONTROLLED_SOURCES, PPMI_CORE_V1, ControlledSourceSpec, get_controlled_source

__all__ = [
    "CohortRecord", "CohortSummary", "parse_participants_tsv", "summarize_cohort",
    "PUBLIC_COHORTS", "PublicCohortSpec", "fetch_public_cohort",
    "DiseaseEvaluationResult", "SubjectDiseaseRecord", "evaluate_binary_disease_endpoint",
    "DiseaseCohortDefinition", "PARKINSON_ANT", "ALZHEIMER_EEG",
    "CoverageResult", "evaluate_module_coverage",
    "ALZHEIMER_V1", "DISEASE_MODULES", "PARKINSON_V1", "Disease", "DiseaseModule",
    "ObservationBinding", "get_disease_module",
    "DatasetReadiness", "assess_dataset_readiness",
    "ExportRecord", "parse_authorized_export",
    "ADNI_PLASMA_V1", "PPMI_CORE_V1", "CONTROLLED_SOURCES",
    "ControlledSourceSpec", "get_controlled_source",
]
