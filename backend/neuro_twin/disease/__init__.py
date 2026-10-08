"""Disease-specific modules, cohort adapters, and readiness gates."""
from .cohorts import CohortRecord, CohortSummary, parse_participants_tsv, summarize_cohort
from .coverage import CoverageResult, evaluate_module_coverage
from .modules import (
    ALZHEIMER_V1,
    DISEASE_MODULES,
    PARKINSON_V1,
    Disease,
    DiseaseModule,
    ObservationBinding,
    get_disease_module,
)
from .public import PUBLIC_COHORTS, PublicCohortSpec, fetch_public_cohort
from .readiness import DatasetReadiness, assess_dataset_readiness
from .exports import ExportRecord, parse_authorized_export
from .sources import ADNI_PLASMA_V1, CONTROLLED_SOURCES, PPMI_CORE_V1, ControlledSourceSpec, get_controlled_source

__all__ = [
    "ALZHEIMER_V1",
    "DISEASE_MODULES",
    "PARKINSON_V1",
    "Disease",
    "DiseaseModule",
    "ObservationBinding",
    "get_disease_module",
    "CoverageResult",
    "evaluate_module_coverage",
    "DatasetReadiness",
    "assess_dataset_readiness",
    "CohortRecord",
    "CohortSummary",
    "parse_participants_tsv",
    "summarize_cohort",
    "PUBLIC_COHORTS",
    "PublicCohortSpec",
    "fetch_public_cohort",
    "ExportRecord", "parse_authorized_export",
    "ADNI_PLASMA_V1", "PPMI_CORE_V1", "CONTROLLED_SOURCES",
    "ControlledSourceSpec", "get_controlled_source",
]
