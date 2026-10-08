"""Disease-specific cohort and validation infrastructure.

Disease labels are kept outside the scientific observation vector. They are
evaluation targets or strata unless an explicit, versioned observation model
is registered for them.
"""
from .cohorts import CohortRecord, CohortSummary, parse_participants_tsv, summarize_cohort
from .public import PUBLIC_COHORTS, PublicCohortSpec, fetch_public_cohort

__all__ = [
    "CohortRecord",
    "CohortSummary",
    "parse_participants_tsv",
    "summarize_cohort",
    "PUBLIC_COHORTS",
    "PublicCohortSpec",
    "fetch_public_cohort",
]
