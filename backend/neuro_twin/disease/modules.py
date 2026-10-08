'''Versioned disease modules for the NEURO-TWIN clinical state x=[P,I,N,Q].

These definitions are a software transcription of the frozen v1.0 conceptual
specification. They do not invent mappings for measurements that are not
present in the source specification.
'''
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Disease(StrEnum):
    ALZHEIMER = 'alzheimer'
    PARKINSON = 'parkinson'


@dataclass(frozen=True)
class ObservationBinding:
    feature: str
    target_state: str
    required: bool
    direction: str
    notes: str


@dataclass(frozen=True)
class DiseaseModule:
    disease: Disease
    version: str
    core_observations: tuple[ObservationBinding, ...]
    discrete_drivers: tuple[str, ...]
    optional_observations: tuple[ObservationBinding, ...]
    forbidden_claims: tuple[str, ...]


ALZHEIMER_V1 = DiseaseModule(
    disease=Disease.ALZHEIMER,
    version='1.0.0',
    core_observations=(
        ObservationBinding('pTau217', 'P', True, 'pathology', 'tau pathology proxy'),
        ObservationBinding('Abeta42_40', 'P', True, 'pathology', 'amyloid pathology ratio'),
        ObservationBinding('GFAP', 'I', True, 'tissue_response', 'glial/astrocytic response proxy'),
        ObservationBinding('NfL', 'N', True, 'neurodegeneration', 'neuroaxonal injury proxy'),
        ObservationBinding('cognition', 'Q', True, 'function', 'clinical functional/cognitive endpoint'),
    ),
    discrete_drivers=('S_AD',),
    optional_observations=(
        ObservationBinding('MRI', 'N', False, 'imaging', 'must be explicitly registered'),
        ObservationBinding('PET', 'P', False, 'imaging', 'must be explicitly registered'),
    ),
    forbidden_claims=(
        'latent P/I/N/Q values are not direct biomarkers',
        'a cross-sectional cohort is not longitudinal progression validation',
        'diagnosis cannot be asserted from one latent state alone',
    ),
)


PARKINSON_V1 = DiseaseModule(
    disease=Disease.PARKINSON,
    version='1.0.0',
    core_observations=(
        ObservationBinding('alphaSyn_SAA', 'N', True, 'molecular_driver', 'binary molecular driver'),
        ObservationBinding('NfL', 'N', True, 'neurodegeneration', 'neuroaxonal injury proxy'),
        ObservationBinding('MDS_UPDRS', 'Q', True, 'function', 'clinical motor/non-motor burden'),
    ),
    discrete_drivers=('S_PD',),
    optional_observations=(
        ObservationBinding('digital_function', 'Q', False, 'function', 'digital functional endpoint'),
        ObservationBinding('MRI', 'N', False, 'imaging', 'must be explicitly registered'),
        ObservationBinding('DAT', 'N', False, 'imaging', 'must be explicitly registered'),
    ),
    forbidden_claims=(
        'alpha-synuclein SAA is a discrete molecular input, not a continuous disease clock',
        'I is optional and must not be reintroduced without sufficient identifiability',
        'latent P/I/N/Q values are not direct biomarkers',
        'diagnosis cannot be asserted from one latent state alone',
    ),
)


DISEASE_MODULES: dict[Disease, DiseaseModule] = {
    Disease.ALZHEIMER: ALZHEIMER_V1,
    Disease.PARKINSON: PARKINSON_V1,
}


def get_disease_module(disease: Disease | str) -> DiseaseModule:
    return DISEASE_MODULES[Disease(disease)]