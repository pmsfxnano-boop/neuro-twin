# Disease modules and readiness gates

## Scientific contract

NEURO-TWIN v1.0 defines x(t)=[P(t),I(t),N(t),Q(t)] with disease-specific observation
models. Alzheimer disease uses molecular pathology, tissue response, neurodegeneration
and cognition. Parkinson disease uses alpha-synuclein pathology, neurodegeneration and
function. The specification prohibits treating latent variables as direct biomarkers
and requires observability, identifiability and out-of-sample validation before
translational claims.

## Implemented software contract

Alzheimer v1.0.0 required observations:
- pTau217 -> P
- Abeta42/40 -> P
- GFAP -> I
- NfL -> N
- cognition -> Q

Parkinson v1.0.0 required observations:
- alphaSyn-SAA -> N
- NfL -> N
- MDS-UPDRS -> Q

Digital function is optional for Parkinson; MRI/DAT are optional only with explicit
registered operators.

Diagnosis/group labels remain external evaluation metadata. They are never automatically
inserted into the P-I-N-Q observation vector.

## Current public-cohort readiness

OpenNeuro ds004504 exposes EEG, MMSE and diagnosis labels. The gate can observe only
Q/cognition.

OpenNeuro ds005892 exposes MRI/fMRI and diagnosis labels. The gate blocks disease-specific
P/I/N/Q inference.

The UCI Parkinson telemonitoring dataset exposes UPDRS-derived endpoints and voice
features; these can support the Q/function side of Parkinson only after an explicit
observation operator is registered.

## Gate semantics

READY_FOR_MODULE_INFERENCE = all required observations are present.

PARTIAL_OBSERVABILITY = at least one required observation is present, but some pathways
remain unobserved.

NOT_READY = no required module observation is available.

A dataset with a diagnostic label can still be NOT_READY: diagnosis is an evaluation
target, not an input into the latent state.
