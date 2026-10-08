# NEURO-TWIN — Batch 04: Explicit Observation Registry + Probabilistic Uncertainty Core

## Objective

Advance from a feature/panel layer to an auditable model-definition layer and a
numerically stable local uncertainty engine, without introducing unsupported
biological mappings.

## Added

- Versioned observation-model registry.
- Explicit linear operator `y = Hx + b + eps` with feature names and model hash.
- Correlated measurement-error covariance `R` with strict positive-definiteness checks.
- Stable Cholesky whitening utilities.
- Local Gaussian/Laplace covariance from the whitened Gauss–Newton geometry.
- Delta-method predictive covariance.
- Independent matrix-exponential propagation of the canonical affine P-I-N-Q ODE.
- Analytical verification test of the DOP853 integrator against the closed-form
  augmented-matrix propagator.
- Benchmark-only observation configurations remain synthetic; no biomarker is
  hard-coded to a P/I/N/Q state.

## Scientific invariant

A biomarker or image-derived feature enters the latent model only through an
explicit, versioned observation configuration. The data fabric never decides
what a biological feature means mechanistically.

## Source reality checked on 2026-10-07

- ADNI continues to operate as a longitudinal multicenter observational study;
  access to participant-level imaging/clinical/biomarker data remains through
  the LONI IDA for approved users.
- Synapse currently documents REST, Python, R and CLI clients and supports
  programmatic reproducible access.

Web verification references are stored in `evidence/public_source_snapshot_2026-10-07.md`.
