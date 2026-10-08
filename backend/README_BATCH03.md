# NEURO-TWIN — Batch 03: Multimodal Observation + Imaging Manifest Layer

## Objective

Advance from source-independent ingestion to an explicit multimodal observation layer while preserving the central scientific invariant: raw data, derived features and latent P/I/N/Q states remain separate objects.

## Added

- Versioned quantitative feature transforms:
  - Abeta42/40 ratio
  - log transforms
  - reference z-scores
- Multimodal observation panel assembly by subject/event.
- Explicit missing-feature behavior (`require_all`).
- BIDS participant/event tabular readers.
- Stable imaging manifest for NIfTI files without loading voxels.
- Sidecar metadata attachment to image records.
- Observation quality gate.
- External-source evidence snapshot documenting current OpenNeuro, BioStudies and NCBI capabilities and the public OpenNeuro `ds004767` metadata smoke-test target.
- Tests for derived features, multimodal panels and BIDS image manifests.

## Scientific invariant

No biomarker or imaging feature is hard-coded to P/I/N/Q. Any observation operator mapping is a model configuration that can be versioned and audited.

## Runtime limitation in this environment

DNS/network access from the execution container was unavailable. Therefore Batch 03 does not falsely claim a live API download. The implementation contains the adapter path and uses a reproducible public metadata target; a live ingestion run should be executed when network access is available.
