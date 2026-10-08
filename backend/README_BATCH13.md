# NEURO-TWIN — Batch 13

## Quantitative MRI extraction boundary

### Objective

Batch 13 creates the reproducible bridge from a BIDS NIfTI object to quantitative MRI features while keeping anatomical interpretation and latent-state inference separate.

### Added

- Dependency-light NIfTI-1 reader with gzip support for common scalar datatypes.
- Deterministic image-header QC: dimensionality, affine validity, finite voxel fraction, constant-image detection and voxel volume.
- Versioned MRI feature extractor with content hashes and explicit lineage.
- Masked intensity statistics: mean, standard deviation, median, p05, p95, coefficient of variation, voxel count and masked volume.
- Pre-aligned integer label-map ROI extraction for region volumes and intensity summaries.
- Shape-alignment checks that reject implicit registration/resampling.
- Feature extraction orchestration layer.

### Scientific boundary

This batch does **not** claim that image intensity or volume is directly equivalent to P, I, N or Q. The extractor outputs observations/features only. Mapping to the latent P-I-N-Q state remains the responsibility of the versioned observation model.

No atlas registration, segmentation, bias-field correction or scanner harmonization is silently performed here. Those operations must be separate, versioned pipeline stages with their own provenance.

### Validation

The batch contains synthetic NIfTI-1 fixtures and tests for:

- compressed/uncompressed reading;
- geometry and voxel volume;
- header-only inspection;
- masked features + lineage;
- label-map ROI volumes;
- non-finite/constant-image QC;
- mask shape mismatch.

### Environment

The runtime did not have nibabel/nilearn installed. Instead of making an unsupported claim, this batch implements a limited NIfTI-1 scalar reader and keeps a full imaging library as an optional future backend. NIfTI-2 and compound datatypes are explicitly rejected.
