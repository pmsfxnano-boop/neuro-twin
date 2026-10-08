# NEURO-TWIN — Batch 15

## Quantitative PET integrity, timing, uncertainty, and explicit PET→MRI transforms

### Objective

Build the PET boundary as an independently versioned quantitative layer. The implementation must validate timing, decay-correction metadata, tracer identity, frame structure, injected activity/body-weight contracts, quantitative PET metrics, uncertainty propagation, and explicit PET→target-space transforms without inventing tracer-specific calibration constants.

### Scientific boundary

PET remains an observation modality. It is not equated with P/I/N/Q. PET-derived features enter the canonical observation model only after metadata/QC/provenance gates.

### Implemented

- Strict PET-BIDS metadata validator for `TimeZero`, `ScanStart`, `InjectionStart`, `FrameTimesStart`, `FrameDuration`, `ImageDecayCorrected`, and `ImageDecayCorrectionTime`.
- Optional but validated injected-radioactivity and body-weight fields for SUV calculations.
- Frame midpoint calculation.
- ROI activity mean.
- Weight-normalized SUV from Bq/mL, kg, and Bq.
- SUVR with explicit target/reference separation.
- First-order delta-method ratio uncertainty with optional target/reference covariance.
- Caller-supplied linear Centiloid transform; no tracer-specific constants are hard-coded.
- Explicit PET-to-target affine resampling for continuous quantitative images with linear interpolation.
- PET 4D frame preservation during spatial transformation.
- Transformation provenance including affine chain, interpolation choice, target/source geometry, and parameter hash.

### What is deliberately NOT automated

- tracer-specific partial-volume correction;
- kinetic modeling (SRTM/Logan/2TCM);
- arterial input estimation;
- reference-region selection;
- PET↔MRI registration estimation;
- Centiloid calibration constants;
- decay correction of raw PET activity values.

Those remain separate scientific stages. Missing metadata causes rejection rather than silent repair.

### Validation

Batch 15 specific tests: 9/9 PASS
Python compile check: PASS

The historical full-suite is not claimed as complete when the execution environment times out.

### Source basis

The current BIDS PET specification requires PET timing metadata including `TimeZero`, `ScanStart`, `InjectionStart`, `FrameTimesStart`, and `FrameDuration`, and requires explicit decay-correction metadata. citeturn664186search1turn471622search4

The Centiloid project established a framework to standardize amyloid PET quantitative outputs across tracers and methods; therefore this code requires the calibration mapping as an explicit validated input rather than embedding a universal constant. citeturn471622search0

### Known environment limitation

The current environment may not contain `nibabel`/`nilearn`; PET calculations in this batch operate on already-loaded arrays and validated metadata. No live PET dataset download is claimed unless network execution is explicitly successful.
