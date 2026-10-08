# NEURO-TWIN — Batch 14

## Scientific imaging-to-observation boundary

### Objective

Strengthen the MRI pathway so that subject/session binding, spatial interpretation, processing transforms, and point-in-time availability are explicit and auditable before a feature is promoted to a canonical observation.

### Changes

- BIDS filename entity parser for subject/session and common MRI/derivative entities; duplicate entities are rejected rather than overwritten.
- Full NIfTI-1 qform quaternion reconstruction, including qfac, when sform is absent.
- Explicit `sform`/`qform` disagreement QC. When both forms are present and materially disagree, the image is rejected by the default QC gate.
- Spatial compatibility assessment based on shape and affine agreement.
- Explicit nearest-neighbour label-map resampling with content hashes, affine mapping, interpolation choice, and parameter hashes. Resampling is opt-in and never implicit.
- Mask QC that permits valid constant binary masks while still rejecting empty selections.
- Strict BIDS `*_scans.tsv` acquisition-time resolver using `filename` + `acq_time`; filesystem modification time and processing time are never substituted.
- Feature-to-observation bridge with three distinct temporal quantities:
  - acquisition time;
  - feature processing / availability time;
  - system ingest time.
- MRI BIDS pipeline now binds subject/session entities and records spatial-transform provenance when a label map is explicitly resampled.

### Scientific rationale

BIDS filenames encode entities such as `sub`, `ses`, `acq`, and `run`; duplicate entities are invalid. MRI files use entity/suffix/extension structures defined by the BIDS specification. BIDS also supports `*_scans.tsv` with `acq_time` for acquisition timestamps. The implementation therefore refuses to infer acquisition time from filesystem metadata or pipeline execution time.

The imaging layer remains separate from the P-I-N-Q latent state. MRI features are observations that enter the versioned observation operator `h(x, theta) + epsilon`; no feature is silently identified with P, I, N, or Q.

### Validation

Executed successfully:

- Batch 13 imaging regressions: 6/6
- Batch 14 specific tests: 8/8
- Batch 09–14 regression subset: 30/30
- Python compile check: PASS
- End-to-end synthetic BIDS MRI -> feature -> canonical observation smoke test: PASS

The entire historical suite was also started, but the execution environment timed out before completion. It is therefore not reported as a full-suite PASS.

### Environment limitations

- `scipy`: available and used only for explicit nearest-neighbour categorical label-map resampling.
- `nibabel`: not installed; the project continues to use the dependency-light NIfTI-1 reader for the supported scalar formats.
- `nilearn`: not installed.
- No claim is made that a live external dataset was downloaded during this batch.

### Important boundary

This batch does **not** silently perform registration, segmentation, atlas mapping, bias-field correction, scanner harmonization, or anatomical normalization. Each operation must remain independently versioned and provenance-bearing.
