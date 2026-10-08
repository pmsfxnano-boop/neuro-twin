# Frontier Console v4

## Runtime topology

Browser: React/TypeScript presentation + Three.js inspection only.

Backend: FastAPI bridge + scientific runtime adapter.

Scientific layer: MRI/PET/biomarkers -> canonical observations -> PIT/QC -> inference -> uncertainty -> OOS -> evidence.

## Integrity invariants

- no synthetic runtime path in production UI;
- no fabricated metrics;
- PIT/provenance/OOS visible;
- PET uncertainty is rendered only when published by RuntimeResult;
- the 3D manifold consumes runtime state/trajectory only;
- the browser does not perform scientific inference.
