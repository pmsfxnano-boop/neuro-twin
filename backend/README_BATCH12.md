# NEURO-TWIN — Batch 12

## Real-source ingestion boundary and immutable provenance snapshots

### Purpose

Batch 12 changes phase: the engine now has a source-snapshot boundary designed for real public research data. The first verified target is OpenNeuro `ds004767` version `1.0.0`.

The target is intentionally treated as an **ingestion/metadata smoke-test**, not as a longitudinal clinical cohort. The public source record describes 55 participants, one session, MRI, BIDS 1.4.0, raw dataset type, CC0 license, and DOI `10.18112/openneuro.ds004767.v1.0.0`.

### New components

- `data/snapshot.py`: immutable snapshot contract, canonical JSON hashing, content-addressed persistence.
- `data/openneuro_manifest.py`: OpenNeuro file-tree to stable source-object/BIDS manifest conversion.
- `data/source_sync.py`: live GraphQL sync orchestration with immutable snapshot persistence.
- `evidence/source_snapshots/openneuro_ds004767_verified.json`: externally verified source facts; explicitly marked as web-verified rather than runtime-fetched in this environment.

### Reproducibility contract

Every live source sync must retain:

1. request SHA-256;
2. raw response SHA-256;
3. manifest SHA-256;
4. source dataset/version;
5. endpoint;
6. retrieval timestamp;
7. access tier/license;
8. immutable snapshot id.

This prevents an upstream dataset update from silently changing a previously run scientific experiment.

### Current environment limitation

The execution container has temporary DNS/network resolution failure. Therefore Batch 12 does **not** claim a live OpenNeuro download. The live adapter is implemented and tested with deterministic fixtures, while the current `ds004767` source facts were verified against the official OpenNeuro web record and recorded separately.
