# NEURO-TWIN Frontier Web App v5

## What changed

The previous console was primarily a runtime viewer. v5 turns it into an executable web application path:

`user data → canonical validation → explicit observation operator → P-I-N-Q inference → uncertainty → strict temporal OOS → Runtime Engine Adapter → WebSocket/UI`

## Main additions

1. `backend/neuro_twin/runtime/runner.py`
   - executable P-I-N-Q runtime for user-supplied canonical observations;
   - irregular/missing feature handling;
   - explicit observation operator;
   - deterministic fit with exact sensitivities;
   - Laplace covariance;
   - retrospective vs prospective-OOS modes;
   - OOS metrics;
   - RuntimeResult creation.

2. `POST /v1/runtime/run`
   - runs the scientific engine from the web application;
   - publishes only validated research-observational results.

3. Frontend Data & Run screen
   - receives a JSON runtime package;
   - validates basic package structure in-browser;
   - submits to the Python engine;
   - shows actual execution state and result.

4. PET uncertainty path
   - optional PET kinetic posterior and explicit PET→Neuro binding can be included in the package;
   - Runtime Engine Adapter propagates PET covariance into the Neuro-Twin state update.

5. Production guardrails
   - synthetic/reference runtime remains blocked;
   - prospective OOS keeps future test observations outside the live posterior envelope;
   - PIT/provenance hashes remain required.

## Validation

- Backend `compileall`: PASS
- Frontend TypeScript source check with local shims: PASS
- Runtime runner unit tests: 2/2 PASS
- HTTP runtime execution smoke test: PASS
- PET→state uncertainty propagation smoke test: PASS

A production Vite bundle was not claimed as executed because the construction environment had no installed React/Vite/Three.js dependency tree and external package installation timed out.
