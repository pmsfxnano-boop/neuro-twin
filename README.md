# NEURO-TWIN

Scientific research application for a mechanistic neurodegenerative-disease digital twin.

NEURO-TWIN is organized as a monorepo with three explicit layers:

```text
MRI / PET / biomarkers
        ↓
canonical observations + PIT + provenance
        ↓
Python scientific engine
        ↓
Runtime Engine Adapter
        ↓
FastAPI + WebSocket
        ↓
React + TypeScript + Three.js PWA
```

## Repository layout

- `backend/` — scientific engine, data fabric, PET/MRI processing, inference, validation, runtime, API bridge.
- `frontend-app/` — production web application (React + TypeScript + Vite + Three.js).
- `contracts/` — future stable API/data contracts; canonical contracts currently live under `INPUT_CONTRACT.md` and backend runtime models.
- `docs/` — architecture and operations documentation.
- `.github/` — CI, issue templates and repository automation.

## Scientific invariants

1. No synthetic/reference runtime path is exposed as a production execution path.
2. Future observations must not enter a historical state estimate under prospective PIT/OOS mode.
3. Provenance and hashes are mandatory for publishable runtime results.
4. PET uncertainty may propagate to the Neuro-Twin posterior only through an explicit, versioned PET→Neuro observation binding.
5. UI state is downstream of the scientific runtime; the frontend must not fabricate scientific values.
6. Raw clinical/imaging data are not committed to Git.

## Six-domain dynamic network — research prototype

An additive six-domain network package is being developed under `backend/neuro_twin/network/`:

- `P` progression, `I` inflammation, `M` metabolism, `N` neuronal integrity, `Q` functional capacity, and `Pr` proteostasis.
- The v1.0 baseline uses an explicit continuous-time affine network with directed coupling parameters, exact irregular-time propagation, explicit observation operators, and six-state filtering/smoothing.
- Observability and local parameter-identifiability diagnostics report rank, weak/unidentified combinations, and schedule/missingness effects before calibration.
- A disease-specific evidence register preserves candidate biological interactions and citations while preventing literature hypotheses from becoming fitted or causal parameters.
- A hierarchical domain ontology decomposes M and Pr into non-interchangeable subconstructs and leaves domain aggregation uncalibrated.
- A typed measurement catalogue validates modality, assay, canonical units, uncertainty semantics and provenance while keeping all latent bindings unregistered.
- A disease-specific coverage report identifies measured proxies, missing subconstructs and blocked observations without claiming latent observability.
- Alzheimer and Parkinson use separate parameterizations. No default biological coefficients or biomarker-to-state mappings are invented.
- This is a research prototype; it does not replace the existing P-I-N-Q runtime and is not clinically validated.

- A separate cumulative persistent-damage module is implemented in `network/damage.py`, with synaptic/structural, axonal/neuritic, neuronal-somatic and neurovascular compartments; explicit damaging exposures and repair capacity; irregular-time integration; no invented biomarker or latent-state mappings. It remains a sidecar, not yet coupled to the six-state dynamics.
- An evidence-graded Alzheimer mechanism catalogue contains 28 multiscale nodes and 30 candidate interactions. It distinguishes familial genetic routes, late-onset susceptibility, pathology, amplifiers, damage, reserve and clinical phenotype; it is not a validated causal DAG.
- Seven competing Alzheimer-origin architectures are stored in `network/alzheimer_origin_architectures_v1.json` and validated against the mechanism graph. Each includes discriminating predictions and falsification criteria; they are explicitly unranked hypotheses.
- A nonlinear mechanism-level simulator executes explicitly parameterized candidate interactions using bounded Hill-type dynamics. It requires a validated evidence-registry binding, explicit edge signs and coefficients, and explicit initial states.
- Virtual intervention protocols support time-bounded mechanism clamps or additive drives. Baseline and intervention trajectories use identical initial states; the engine reports component-wise accumulated damage prevented at the horizon and damage-burden AUC differences. These are conditional model counterfactuals, not empirical causal effects.
- A parameter-ensemble summarizer propagates only caller-supplied parameter draws. A held-out predictive scoring module compares rival architecture forecasts on a common observation mask using Gaussian log predictive density, RMSE and interval coverage.
- A versioned mechanism-level observation operator maps explicit node-state combinations to transformed measurement features. It requires an explicit H matrix, units, noise covariance, source-catalog hash and transformation version; it creates no biomarker mappings by default.
- A constrained nonlinear least-squares calibration module fits only caller-selected mechanism parameters within explicit bounds, whitens residuals with the versioned observation covariance, and reports local Jacobian rank/conditioning. Fits remain candidate estimates and never auto-promote model parameters to validated.
- An end-to-end architecture comparison runner executes parameter-draw ensembles for multiple competing origin hypotheses, applies architecture-specific but outcome-compatible observation operators, and scores all forecasts on the same held-out time grid and observation mask.
- A timing-sweep engine tests the same explicit virtual perturbation across candidate start/end windows and reports component-wise damage prevented, damage-burden AUC reduction, quantile ranges and the fraction of supplied parameter draws with beneficial or harmful effects.
- The Alzheimer systems reverse-engineering dossier is in `docs/ALZHEIMER_SYSTEMS_REVERSE_ENGINEERING_V1.md`; it documents the decomposition, rival origin hypotheses, feedback loops, candidate intervention layers and validation requirements.

See `docs/NEURO_TWIN_SIX_DOMAIN_NETWORK_V1.md` for the mathematical and validation contract.

## Local development

### Backend

```bash
cd backend
python -m pytest -q
python -m neuro_twin.api_server --host 127.0.0.1 --port 8000
```

### Frontend

```bash
cd frontend-app
npm install
npm run dev
```

The Vite development server and FastAPI runtime are deliberately separate during development. Production builds can be served by the FastAPI bridge once `frontend-app/dist/` exists.

## Runtime API

- `GET /health`
- `GET /v1/capabilities`
- `GET /v1/runtime/status`
- `POST /v1/runtime/run`
- `POST /v1/runtime/publish`
- `GET /v1/trajectory`
- `GET /v1/evidence/{evidence_id}`
- `WS /ws`

`POST /v1/run/reference` is intentionally removed from the production execution path.

## GitHub

GitHub is the source of truth for source code and CI. Data-provider credentials, controlled clinical data and large imaging objects remain outside Git. Dataset snapshots are represented by immutable manifests and hashes.

The repository should be kept private until the scientific/data-governance review is complete.
