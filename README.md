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
