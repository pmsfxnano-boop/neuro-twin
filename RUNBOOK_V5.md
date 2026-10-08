# NEURO-TWIN Frontier Web App v5 — Runtime Data Lab

## Purpose

This version changes the console from a read-only runtime viewer into a real web application that can **receive a user-supplied scientific run package and execute the P-I-N-Q engine** through the FastAPI runtime bridge.

The production path does not generate synthetic/reference results.

## Real execution path

```text
User data package
      ↓
Pydantic canonical observation validation
      ↓
Explicit versioned observation operator
      ↓
PIT chronology + provenance checks
      ↓
Temporal train/OOS split
      ↓
P-I-N-Q constrained fit
      ↓
Exact sensitivities + Laplace covariance
      ↓
State posterior / trajectory
      ↓
OOS metrics
      ↓
Optional PET kinetic posterior + explicit PET→Neuro binding
      ↓
PET covariance → state posterior propagation
      ↓
Runtime Engine Adapter
      ↓
Immutable runtime publication
      ↓
React/TypeScript + WebSocket UI
```

## Input

The UI accepts a JSON **runtime package** containing:

- canonical observations;
- provenance and raw hashes;
- a versioned observation operator;
- initial state and covariance;
- initial P-I-N-Q parameters;
- optional parameter bounds;
- PIT/OOS configuration;
- optional PET kinetic posterior;
- optional explicit PET→Neuro observation binding.

Download the schema from the application itself. Do not use a hidden/default biological mapping: the observation operator must be explicitly supplied and versioned.

## Endpoints

- `GET /health`
- `GET /v1/capabilities`
- `GET /v1/runtime/status`
- `POST /v1/runtime/run` — execute the scientific engine on user-supplied data.
- `POST /v1/runtime/publish` — publish an already materialized runtime result.
- `POST /v1/runtime/refresh`
- `GET /v1/trajectory`
- `GET /v1/evidence/{id}`
- `WS /ws`

`POST /v1/run/reference` is permanently disabled.

## Frontend build

The frontend is a React + TypeScript + Vite application.

```bash
cd frontend-app
npm install
npm run check
npm run build
```

The backend will serve `frontend-app/dist` from `/` once it exists.

## Backend run

```bash
cd backend
PYTHONPATH=. python -m neuro_twin.api_server --host 127.0.0.1 --port 8000
```

## Scientific status

The current executable web path runs the deterministic P-I-N-Q baseline with exact state/parameter sensitivities and local Laplace covariance, strict temporal OOS evaluation, and the existing Runtime Engine Adapter. Bayesian/NUTS remains capability-gated because NumPyro is not installed in the current environment.

The numerical fixtures used during development are test-only and are not bundled as a production data source.
