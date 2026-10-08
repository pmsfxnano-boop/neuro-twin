# Render — NEURO-TWIN production topology

The root `render.yaml` now defines two independent services:

- `neuro-twin-api`: FastAPI scientific runtime (`backend/`).
- `neuro-twin-frontend`: React/Vite static site (`frontend-app/`).

The frontend is deployed as a static site through Render's CDN; it is not the
scientific runtime. The frontend must receive `VITE_NEURO_TWIN_API_URL` at build
time. The API must receive `NEURO_TWIN_CORS_ORIGINS` with the final frontend
origin.

## Deployment order

1. Push this repository to the dedicated GitHub repository `neuro-twin`.
2. Create the Render Blueprint from the root `render.yaml`.
3. Set `NEURO_TWIN_CORS_ORIGINS` on the API to the frontend's final HTTPS origin.
4. Set `VITE_NEURO_TWIN_API_URL` on the frontend to the API's final HTTPS origin.
5. Deploy both services.
6. Verify `GET /health`, `GET /v1/runtime/status`, and the WebSocket `/ws` before
   accepting research workloads.

## Scientific release rule

No RuntimeResult JSON is committed to source control. The UI must enter
`WAITING_FOR_LIVE_RUNTIME` until a real runtime publishes a validated result.
Synthetic/reference execution is disabled on the production API path.
