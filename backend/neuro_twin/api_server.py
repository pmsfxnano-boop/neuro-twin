"""FastAPI/WebSocket research bridge for real NEURO-TWIN runtime results.

The bridge never fabricates scientific state. Until a validated RuntimeResult
is published by the scientific pipeline, the UI remains in WAITING state.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from neuro_twin.runtime.adapter import RuntimeEngineAdapter, RuntimePublicationError
from neuro_twin.runtime.contracts import RuntimeResult
from neuro_twin.runtime.runner import RuntimeRunRequest, execute_runtime

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "evidence"
FRONTEND_DIR = ROOT.parent / "frontend"
REACT_DIST_DIR = ROOT.parent.parent / "frontend-app" / "dist"
PUBLIC_RUNTIME_PACKAGE = ROOT / "public_runtime" / "uci_parkinsons_subject1_v1.json"
runtime_adapter = RuntimeEngineAdapter(ROOT)


class RuntimeBus:
    def __init__(self) -> None:
        self.snapshot = runtime_adapter.load_ui_snapshot()
        self.clients: set[WebSocket] = set()
        self.lock = asyncio.Lock()

    async def publish_payload(self, snapshot: dict[str, Any]) -> None:
        snapshot = dict(snapshot)
        snapshot["updated_at"] = time.time()
        self.snapshot = snapshot
        message = json.dumps(snapshot, ensure_ascii=False)
        dead: list[WebSocket] = []
        async with self.lock:
            for ws in self.clients:
                try:
                    await ws.send_text(message)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.clients.discard(ws)


def _waiting_snapshot() -> dict[str, Any]:
    return {
        "runtime_id": None,
        "state_status": "WAITING_FOR_LIVE_RUNTIME",
        "source": None,
        "subject_id": None,
        "event_id": None,
        "time": None,
        "state": None,
        "ci": None,
        "metrics": {
            "observations": 0,
            "pet_uncertainty_channels": 0,
            "state_trace": None,
            "oos_temporal_leakage": 0,
        },
        "provenance": None,
        "pit": None,
        "oos": None,
        "pet_kinetic_posterior": None,
        "pet_to_state_assimilation": None,
        "trajectory": None,
        "prediction": None,
        "evidence_id": None,
        "updated_at": time.time(),
    }


def _load_evidence(evidence_id: str) -> dict[str, Any] | None:
    safe = evidence_id.replace("/", "_")
    runtime_event = ROOT / "runtime" / f"{safe}.json"
    candidates = [runtime_event, EVIDENCE_DIR / f"{safe}.json", EVIDENCE_DIR / f"{safe}_benchmark.json", EVIDENCE_DIR / f"{safe}_test_execution.json"]
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def create_app() -> FastAPI:
    app = FastAPI(title="NEURO-TWIN Research Bridge", version="0.3.0")
    cors_origins = [
        origin.strip()
        for origin in os.getenv("NEURO_TWIN_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )
    bus = RuntimeBus()
    app.state.runtime_bus = bus

    @app.on_event("startup")
    async def autoload_public_runtime() -> None:
        """Optionally materialize a real public-data runtime on service startup.

        The packaged source is a small derived excerpt of the UCI Parkinsons
        Telemonitoring dataset. It is real public data, not synthetic data.
        The loader is disabled unless explicitly enabled by environment.
        """
        enabled = os.getenv("NEURO_TWIN_AUTOLOAD_PUBLIC_RUNTIME", "0").strip().lower() in {"1", "true", "yes"}
        if not enabled or bus.snapshot is not None or not PUBLIC_RUNTIME_PACKAGE.exists():
            return
        try:
            payload = json.loads(PUBLIC_RUNTIME_PACKAGE.read_text(encoding="utf-8"))
            request = RuntimeRunRequest.model_validate(payload)
            result = execute_runtime(request)
            runtime_adapter.publish(result)
            current = runtime_adapter.load_current()
            snapshot = runtime_adapter.ui_snapshot(current) if current is not None else _waiting_snapshot()
            await bus.publish_payload(snapshot)
        except Exception as exc:
            # Startup must remain available even if the public-data adapter fails.
            print(f"NEURO_TWIN_AUTOLOAD_PUBLIC_RUNTIME failed: {exc}")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": "0.3.0",
            "engine": "neuro-twin",
            "bridge": "fastapi-websocket",
            "state_source": (bus.snapshot or _waiting_snapshot())["state_status"],
            "synthetic_reference": False,
        }

    @app.get("/")
    async def api_root() -> dict[str, Any]:
        return {
            "service": "neuro-twin-api",
            "status": "ok",
            "health": "/health",
            "runtime_status": "/v1/runtime/status",
            "capabilities": "/v1/capabilities",
        }

    @app.get("/v1/public/source")
    async def public_source() -> JSONResponse:
        if not PUBLIC_RUNTIME_PACKAGE.exists():
            raise HTTPException(status_code=404, detail="Public runtime package not installed")
        payload = json.loads(PUBLIC_RUNTIME_PACKAGE.read_text(encoding="utf-8"))
        return JSONResponse({
            "dataset_id": payload["dataset_id"],
            "dataset_version": payload["dataset_version"],
            "source_name": payload["source_name"],
            "source_version": payload["source_version"],
            "retrieval_uri": payload["retrieval_uri"],
            "subject_id": payload["subject_id"],
            "observation_count": len(payload["observations"]),
            "operator": payload["operator"],
            "real_public_data": True,
            "synthetic_reference": False,
        })

    @app.get("/v1/runtime/status")
    async def runtime_status() -> JSONResponse:
        return JSONResponse(bus.snapshot or _waiting_snapshot())

    @app.post("/v1/runtime/publish")
    async def runtime_publish(payload: dict[str, Any]) -> JSONResponse:
        try:
            result = RuntimeResult.model_validate(payload)
            prepared = runtime_adapter.publish(result)
            snapshot = runtime_adapter.ui_snapshot(RuntimeResult.model_validate({k: v for k, v in prepared.items() if k not in {"published_at", "publication_status"}}))
            await bus.publish_payload(snapshot)
            return JSONResponse({"status": "PUBLISHED", "runtime_id": result.runtime_id, "summary": snapshot})
        except RuntimePublicationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid runtime payload: {exc}") from exc

    @app.post("/v1/runtime/run")
    async def runtime_run(request: RuntimeRunRequest) -> JSONResponse:
        """Execute the scientific runtime on user-supplied canonical observations.

        This is the production data path: no synthetic/reference data is created
        here. The client supplies canonical observations plus an explicit
        versioned observation operator and initial conditions.
        """
        try:
            result = execute_runtime(request)
            prepared = runtime_adapter.publish(result)
            current = runtime_adapter.load_current()
            snapshot = runtime_adapter.ui_snapshot(current) if current is not None else _waiting_snapshot()
            await bus.publish_payload(snapshot)
            return JSONResponse({"status": "PUBLISHED", "runtime_id": result.runtime_id, "summary": snapshot})
        except (RuntimePublicationError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"runtime execution failed: {exc}") from exc

    @app.post("/v1/runtime/refresh")
    async def runtime_refresh() -> JSONResponse:
        current = runtime_adapter.load_current()
        if current is None:
            await bus.publish_payload(_waiting_snapshot())
            return JSONResponse({"status": "WAITING", "summary": bus.snapshot})
        snapshot = runtime_adapter.ui_snapshot(current)
        await bus.publish_payload(snapshot)
        return JSONResponse({"status": "PUBLISHED", "runtime_id": current.runtime_id, "summary": snapshot})

    @app.get("/v1/summary")
    async def summary() -> JSONResponse:
        return JSONResponse(bus.snapshot or _waiting_snapshot())

    @app.get("/v1/trajectory")
    async def trajectory() -> dict[str, Any]:
        snap = bus.snapshot
        if not snap or not snap.get("trajectory"):
            raise HTTPException(status_code=404, detail="No live trajectory has been published")
        return snap["trajectory"]

    @app.get("/v1/evidence/{evidence_id}")
    async def evidence(evidence_id: str) -> dict[str, Any]:
        record = _load_evidence(evidence_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Evidence record not found")
        return record

    @app.get("/v1/sources")
    async def sources() -> dict[str, Any]:
        path = ROOT / "neuro_twin" / "catalog" / "sources_v1.json"
        return json.loads(path.read_text(encoding="utf-8"))

    @app.post("/v1/run/reference")
    async def removed_reference_run() -> None:
        raise HTTPException(status_code=410, detail="Synthetic reference execution has been removed from the production runtime path")

    @app.get("/v1/capabilities")
    async def capabilities() -> dict[str, Any]:
        return {
            "state_space": True,
            "pinq": True,
            "mri": True,
            "pet_frame_integrated": True,
            "pet_aif_uncertainty": True,
            "pet_uncertainty_to_state_posterior": True,
            "pet_to_neuro_observation_registry_required": True,
            "hierarchical_map": True,
            "bayesian_runtime": False,
            "numpyro_available": False,
            "runtime_publish": True,
            "runtime_execute_user_data": True,
            "runtime_waiting_without_data": True,
            "websocket_stream": True,
            "synthetic_reference": False,
        }

    @app.websocket("/ws")
    async def websocket(ws: WebSocket) -> None:
        await ws.accept()
        async with bus.lock:
            bus.clients.add(ws)
        try:
            await ws.send_text(json.dumps(bus.snapshot or _waiting_snapshot(), ensure_ascii=False))
            while True:
                await asyncio.sleep(10.0)
                await ws.send_json({"type": "heartbeat", "t": time.time()})
        except WebSocketDisconnect:
            pass
        finally:
            async with bus.lock:
                bus.clients.discard(ws)

    active_frontend = REACT_DIST_DIR if REACT_DIST_DIR.exists() else FRONTEND_DIR
    if active_frontend.exists():
        app.mount("/console", StaticFiles(directory=active_frontend, html=True), name="console")

        @app.get("/")
        async def root() -> FileResponse:
            return FileResponse(active_frontend / "index.html")

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run("neuro_twin.api_server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
