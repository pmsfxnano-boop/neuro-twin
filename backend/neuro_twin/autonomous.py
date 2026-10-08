"""Backend-owned autonomous research/learning loop.

This module is intentionally independent of ChatGPT/notifications.  It retrieves
real public scientific data, derives a versioned population parameter prior from
other subjects, and uses that learned prior to execute the live subject runtime.
All outputs are persisted as backend evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any

import httpx
import numpy as np

from neuro_twin.runtime.adapter import RuntimeEngineAdapter
from neuro_twin.runtime.runner import RuntimeRunRequest, execute_runtime

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/189/parkinsons%2Btelemonitoring.zip"
DATASET_ID = "uci-parkinsons-telemonitoring-189"
DATASET_VERSION = "DOI-10.24432/C5ZS3N"
SOURCE_NAME = "UCI Parkinsons Telemonitoring"
SOURCE_VERSION_BASE = "UCI-189/DOI-10.24432/C5ZS3N"
LICENSE = "CC BY 4.0"
PROCESSING_PIPELINE = "neuro-twin-autonomous-uci-adapter"
PROCESSING_VERSION = "0.3.0"
LIVE_SUBJECT_ID = "1"
RUNTIME_ARTIFACT_NAME = "autonomous_latest_runtime.json"
STATUS_ARTIFACT_NAME = "autonomous_learning.json"
HISTORY_ARTIFACT_NAME = "autonomous_learning_history.jsonl"
FEATURE_NAMES = ("motor_updrs_norm", "total_updrs_norm")
ANCHOR = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _runtime_dir(root: Path) -> Path:
    d = root / "runtime"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _status_path(root: Path) -> Path:
    return _runtime_dir(root) / "autonomous_learning.json"


def _history_path(root: Path) -> Path:
    return _runtime_dir(root) / "autonomous_learning_history.jsonl"


def _load_status(root: Path) -> dict[str, Any]:
    path = _status_path(root)
    if not path.exists():
        return {"status": "NEVER_RUN"}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_status(root: Path, status: dict[str, Any]) -> None:
    _status_path(root).write_text(
        json.dumps(status, sort_keys=True, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _append_history(root: Path, record: dict[str, Any]) -> None:
    with _history_path(root).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def fetch_uci_dataset() -> tuple[bytes, list[dict[str, str]], str]:
    with httpx.Client(timeout=60.0, follow_redirects=True, headers={"User-Agent": "NEURO-TWIN/0.2"}) as client:
        response = client.get(UCI_ZIP_URL)
        response.raise_for_status()
        raw = response.content

    raw_hash = hashlib.sha256(raw).hexdigest()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        candidates = [name for name in zf.namelist() if name.endswith("parkinsons_updrs.data")]
        if not candidates:
            raise RuntimeError("UCI archive does not contain parkinsons_updrs.data")
        payload = zf.read(candidates[0])

    reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
    rows = list(reader)
    required = {"subject#", "test_time", "motor_UPDRS", "total_UPDRS"}
    if not required.issubset(set(reader.fieldnames or [])):
        raise RuntimeError("UCI source schema is missing required fields")
    if len(rows) != 5875:
        raise RuntimeError(f"unexpected UCI row count: {len(rows)}")
    return raw, rows, raw_hash


def _observation_hash(rows: list[dict[str, str]], subject_id: str, raw_hash: str) -> str:
    payload = "\n".join(
        f"{r['subject#']}|{r['test_time']}|{r['motor_UPDRS']}|{r['total_UPDRS']}"
        for r in rows
        if r["subject#"] == subject_id
    ).encode("utf-8")
    return hashlib.sha256(raw_hash.encode("ascii") + b"|" + payload).hexdigest()


def _to_observations(
    rows: list[dict[str, str]],
    subject_id: str,
    raw_hash: str,
) -> list[dict[str, Any]]:
    subject_rows = [r for r in rows if r["subject#"] == subject_id]
    subject_rows.sort(key=lambda r: float(r["test_time"]))
    obs: list[dict[str, Any]] = []
    feature_hash = _observation_hash(rows, subject_id, raw_hash)
    source_version = f"{SOURCE_VERSION_BASE};raw_sha256={raw_hash}"
    for idx, row in enumerate(subject_rows, start=1):
        ts = ANCHOR + timedelta(days=float(row["test_time"]))
        iso = ts.isoformat().replace("+00:00", "Z")
        event_id = f"uci189-sub{subject_id}-t{idx:04d}"
        common = {
            "subject_id": subject_id,
            "event_id": event_id,
            "acquisition_time": iso,
            "result_time": iso,
            "ingest_time": iso,
            "source": SOURCE_NAME,
            "center": "Oxford Parkinson's Disease Telemonitoring Trial",
            "modality": "voice_telemonitoring",
            "assay": "UPDRS-derived longitudinal endpoint",
            "unit": "normalized_100",
            "uncertainty": 0.03,
            "missing": False,
            "quality": {
                "schema_valid": True,
                "temporal_integrity": True,
                "unit_consistent": True,
                "assay_consistent": True,
                "center_effect_flag": False,
                "scanner_effect_flag": False,
                "outlier_flag": False,
                "missingness_flag": False,
                "provenance_complete": True,
            },
            "provenance": {
                "source_name": SOURCE_NAME,
                "source_version": source_version,
                "source_record_id": f"subject={subject_id};test_time={row['test_time']}",
                "retrieval_uri": UCI_ZIP_URL,
                "raw_hash": raw_hash,
                "processing_pipeline": PROCESSING_PIPELINE,
                "processing_version": PROCESSING_VERSION,
                "license": LICENSE,
                "access_tier": "public_dataset",
            },
            "extra": {
                "temporal_anchor": "1970-01-01T00:00:00Z + test_time_days; computational ordering anchor, not claimed original calendar acquisition time",
                "raw_row": int(idx),
            },
        }
        motor = dict(common)
        motor.update({
            "observation_kind": "clinical",
            "feature": FEATURE_NAMES[0],
            "value": float(row["motor_UPDRS"]) / 100.0,
            "extra": {**common["extra"], "raw_feature": "motor_UPDRS", "raw_value": float(row["motor_UPDRS"]), "feature_hash": feature_hash},
        })
        total = dict(common)
        total.update({
            "observation_kind": "clinical",
            "feature": FEATURE_NAMES[1],
            "value": float(row["total_UPDRS"]) / 100.0,
            "extra": {**common["extra"], "raw_feature": "total_UPDRS", "raw_value": float(row["total_UPDRS"]), "feature_hash": feature_hash},
        })
        obs.extend([motor, total])
    return obs


def _package_for_subject(
    rows: list[dict[str, str]],
    subject_id: str,
    raw_hash: str,
    theta_seed: list[float],
    analysis_mode: str = "prospective_oos",
) -> RuntimeRunRequest:
    obs = _to_observations(rows, subject_id, raw_hash)
    first_motor = next(o for o in obs if o["feature"] == FEATURE_NAMES[0])
    first_total = next(o for o in obs if o["feature"] == FEATURE_NAMES[1])
    operator = {
        "name": "uci_pd_severity_projection",
        "version": "0.2.0",
        "metadata": {
            "interpretation": "research-only coordinate projection; P/Q are model coordinates and are not clinically validated biological labels",
            "mapping": "motor_UPDRS/100 -> P; total_UPDRS/100 -> Q",
            "unobserved_axes": "I,N",
            "normalization": "each source endpoint divided by 100",
            "autonomous_learning": "population parameter prior learned from other subjects",
        },
        "specs": [
            {"name": FEATURE_NAMES[0], "weights": [1, 0, 0, 0], "bias": 0, "sigma": 0.03},
            {"name": FEATURE_NAMES[1], "weights": [0, 0, 0, 1], "bias": 0, "sigma": 0.03},
        ],
    }
    return RuntimeRunRequest.model_validate({
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "source_name": SOURCE_NAME,
        "source_version": f"{SOURCE_VERSION_BASE};raw_sha256={raw_hash}",
        "access_tier": "public_dataset",
        "subject_id": subject_id,
        "event_id": f"uci189-sub{subject_id}-autonomous",
        "processing_pipeline": PROCESSING_PIPELINE,
        "processing_version": PROCESSING_VERSION,
        "retrieval_uri": UCI_ZIP_URL,
        "raw_hashes": [raw_hash],
        "feature_hashes": [first_motor["extra"]["feature_hash"], first_total["extra"]["feature_hash"]],
        "observations": obs,
        "operator": operator,
        "initial_state": [float(first_motor["value"]), 0.0, 0.0, float(first_total["value"])],
        "initial_state_covariance": [
            [0.0025, 0, 0, 0],
            [0, 0.01, 0, 0],
            [0, 0, 0.01, 0],
            [0, 0, 0, 0.0025],
        ],
        "initial_theta": theta_seed,
        "initial_theta_covariance": None,
        "theta_lower": [0.0] * 11,
        "theta_upper": [2.0] * 11,
        "x0_lower": [0.0] * 4,
        "x0_upper": [2.0] * 4,
        "oos_fraction": 0.8,
        "analysis_mode": analysis_mode,
        "pit_as_of": None,
        "model_input_raw_hash": raw_hash,
        "pet_kinetic_posterior": None,
        "pet_neuro_binding": None,
    })


def _fit_population_prior(
    rows: list[dict[str, str]],
    raw_hash: str,
    excluded_subject: str,
) -> tuple[list[float], dict[str, Any]]:
    subject_ids = sorted({r["subject#"] for r in rows if r["subject#"] != excluded_subject}, key=int)
    estimates: list[list[float]] = []
    metrics: list[float] = []
    failures: list[str] = []
    zero = [0.0] * 11

    for position, subject_id in enumerate(subject_ids, start=1):
        try:
            print(f"NEURO_TWIN_AUTONOMOUS_FIT subject={subject_id} progress={position}/{len(subject_ids)}", flush=True)
            request = _package_for_subject(rows, subject_id, raw_hash, zero, "prospective_oos")
            result = execute_runtime(request)
            theta = result.evidence.get("run", {}).get("parameter_mean")
            if isinstance(theta, list) and len(theta) == 11 and all(np.isfinite(theta)):
                estimates.append([float(x) for x in theta])
                if result.oos and result.oos.metrics.get("RMSE") is not None:
                    metrics.append(float(result.oos.metrics["RMSE"]))
            else:
                failures.append(subject_id)
        except Exception:
            failures.append(subject_id)

    if not estimates:
        raise RuntimeError("population learner produced no valid parameter estimates")

    matrix = np.asarray(estimates, dtype=float)
    prior = np.median(matrix, axis=0)
    prior = np.clip(prior, 0.0, 2.0)
    summary = {
        "excluded_subject": excluded_subject,
        "training_subjects": subject_ids,
        "successful_subjects": len(estimates),
        "failed_subjects": failures,
        "parameter_mean": prior.tolist(),
        "parameter_mad": (np.median(np.abs(matrix - prior), axis=0)).tolist(),
        "median_subject_oos_rmse": float(median(metrics)) if metrics else None,
    }
    return prior.tolist(), summary


def run_autonomous_cycle(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    started = time.time()
    current = _load_status(root)

    try:
        _, rows, raw_hash = fetch_uci_dataset()
        prior = current.get("learned_parameter_prior")
        prior_summary: dict[str, Any] | None = None

        if current.get("source_raw_sha256") != raw_hash or not isinstance(prior, list) or len(prior) != 11:
            prior, prior_summary = _fit_population_prior(rows, raw_hash, LIVE_SUBJECT_ID)

        live_request = _package_for_subject(rows, LIVE_SUBJECT_ID, raw_hash, prior, "prospective_oos")
        result = execute_runtime(live_request)
        prepared = RuntimeEngineAdapter(root).prepare(result)
        runtime_payload = json.loads(prepared.model_dump_json())
        artifact_envelope = {
            "schema_version": "neuro-twin.autonomous-runtime.v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "code_revision": os.getenv("GITHUB_SHA", "local"),
            "runtime": runtime_payload,
        }
        _runtime_artifact_path(root).write_text(
            json.dumps(artifact_envelope, sort_keys=True, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        snapshot = {
            "status": "PASS",
            "cycle_completed_at": datetime.now(timezone.utc).isoformat(),
            "cycle_seconds": round(time.time() - started, 3),
            "source_name": SOURCE_NAME,
            "dataset_id": DATASET_ID,
            "dataset_version": DATASET_VERSION,
            "source_raw_sha256": raw_hash,
            "source_row_count": len(rows),
            "live_subject_id": LIVE_SUBJECT_ID,
            "live_observation_count": len(result.observations),
            "live_runtime_id": result.runtime_id,
            "runtime_result_hash": prepared.provenance.result_hash,
            "code_revision": os.getenv("GITHUB_SHA", "local"),
            "live_oos": result.oos.model_dump(mode="json") if result.oos else None,
            "live_parameter_mean": result.evidence.get("run", {}).get("parameter_mean"),
            "learned_parameter_prior": prior,
            "population_learning": prior_summary or current.get("population_learning"),
            "learning_mode": "leave-one-subject-out population parameter prior",
            "synthetic_reference": False,
        }
        _write_status(root, snapshot)
        _append_history(root, snapshot)
        return {"status": "PASS", "runtime_id": result.runtime_id, "snapshot": snapshot}
    except Exception as exc:
        failure = {
            "status": "ERROR",
            "cycle_completed_at": datetime.now(timezone.utc).isoformat(),
            "cycle_seconds": round(time.time() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
            "source_raw_sha256": current.get("source_raw_sha256"),
            "live_runtime_id": current.get("live_runtime_id"),
        }
        _write_status(root, failure)
        _append_history(root, failure)
        raise


def autonomous_interval_seconds() -> int:
    return max(3600, int(os.getenv("NEURO_TWIN_AUTONOMOUS_INTERVAL_SECONDS", "21600")))


def autonomous_enabled() -> bool:
    return os.getenv("NEURO_TWIN_AUTONOMOUS_ENABLED", "0").strip().lower() in {"1", "true", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one NEURO-TWIN autonomous scientific learning cycle.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Backend root used for runtime/evidence artifacts.")
    args = parser.parse_args()
    outcome = run_autonomous_cycle(args.root)
    print(json.dumps(outcome, sort_keys=True, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
