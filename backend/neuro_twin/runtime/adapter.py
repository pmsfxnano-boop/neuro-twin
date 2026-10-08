"""Runtime Engine Adapter: scientific result -> validated UI runtime payload.

The adapter is intentionally downstream of the scientific engine. It does not
perform discovery, feature extraction or clinical inference. It validates a
fully materialized result, propagates PET kinetic covariance through the
registered PET→Neuro-Twin linearization, computes a posterior state update,
and publishes an immutable evidence envelope to the bridge.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from neuro_twin.runtime.contracts import RuntimeClass, RuntimeResult


class RuntimePublicationError(ValueError):
    pass


@dataclass(frozen=True)
class PETAssimilationResult:
    observable_names: tuple[str, ...]
    effective_covariance: np.ndarray
    state_mean: np.ndarray
    state_covariance: np.ndarray
    innovation: np.ndarray
    innovation_covariance: np.ndarray
    kalman_gain: np.ndarray
    parameter_uncertainty_contribution: np.ndarray


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise RuntimePublicationError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_payload(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def assimilate_pet_to_state(result: RuntimeResult) -> PETAssimilationResult:
    """Propagate PET kinetic posterior covariance into the Neuro-Twin state posterior.

    Given z ~ N(h(x,theta), R_kinetic), parameter uncertainty contributes
    Htheta Sigma_theta Htheta^T to the effective observation covariance.
    The state update is the local Laplace/EKF Gaussian update around the
    declared reference state. No PET→P/I/N/Q mapping is invented here: the
    registered `pet_neuro_binding` supplies Hx and Htheta.
    """
    if result.pet_kinetic_posterior is None or result.pet_neuro_binding is None:
        raise RuntimePublicationError("PET kinetic posterior and PET→Neuro binding are both required")

    kin = result.pet_kinetic_posterior
    bind = result.pet_neuro_binding
    z = np.asarray(kin.mean, dtype=float)
    R_kin = np.asarray(kin.covariance, dtype=float)
    Hx = np.asarray(bind.jacobian_state, dtype=float)
    h0 = np.asarray(bind.predicted_at_reference, dtype=float)
    x_ref = np.asarray(bind.reference_state, dtype=float)

    if Hx.shape != (z.size, 4):
        raise RuntimePublicationError("PET state Jacobian shape mismatch")
    if h0.shape != z.shape:
        raise RuntimePublicationError("PET model prediction dimension mismatch")

    R_param = np.zeros_like(R_kin)
    if bind.jacobian_parameters is not None:
        if bind.parameter_mean is None or bind.parameter_covariance is None:
            raise RuntimePublicationError("parameter Jacobian requires parameter mean/covariance")
        Htheta = np.asarray(bind.jacobian_parameters, dtype=float)
        Sigma_theta = np.asarray(bind.parameter_covariance, dtype=float)
        R_param = Htheta @ Sigma_theta @ Htheta.T

    R_eff = R_kin + R_param
    R_eff = 0.5 * (R_eff + R_eff.T)
    P0 = np.asarray(result.state_prior.covariance, dtype=float)
    m0 = np.asarray(result.state_prior.mean, dtype=float)

    expected = h0 + Hx @ (m0 - x_ref)
    innovation = z - expected
    S = Hx @ P0 @ Hx.T + R_eff
    S = 0.5 * (S + S.T)
    K = P0 @ Hx.T @ np.linalg.pinv(S)
    I4 = np.eye(4)
    m = m0 + K @ innovation
    P = (I4 - K @ Hx) @ P0 @ (I4 - K @ Hx).T + K @ R_eff @ K.T
    P = 0.5 * (P + P.T)

    return PETAssimilationResult(
        observable_names=kin.names,
        effective_covariance=R_eff,
        state_mean=m,
        state_covariance=P,
        innovation=innovation,
        innovation_covariance=S,
        kalman_gain=K,
        parameter_uncertainty_contribution=R_param,
    )


def validate_runtime_result(result: RuntimeResult) -> None:
    """Reject runtime payloads that are not safe for scientific publication."""
    if result.runtime_class == RuntimeClass.SYNTHETIC_REFERENCE:
        raise RuntimePublicationError("synthetic/reference runs are forbidden on the production runtime path")
    as_of = _utc(result.pit.as_of)
    for obs in result.observations:
        acquisition = _utc(obs.acquisition_time)
        result_time = _utc(obs.result_time or obs.acquisition_time)
        ingest = _utc(obs.ingest_time)
        if acquisition > result_time or result_time > ingest:
            raise RuntimePublicationError(f"invalid PIT chronology for {obs.event_id}/{obs.feature}")
        if result_time > as_of:
            raise RuntimePublicationError(f"observation {obs.event_id}/{obs.feature} is future information relative to PIT cutoff")
        if not obs.provenance.raw_hash:
            raise RuntimePublicationError(f"observation {obs.event_id}/{obs.feature} has no raw hash")

    if result.oos is not None and result.oos.temporal_leakage:
        raise RuntimePublicationError("OOS gate reports temporal leakage")

    if result.provenance.result_hash is not None:
        # result_hash is an externally supplied content hash; recomputation is done
        # by the publisher before accepting the envelope.
        if len(result.provenance.result_hash) != 64:
            raise RuntimePublicationError("result_hash must be SHA-256 hex")

    if result.pet_kinetic_posterior is not None and result.pet_neuro_binding is None:
        raise RuntimePublicationError("PET posterior cannot enter the runtime without a registered PET→Neuro observation model")


def enrich_with_pet_posterior(result: RuntimeResult) -> RuntimeResult:
    if result.pet_kinetic_posterior is None:
        return result
    assim = assimilate_pet_to_state(result)
    state = result.state_posterior.model_copy(
        update={
            "mean": assim.state_mean.tolist(),
            "covariance": assim.state_covariance.tolist(),
        }
    )
    evidence = dict(result.evidence)
    evidence["pet_to_state_assimilation"] = {
        "observable_names": list(assim.observable_names),
        "effective_covariance": assim.effective_covariance.tolist(),
        "parameter_uncertainty_contribution": assim.parameter_uncertainty_contribution.tolist(),
        "innovation": assim.innovation.tolist(),
        "innovation_covariance": assim.innovation_covariance.tolist(),
        "kalman_gain": assim.kalman_gain.tolist(),
        "method": "local_gaussian_update",
        "assumptions": list(result.pet_neuro_binding.assumptions if result.pet_neuro_binding else ()),
    }
    return result.model_copy(update={"state_posterior": state, "evidence": evidence})


class RuntimeEngineAdapter:
    """File-backed publisher shared by the scientific runtime and FastAPI bridge."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.runtime_dir = self.root / "runtime"
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.current_path = self.runtime_dir / "current.json"

    def prepare(self, result: RuntimeResult) -> RuntimeResult:
        validate_runtime_result(result)
        enriched = enrich_with_pet_posterior(result)
        base = json.loads(enriched.model_dump_json())
        base["provenance"]["result_hash"] = None
        result_hash = sha256_payload(base)
        provenance = enriched.provenance.model_copy(update={"result_hash": result_hash})
        return enriched.model_copy(update={"provenance": provenance})

    def publish(self, result: RuntimeResult) -> dict[str, Any]:
        prepared = self.prepare(result)
        payload = json.loads(prepared.model_dump_json())
        payload["published_at"] = datetime.now(timezone.utc).isoformat()
        payload["publication_status"] = "PUBLISHED"
        self.current_path.write_text(_canonical_json(payload), encoding="utf-8")
        event_path = self.runtime_dir / f"{prepared.runtime_id}.json"
        event_path.write_text(_canonical_json(payload), encoding="utf-8")
        return payload

    def load_current(self) -> RuntimeResult | None:
        if not self.current_path.exists():
            return None
        payload = json.loads(self.current_path.read_text(encoding="utf-8"))
        payload.pop("published_at", None)
        payload.pop("publication_status", None)
        return RuntimeResult.model_validate(payload)

    def ui_snapshot(self, result: RuntimeResult) -> dict[str, Any]:
        state = result.state_posterior
        ci = state.ci95()
        metrics = {
            "observations": float(len(result.observations)),
            "pet_uncertainty_channels": float(len(result.pet_kinetic_posterior.names)) if result.pet_kinetic_posterior else 0.0,
            "state_trace": float(np.trace(np.asarray(state.covariance, dtype=float))),
            "oos_temporal_leakage": 1.0 if result.oos and result.oos.temporal_leakage else 0.0,
        }
        return {
            "runtime_id": result.runtime_id,
            "state_status": "LIVE_RESEARCH",
            "source": result.provenance.source_name,
            "subject_id": result.subject_id,
            "event_id": result.event_id,
            "time": state.time_months,
            "state": dict(zip(state.names, state.mean)),
            "ci": ci,
            "metrics": metrics,
            "provenance": result.provenance.model_dump(mode="json"),
            "pit": result.pit.model_dump(mode="json"),
            "oos": result.oos.model_dump(mode="json") if result.oos else None,
            "pet_kinetic_posterior": result.pet_kinetic_posterior.model_dump(mode="json") if result.pet_kinetic_posterior else None,
            "pet_to_state_assimilation": result.evidence.get("pet_to_state_assimilation"),
            "trajectory": result.trajectory,
            "prediction": result.prediction,
            "evidence": result.evidence,
            "evidence_id": result.runtime_id,
            "updated_at": time.time(),
        }

    def load_ui_snapshot(self) -> dict[str, Any] | None:
        current = self.load_current()
        if current is None:
            return None
        return self.ui_snapshot(current)
