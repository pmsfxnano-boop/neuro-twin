"""Executable observational runtime: canonical observations -> P-I-N-Q -> RuntimeResult.

This module is deliberately separate from the UI bridge. It consumes canonical
observations and an explicit observation-model registry entry supplied by the
user/research pipeline. It never invents biomarker->latent-state mappings.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal
import hashlib

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator
from scipy.optimize import least_squares

from neuro_twin.core.dynamics import integrate_with_sensitivities, integrate_with_state_transition
from neuro_twin.core.observation import LinearObservationModel
from neuro_twin.core.parameters import PARAMETER_NAMES, PINQParameters
from neuro_twin.data.schema import Observation
from neuro_twin.model.observation_registry import ObservationModelConfig, ObservationSpec
from neuro_twin.uncertainty.laplace import laplace_covariance
from neuro_twin.disease.coverage import evaluate_module_coverage
from neuro_twin.disease.modules import get_disease_module
from neuro_twin.runtime.contracts import (
    OOSResult,
    PITEnvelope,
    PETKineticPosterior,
    PETNeuroBinding,
    PosteriorState,
    RuntimeClass,
    RuntimeProvenance,
    RuntimeResult,
)


class OperatorSpecIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    version: str
    specs: list[dict[str, Any]] = Field(min_length=1)
    correlation: list[list[float]] | None = None
    metadata: dict[str, str] | None = None


class RuntimeRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    access_tier: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    processing_pipeline: str = Field(min_length=1)
    processing_version: str = Field(min_length=1)
    retrieval_uri: str | None = None
    raw_hashes: list[str] = Field(default_factory=list)
    feature_hashes: list[str] = Field(default_factory=list)
    observations: list[Observation] = Field(min_length=1)
    operator: OperatorSpecIn
    initial_state: list[float] = Field(min_length=4, max_length=4)
    initial_state_covariance: list[list[float]]
    initial_theta: list[float] = Field(min_length=11, max_length=11)
    initial_theta_covariance: list[list[float]] | None = None
    theta_lower: list[float] | None = None
    theta_upper: list[float] | None = None
    x0_lower: list[float] | None = None
    x0_upper: list[float] | None = None
    oos_fraction: float = Field(default=0.8, gt=0.0, lt=1.0)
    analysis_mode: Literal["retrospective", "prospective_oos"] = "retrospective"
    pit_as_of: datetime | None = None
    model_input_raw_hash: str | None = None
    pet_kinetic_posterior: dict[str, Any] | None = None
    pet_neuro_binding: dict[str, Any] | None = None
    disease: str | None = None
    disease_module_version: str | None = None

    @field_validator("initial_state_covariance")
    @classmethod
    def cov4(cls, value: list[list[float]]) -> list[list[float]]:
        cov = np.asarray(value, dtype=float)
        if cov.shape != (4, 4):
            raise ValueError("initial_state_covariance must be 4x4")
        if np.linalg.eigvalsh(0.5 * (cov + cov.T)).min() < -1e-10:
            raise ValueError("initial_state_covariance must be PSD")
        return value

    @field_validator("initial_theta_covariance")
    @classmethod
    def cov11(cls, value: list[list[float]] | None) -> list[list[float]] | None:
        if value is None:
            return value
        cov = np.asarray(value, dtype=float)
        if cov.shape != (11, 11):
            raise ValueError("initial_theta_covariance must be 11x11")
        if np.linalg.eigvalsh(0.5 * (cov + cov.T)).min() < -1e-10:
            raise ValueError("initial_theta_covariance must be PSD")
        return value


@dataclass(frozen=True)
class Fitted:
    x0: np.ndarray
    theta: np.ndarray
    pred: np.ndarray
    jacobian: np.ndarray
    objective: float
    covariance: np.ndarray | None
    times: np.ndarray
    states: np.ndarray


def _months(dt: datetime, origin: datetime) -> float:
    return (dt.astimezone(timezone.utc) - origin.astimezone(timezone.utc)).total_seconds() / (30.4375 * 86400.0)


def _build_operator(spec: OperatorSpecIn) -> tuple[ObservationModelConfig, LinearObservationModel]:
    obs_specs = tuple(
        ObservationSpec(
            name=str(s["name"]),
            weights=tuple(float(x) for x in s["weights"]),
            bias=float(s.get("bias", 0.0)),
            sigma=float(s.get("sigma", 1.0)),
            group=s.get("group"),
        )
        for s in spec.specs
    )
    config = ObservationModelConfig(
        name=spec.name,
        version=spec.version,
        specs=obs_specs,
        correlation=tuple(tuple(float(x) for x in row) for row in spec.correlation) if spec.correlation is not None else None,
        metadata=spec.metadata,
    )
    return config, config.build()


def _pivot(observations: list[Observation], feature_names: tuple[str, ...]) -> tuple[np.ndarray, list[dict[str, Observation | None]]]:
    grouped: dict[datetime, dict[str, Observation]] = {}
    for obs in observations:
        if obs.feature not in feature_names or obs.missing:
            continue
        grouped.setdefault(obs.acquisition_time, {})[obs.feature] = obs
    times = sorted(grouped)
    rows: list[dict[str, Observation | None]] = []
    for t in times:
        rows.append({name: grouped[t].get(name) for name in feature_names})
    if len(times) < 2:
        raise ValueError("at least two distinct acquisition times are required")
    return np.asarray([t.timestamp() for t in times], dtype=float), rows


def _fit_irregular(
    times_abs: np.ndarray,
    rows: list[dict[str, Observation | None]],
    feature_names: tuple[str, ...],
    model: LinearObservationModel,
    initial_x0: np.ndarray,
    initial_theta: np.ndarray,
    *,
    theta_lower: np.ndarray | None,
    theta_upper: np.ndarray | None,
    x0_lower: np.ndarray | None,
    x0_upper: np.ndarray | None,
) -> Fitted:
    origin_seconds = times_abs[0]
    t = (times_abs - origin_seconds) / (30.4375 * 86400.0)
    x0 = np.asarray(initial_x0, dtype=float)
    th0 = np.asarray(initial_theta, dtype=float)
    z0 = np.concatenate([x0, th0])
    n = 15
    lb = np.full(n, -np.inf)
    ub = np.full(n, np.inf)
    if x0_lower is not None: lb[:4] = np.asarray(x0_lower, dtype=float)
    if x0_upper is not None: ub[:4] = np.asarray(x0_upper, dtype=float)
    if theta_lower is not None: lb[4:] = np.asarray(theta_lower, dtype=float)
    if theta_upper is not None: ub[4:] = np.asarray(theta_upper, dtype=float)
    eps = 1e-12
    z0 = np.minimum(np.maximum(z0, lb + eps), ub - eps)
    name_to_col = {n: i for i, n in enumerate(feature_names)}

    def evaluate(z: np.ndarray):
        xx = z[:4]
        theta = PINQParameters.from_array(z[4:])
        sens = integrate_with_sensitivities(xx, t, theta)
        _, phi = integrate_with_state_transition(xx, t, theta)
        residuals: list[float] = []
        jac_rows: list[np.ndarray] = []
        for ti, row in enumerate(rows):
            available = [i for i, name in enumerate(feature_names) if row[name] is not None]
            if not available:
                continue
            obs = np.asarray([float(row[feature_names[i]].value) for i in available], dtype=float)
            H = model.H[available]
            bias = model.bias[available]
            R = model.R[np.ix_(available, available)]
            pred = H @ sens.state[ti] + bias
            raw = pred - obs
            L = np.linalg.cholesky(R)
            white = np.linalg.solve(L, raw)
            residuals.extend(white.tolist())
            Jx0 = H @ phi[ti]
            Jth = np.einsum("ij,jk->ik", H, sens.sensitivity[ti])
            J = np.concatenate([Jx0, Jth], axis=1)
            jac_rows.append(np.linalg.solve(L, J))
        return np.asarray(residuals), np.vstack(jac_rows), sens.state, theta

    fun = lambda z: evaluate(z)[0]
    jac = lambda z: evaluate(z)[1]
    result = least_squares(fun, z0, jac=jac, bounds=(lb, ub), method="trf", x_scale="jac", loss="linear")
    residual, J, states, theta = evaluate(result.x)
    cov = laplace_covariance(J, objective=float(0.5 * residual @ residual), n_residuals=J.shape[0]).covariance
    # model prediction on all rows, preserving missingness as NaN
    pred_all = np.full((len(rows), len(feature_names)), np.nan, dtype=float)
    for ti, row in enumerate(rows):
        vals = model.predict(states[ti])
        for i, name in enumerate(feature_names):
            if row[name] is not None:
                pred_all[ti, i] = vals[i]
    return Fitted(result.x[:4], result.x[4:], pred_all, J, float(0.5 * residual @ residual), cov, t, states)


def _rmse_oos(
    fitted: Fitted,
    rows: list[dict[str, Observation | None]],
    feature_names: tuple[str, ...],
    model: LinearObservationModel,
    test_indices: np.ndarray,
) -> dict[str, float]:
    errs: list[float] = []
    abs_errs: list[float] = []
    for i in test_indices:
        pred = model.predict(fitted.states[i])
        for j, name in enumerate(feature_names):
            obs = rows[i].get(name)
            if obs is None:
                continue
            e = float(pred[j] - float(obs.value))
            errs.append(e * e)
            abs_errs.append(abs(e))
    if not errs:
        return {}
    return {"RMSE": float(np.sqrt(np.mean(errs))), "MAE": float(np.mean(abs_errs)), "n": float(len(errs))}


def execute_runtime(req: RuntimeRunRequest) -> RuntimeResult:
    if any(o.subject_id != req.subject_id for o in req.observations):
        raise ValueError("all observations in a runtime execution must belong to the declared subject_id")
    config, operator = _build_operator(req.operator)
    feature_names = operator.feature_names
    observed_features = {o.feature for o in req.observations if not o.missing}
    missing_features = [name for name in feature_names if name not in observed_features]
    disease_gate: dict[str, Any] | None = None
    if req.disease is not None or req.disease_module_version is not None:
        if req.disease is None or req.disease_module_version is None:
            raise ValueError("disease and disease_module_version must be provided together")
        module = get_disease_module(req.disease)
        if module.version != req.disease_module_version:
            raise ValueError(
                f"requested disease module version {req.disease_module_version!r} "
                f"does not match registered {module.disease.value}@{module.version}"
            )
        coverage = evaluate_module_coverage(module, feature_names)
        disease_gate = {
            "disease": module.disease.value,
            "module_version": module.version,
            "coverage": coverage.__dict__,
            "latent_interpretation": "ALLOWED" if coverage.status == "READY_FOR_FULL_OBSERVABILITY" else "BLOCKED",
            "diagnosis_target_external": True,
        }
    if len(observed_features) == 0:
        raise ValueError("no non-missing observations match the observation operator")

    times_abs, rows = _pivot(req.observations, feature_names)
    unique_n = len(times_abs)
    split_index = max(1, min(unique_n - 1, int(np.floor(unique_n * req.oos_fraction))))
    train_rows = rows[:split_index]
    train_times = times_abs[:split_index]
    train_cutoff = datetime.fromtimestamp(float(times_abs[split_index - 1]), tz=timezone.utc)
    published_observations = (
        [o for o in req.observations if o.acquisition_time <= train_cutoff]
        if req.analysis_mode == "prospective_oos"
        else list(req.observations)
    )
    if not published_observations:
        raise ValueError("no observations remain inside the publication PIT envelope")
    if len(train_times) < 2:
        raise ValueError("OOS split leaves fewer than two training time points")

    train_fit = _fit_irregular(
        train_times,
        train_rows,
        feature_names,
        operator,
        np.asarray(req.initial_state, dtype=float),
        np.asarray(req.initial_theta, dtype=float),
        theta_lower=np.asarray(req.theta_lower, dtype=float) if req.theta_lower is not None else None,
        theta_upper=np.asarray(req.theta_upper, dtype=float) if req.theta_upper is not None else None,
        x0_lower=np.asarray(req.x0_lower, dtype=float) if req.x0_lower is not None else None,
        x0_upper=np.asarray(req.x0_upper, dtype=float) if req.x0_upper is not None else None,
    )

    # Strict OOS prediction uses parameters/state inferred from train only.
    # Propagate from the end of training into all test times.
    test_t_months = (times_abs[split_index:] - times_abs[split_index - 1]) / (30.4375 * 86400.0)
    if len(test_t_months):
        t_from_train = np.concatenate([[0.0], test_t_months])
        pred_states = integrate_with_sensitivities(train_fit.states[-1], t_from_train, PINQParameters.from_array(train_fit.theta)).state[1:]
        full_states_for_oos = np.vstack([train_fit.states, pred_states])
    else:
        full_states_for_oos = train_fit.states
    oos_fit = Fitted(train_fit.x0, train_fit.theta, train_fit.pred, train_fit.jacobian, train_fit.objective, train_fit.covariance, (times_abs - times_abs[0]) / (30.4375 * 86400.0), full_states_for_oos)
    metrics = _rmse_oos(oos_fit, rows, feature_names, operator, np.arange(split_index, unique_n))

    # Final state posterior. Retrospective mode uses all available observations;
    # prospective_oos keeps the train-only fit so future observations are not
    # allowed to alter the current state estimate.
    if req.analysis_mode == "retrospective":
        final_fit = _fit_irregular(
            times_abs, rows, feature_names, operator,
            np.asarray(req.initial_state, dtype=float),
            np.asarray(req.initial_theta, dtype=float),
            theta_lower=np.asarray(req.theta_lower, dtype=float) if req.theta_lower is not None else None,
            theta_upper=np.asarray(req.theta_upper, dtype=float) if req.theta_upper is not None else None,
            x0_lower=np.asarray(req.x0_lower, dtype=float) if req.x0_lower is not None else None,
            x0_upper=np.asarray(req.x0_upper, dtype=float) if req.x0_upper is not None else None,
        )
        fit_origin = times_abs[-1]
        state_mean = final_fit.states[-1]
        J_final = np.concatenate([
            integrate_with_state_transition(final_fit.x0, final_fit.times, PINQParameters.from_array(final_fit.theta))[1][-1],
            integrate_with_sensitivities(final_fit.x0, final_fit.times, PINQParameters.from_array(final_fit.theta)).sensitivity[-1],
        ], axis=1)
        cov = final_fit.covariance
        pit_base = req.pit_as_of or datetime.fromtimestamp(float(times_abs[-1]), tz=timezone.utc)
    else:
        final_fit = train_fit
        fit_origin = times_abs[split_index - 1]
        state_mean = train_fit.states[-1]
        phi_final = integrate_with_state_transition(train_fit.x0, train_fit.times, PINQParameters.from_array(train_fit.theta))[1][-1]
        s_final = integrate_with_sensitivities(train_fit.x0, train_fit.times, PINQParameters.from_array(train_fit.theta)).sensitivity[-1]
        J_final = np.concatenate([phi_final, s_final], axis=1)
        cov = train_fit.covariance
        pit_base = req.pit_as_of or datetime.fromtimestamp(float(times_abs[split_index - 1]), tz=timezone.utc)

    if cov is None:
        raise RuntimeError("Laplace covariance unavailable; cannot construct state posterior")
    P_final = J_final @ cov @ J_final.T
    P_final = 0.5 * (P_final + P_final.T)
    # Guard only tiny negative numerical eigenvalues.
    vals, vecs = np.linalg.eigh(P_final)
    vals = np.maximum(vals, 0.0)
    P_final = (vecs * vals) @ vecs.T

    origin_dt = datetime.fromtimestamp(float(times_abs[0]), tz=timezone.utc)
    state_time_months = float((fit_origin - times_abs[0]) / (30.4375 * 86400.0))
    state_prior_cov = np.asarray(req.initial_state_covariance, dtype=float)
    state_prior = PosteriorState(mean=list(np.asarray(req.initial_state, dtype=float)), covariance=state_prior_cov.tolist(), time_months=0.0)
    state_post = PosteriorState(mean=state_mean.tolist(), covariance=P_final.tolist(), time_months=state_time_months)

    max_acq = max(o.acquisition_time for o in published_observations)
    max_result = max((o.result_time or o.acquisition_time) for o in published_observations)
    ingest = max(o.ingest_time for o in published_observations)
    as_of = pit_base
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    as_of = as_of.astimezone(timezone.utc)
    if max_result.astimezone(timezone.utc) > as_of:
        if req.analysis_mode == "prospective_oos":
            train_max_result = max((o.result_time or o.acquisition_time) for o in req.observations if o.acquisition_time <= datetime.fromtimestamp(float(fit_origin), tz=timezone.utc))
            as_of = train_max_result.astimezone(timezone.utc)
        else:
            raise ValueError("PIT cutoff precedes one or more observations used by the retrospective runtime")

    raw_hashes = sorted(set(req.raw_hashes + [h for h in [req.model_input_raw_hash] if h] + [o.provenance.raw_hash for o in published_observations if o.provenance.raw_hash]))
    feature_hashes = sorted(set(req.feature_hashes + [o.provenance.raw_hash for o in published_observations if o.observation_kind.value in {"mri_feature", "pet_feature"} and o.provenance.raw_hash]))
    pit = PITEnvelope(as_of=as_of, acquisition_max=max_acq, result_max=max_result, ingest_time=ingest)
    rid_seed = f"{req.subject_id}|{req.event_id}|{req.dataset_id}|{datetime.now(timezone.utc).isoformat()}".encode("utf-8")
    runtime_id = "rt_" + hashlib.sha256(rid_seed).hexdigest()[:24]
    provenance = RuntimeProvenance(
        source_name=req.source_name,
        dataset_id=req.dataset_id,
        dataset_version=req.dataset_version,
        model_version=f"{config.name}@{config.version}",
        processing_pipeline=req.processing_pipeline,
        processing_version=req.processing_version,
        raw_hashes=tuple(raw_hashes),
        feature_hashes=tuple(feature_hashes),
        observation_hash=req.model_input_raw_hash or (raw_hashes[0] if raw_hashes else config.config_hash),
        retrieval_uri=req.retrieval_uri,
        access_tier=req.access_tier,
    )

    oos = OOSResult(status="PASS" if metrics else "INCONCLUSIVE", temporal_leakage=False, metrics=metrics)
    trajectory_states = final_fit.states if req.analysis_mode == "retrospective" else full_states_for_oos
    trajectory = {
        "times": [float(x) for x in final_fit.times if x <= state_time_months + 1e-12],
        "states": {k: [float(row[i]) for row in trajectory_states[: len(final_fit.times)]] for i, k in enumerate(("P", "I", "N", "Q"))},
    }

    pet_post = PETKineticPosterior.model_validate(req.pet_kinetic_posterior) if req.pet_kinetic_posterior is not None else None
    binding = PETNeuroBinding.model_validate(req.pet_neuro_binding) if req.pet_neuro_binding is not None else None

    result = RuntimeResult(
        runtime_id=runtime_id,
        runtime_class=RuntimeClass.RESEARCH_OBSERVATIONAL,
        subject_id=req.subject_id,
        event_id=req.event_id,
        produced_at=datetime.now(timezone.utc),
        pit=pit,
        provenance=provenance,
        observations=published_observations,
        state_prior=state_prior,
        state_posterior=state_post,
        pet_kinetic_posterior=pet_post,
        pet_neuro_binding=binding,
        oos=oos,
        trajectory=trajectory,
        prediction={"mode": req.analysis_mode, "oos_split_index": split_index, "operator_config_hash": config.config_hash, "missing_operator_features": missing_features,
                    "disease_interpretation": disease_gate,
                },
        evidence={
            "run": {
                "fit_mode": req.analysis_mode,
                "operator_config_hash": config.config_hash,
                "train_time_count": split_index,
                "test_time_count": unique_n - split_index,
                "parameter_names": list(PARAMETER_NAMES),
                "parameter_mean": [float(x) for x in final_fit.theta],
                "parameter_covariance_available": final_fit.covariance is not None,
                "publication_observation_count": len(published_observations),
            },
            "disease_module": disease_gate,
            "oos": {
                "test_event_ids": sorted({o.event_id for o in req.observations if o not in published_observations}),
                "future_data_not_published_to_live_state": req.analysis_mode == "prospective_oos",
            },
        },
    )
    return result
