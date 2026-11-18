"""Explicit observation operators for mechanism-level Alzheimer models.

No biomarker-to-mechanism mapping is inferred. Features must already be
transformed into declared model-compatible units by a versioned source adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from neuro_twin.network.mechanistic_dynamics import MechanismTrajectory


def _validate_names(values: tuple[str, ...], *, name: str) -> tuple[str, ...]:
    result = tuple(values)
    if not result:
        raise ValueError(f"{name} must contain at least one item")
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique values")
    return result


@dataclass(frozen=True)
class MechanismObservationPrediction:
    time: np.ndarray
    feature_names: tuple[str, ...]
    predictive_mean: np.ndarray
    observation_covariance: np.ndarray
    operator_hash: str


@dataclass(frozen=True)
class MechanismForecastEnsemble:
    architecture_id: str
    time: np.ndarray
    feature_names: tuple[str, ...]
    predictive_mean: np.ndarray
    predictive_sd: np.ndarray
    operator_hash: str
    parameter_draw_count: int


@dataclass(frozen=True)
class MechanismObservationModel:
    """Versioned affine observation operator y = H z + bias + error."""

    model_id: str
    version: str
    mechanism_node_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_units: tuple[str, ...]
    H: np.ndarray
    bias: np.ndarray
    R: np.ndarray
    source_catalog_hash: str
    transformation_id: str
    transformation_version: str

    def __post_init__(self) -> None:
        if not self.model_id.strip() or not self.version.strip():
            raise ValueError("model_id and version must be non-empty")
        if not self.source_catalog_hash.strip():
            raise ValueError("source_catalog_hash must be non-empty")
        if not self.transformation_id.strip() or not self.transformation_version.strip():
            raise ValueError("transformation identity and version are required")
        nodes = _validate_names(self.mechanism_node_ids, name="mechanism_node_ids")
        features = _validate_names(self.feature_names, name="feature_names")
        units = tuple(self.feature_units)
        if len(units) != len(features) or any(
            not isinstance(unit, str) or not unit.strip() for unit in units
        ):
            raise ValueError("feature_units must contain one non-empty unit per feature")
        H = np.array(self.H, dtype=float, copy=True)
        bias = np.array(self.bias, dtype=float, copy=True)
        R = np.array(self.R, dtype=float, copy=True)
        if H.shape != (len(features), len(nodes)):
            raise ValueError("H must have shape (n_features, n_mechanism_nodes)")
        if bias.shape != (len(features),):
            raise ValueError("bias must have shape (n_features,)")
        if R.shape != (len(features), len(features)):
            raise ValueError("R must have shape (n_features, n_features)")
        if not np.all(np.isfinite(H)) or not np.all(np.isfinite(bias)):
            raise ValueError("H and bias must be finite")
        if not np.all(np.isfinite(R)) or not np.allclose(R, R.T, atol=1e-10):
            raise ValueError("R must be finite and symmetric")
        try:
            np.linalg.cholesky(R)
        except np.linalg.LinAlgError as exc:
            raise ValueError("R must be positive definite") from exc
        for array in (H, bias, R):
            array.setflags(write=False)
        object.__setattr__(self, "mechanism_node_ids", nodes)
        object.__setattr__(self, "feature_names", features)
        object.__setattr__(self, "feature_units", units)
        object.__setattr__(self, "H", H)
        object.__setattr__(self, "bias", bias)
        object.__setattr__(self, "R", R)

    @property
    def config_hash(self) -> str:
        payload = {
            "model_id": self.model_id,
            "version": self.version,
            "mechanism_node_ids": list(self.mechanism_node_ids),
            "feature_names": list(self.feature_names),
            "feature_units": list(self.feature_units),
            "H": self.H.tolist(),
            "bias": self.bias.tolist(),
            "R": self.R.tolist(),
            "source_catalog_hash": self.source_catalog_hash,
            "transformation_id": self.transformation_id,
            "transformation_version": self.transformation_version,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def predict(
        self, trajectory: MechanismTrajectory
    ) -> MechanismObservationPrediction:
        if trajectory.node_ids != self.mechanism_node_ids:
            raise ValueError(
                "trajectory node order must exactly match the observation model"
            )
        mean = trajectory.activity @ self.H.T + self.bias
        return MechanismObservationPrediction(
            time=trajectory.time.copy(),
            feature_names=self.feature_names,
            predictive_mean=mean,
            observation_covariance=self.R.copy(),
            operator_hash=self.config_hash,
        )


def summarize_mechanism_forecast_ensemble(
    *,
    architecture_id: str,
    trajectories: tuple[MechanismTrajectory, ...],
    observation_model: MechanismObservationModel,
) -> MechanismForecastEnsemble:
    """Convert parameter-draw trajectories into a held-out forecast distribution.

    At least two supplied trajectories are required. predictive_sd captures
    only between-draw parameter variation; observation noise remains in R and
    should be passed separately as measurement_sd to model comparison.
    """
    if not architecture_id.strip():
        raise ValueError("architecture_id must be non-empty")
    if len(trajectories) < 2:
        raise ValueError("at least two parameter-draw trajectories are required")
    predictions = [
        observation_model.predict(trajectory) for trajectory in trajectories
    ]
    first = predictions[0]
    for trajectory, prediction in zip(trajectories[1:], predictions[1:]):
        if not np.array_equal(trajectory.time, trajectories[0].time):
            raise ValueError("all parameter-draw trajectories must share identical times")
        if prediction.feature_names != first.feature_names:
            raise ValueError("all parameter-draw predictions must share feature order")
        if prediction.operator_hash != first.operator_hash:
            raise ValueError("all parameter-draw predictions must share one operator")
    draws = np.stack([item.predictive_mean for item in predictions], axis=0)
    return MechanismForecastEnsemble(
        architecture_id=architecture_id,
        time=first.time.copy(),
        feature_names=first.feature_names,
        predictive_mean=np.mean(draws, axis=0),
        predictive_sd=np.std(draws, axis=0, ddof=1),
        operator_hash=first.operator_hash,
        parameter_draw_count=len(predictions),
    )
