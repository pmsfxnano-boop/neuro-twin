"""Out-of-sample predictive comparison of competing Alzheimer architectures.

Scores must be computed on a declared held-out evaluation set. This module
does not perform causal discovery or Bayesian model evidence integration.
It compares supplied predictive distributions on identical observed targets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


HoldoutKind = Literal["subject_holdout", "temporal_holdout", "external_cohort"]


@dataclass(frozen=True)
class ArchitectureForecast:
    architecture_id: str
    predictive_mean: np.ndarray
    predictive_sd: np.ndarray


@dataclass(frozen=True)
class ArchitecturePredictiveScore:
    architecture_id: str
    mean_log_predictive_density: float
    rmse: float
    coverage_50: float
    coverage_90: float
    n_scored: int
    rank: int


@dataclass(frozen=True)
class ArchitectureComparisonReport:
    evaluation_id: str
    holdout_kind: HoldoutKind
    scored_mask: np.ndarray
    scores: tuple[ArchitecturePredictiveScore, ...]
    ranking_metric: Literal["mean_log_predictive_density"] = "mean_log_predictive_density"
    interpretation: Literal[
        "held_out_predictive_comparison_not_causal_model_confirmation"
    ] = "held_out_predictive_comparison_not_causal_model_confirmation"


def compare_architecture_forecasts(
    *,
    evaluation_id: str,
    holdout_kind: HoldoutKind,
    observed: np.ndarray,
    measurement_sd: np.ndarray,
    available_mask: np.ndarray,
    forecasts: tuple[ArchitectureForecast, ...],
) -> ArchitectureComparisonReport:
    """Rank forecasts by Gaussian log score on a common held-out observation mask.

    Each forecast must come from a fully specified architecture model and a
    versioned observation operator. predictive_sd is the model predictive
    standard deviation; measurement_sd is the observation-level standard
    uncertainty. They are combined in quadrature. This is a predictive
    comparison only: the best score does not establish a causal origin.
    """
    if not evaluation_id.strip():
        raise ValueError("evaluation_id must be non-empty")
    if holdout_kind not in ("subject_holdout", "temporal_holdout", "external_cohort"):
        raise ValueError("holdout_kind must describe held-out evaluation data")
    if len(forecasts) < 2:
        raise ValueError("at least two architecture forecasts are required")
    ids = tuple(item.architecture_id for item in forecasts)
    if any(not item.strip() for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("architecture forecasts must have unique non-empty IDs")

    y = np.asarray(observed, dtype=float)
    obs_sd = np.asarray(measurement_sd, dtype=float)
    mask = np.asarray(available_mask)
    if y.ndim == 0 or y.size == 0:
        raise ValueError("observed must be a non-empty array")
    if obs_sd.shape != y.shape or mask.shape != y.shape:
        raise ValueError("measurement_sd and available_mask must match observed shape")
    if mask.dtype != np.bool_:
        raise ValueError("available_mask must be boolean")
    if not np.any(mask):
        raise ValueError("at least one held-out observation must be available")
    if not np.all(np.isfinite(y[mask])):
        raise ValueError("available observed values must be finite")
    if not np.all(np.isfinite(obs_sd[mask])) or np.any(obs_sd[mask] <= 0.0):
        raise ValueError("available measurement_sd values must be finite and > 0")

    raw_scores: list[tuple[str, float, float, float, float, int]] = []
    z50 = 0.6744897501960817
    z90 = 1.6448536269514722
    for forecast in forecasts:
        mean = np.asarray(forecast.predictive_mean, dtype=float)
        pred_sd = np.asarray(forecast.predictive_sd, dtype=float)
        if mean.shape != y.shape or pred_sd.shape != y.shape:
            raise ValueError(
                f"forecast {forecast.architecture_id} arrays must match observed shape"
            )
        if not np.all(np.isfinite(mean[mask])):
            raise ValueError(f"forecast {forecast.architecture_id} has non-finite means")
        if not np.all(np.isfinite(pred_sd[mask])) or np.any(pred_sd[mask] < 0.0):
            raise ValueError(f"forecast {forecast.architecture_id} predictive_sd must be >= 0")

        total_sd = np.sqrt(pred_sd[mask] ** 2 + obs_sd[mask] ** 2)
        residual = y[mask] - mean[mask]
        log_density = (
            -0.5 * np.log(2.0 * np.pi)
            - np.log(total_sd)
            - 0.5 * (residual / total_sd) ** 2
        )
        mean_lpd = float(np.mean(log_density))
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        coverage_50 = float(np.mean(np.abs(residual) <= z50 * total_sd))
        coverage_90 = float(np.mean(np.abs(residual) <= z90 * total_sd))
        raw_scores.append(
            (forecast.architecture_id, mean_lpd, rmse, coverage_50, coverage_90, int(mask.sum()))
        )

    ordered = sorted(raw_scores, key=lambda row: (-row[1], row[2], row[0]))
    rank_by_id = {row[0]: index + 1 for index, row in enumerate(ordered)}
    scores = tuple(
        ArchitecturePredictiveScore(
            architecture_id=row[0],
            mean_log_predictive_density=row[1],
            rmse=row[2],
            coverage_50=row[3],
            coverage_90=row[4],
            n_scored=row[5],
            rank=rank_by_id[row[0]],
        )
        for row in ordered
    )
    return ArchitectureComparisonReport(
        evaluation_id=evaluation_id,
        holdout_kind=holdout_kind,
        scored_mask=mask.copy(),
        scores=scores,
    )
