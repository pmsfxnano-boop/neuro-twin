"""Strict cohort-external validation utilities."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CohortSplit:
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_cohorts: tuple[str, ...]
    test_cohort: str


def leave_one_cohort_out(cohorts: np.ndarray | list[str], held_out: str) -> CohortSplit:
    c = np.asarray(cohorts, dtype=str)
    if c.ndim != 1 or not len(c):
        raise ValueError("cohorts must be a non-empty 1-D array")
    if held_out not in set(c.tolist()):
        raise ValueError("held_out cohort is not present")
    test = np.flatnonzero(c == held_out)
    train = np.flatnonzero(c != held_out)
    if not len(train) or not len(test):
        raise ValueError("external validation requires non-empty train and test sets")
    return CohortSplit(train, test, tuple(sorted(set(c[train].tolist()))), held_out)


@dataclass(frozen=True)
class PredictiveMetrics:
    n: int
    rmse: float
    mae: float
    bias: float
    coverage_95: float | None
    mean_log_score: float | None


def predictive_metrics(observed: np.ndarray, predicted_mean: np.ndarray, predicted_variance: np.ndarray | None = None) -> PredictiveMetrics:
    y = np.asarray(observed, float).ravel()
    mu = np.asarray(predicted_mean, float).ravel()
    if y.shape != mu.shape:
        raise ValueError("observed and predicted_mean shapes differ")
    mask = np.isfinite(y) & np.isfinite(mu)
    y, mu = y[mask], mu[mask]
    if not len(y):
        raise ValueError("no finite predictions")
    err = mu - y
    coverage = None
    log_score = None
    if predicted_variance is not None:
        var = np.asarray(predicted_variance, float).ravel()
        if var.shape != mask.shape:
            raise ValueError("predicted_variance shape mismatch")
        var = np.maximum(var[mask], 1e-12)
        sd = np.sqrt(var)
        coverage = float(np.mean(np.abs(err) <= 1.96 * sd))
        log_score = float(np.mean(0.5 * (np.log(2*np.pi*var) + (err * err) / var)))
    return PredictiveMetrics(
        n=int(len(y)),
        rmse=float(np.sqrt(np.mean(err * err))),
        mae=float(np.mean(np.abs(err))),
        bias=float(np.mean(err)),
        coverage_95=coverage,
        mean_log_score=log_score,
    )
