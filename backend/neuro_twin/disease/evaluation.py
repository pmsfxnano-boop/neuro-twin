"""Leakage-resistant disease endpoint evaluation on NEURO-TWIN latent summaries.

The evaluator consumes subject-level features that were generated upstream by
the scientific runtime. It performs patient-level cross-validation, fits a
regularized logistic model inside each training fold, computes AUROC/AUPRC,
balanced accuracy and Brier score, and supports a deterministic permutation
null. No image-level split or subject duplication is permitted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class SubjectDiseaseRecord:
    subject_id: str
    label: int
    features: tuple[float, ...]


@dataclass(frozen=True)
class FoldMetrics:
    fold: int
    n_train: int
    n_test: int
    positives_test: int
    negatives_test: int
    auroc: float
    auprc: float
    balanced_accuracy: float
    brier: float


@dataclass(frozen=True)
class DiseaseEvaluationResult:
    endpoint: str
    n_subjects: int
    n_features: int
    positive_subjects: int
    negative_subjects: int
    folds: tuple[FoldMetrics, ...]
    pooled_auroc: float
    pooled_auprc: float
    pooled_balanced_accuracy: float
    pooled_brier: float
    null_auroc_mean: float | None
    null_auroc_p95: float | None
    null_auroc_pvalue: float | None
    permutation_iterations: int
    status: str
    limitations: tuple[str, ...]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def _fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(X)), X])
    p = X1.shape[1]
    scale = np.ones(p)
    scale[0] = 0.0

    def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
        p_hat = _sigmoid(X1 @ w)
        eps = 1e-12
        loss = -np.sum(y * np.log(p_hat + eps) + (1.0 - y) * np.log(1.0 - p_hat + eps))
        loss += 0.5 * l2 * np.sum((scale * w) ** 2)
        grad = X1.T @ (p_hat - y) + l2 * scale * w
        return float(loss), grad

    result = minimize(
        lambda w: objective(w)[0],
        np.zeros(p, dtype=float),
        jac=lambda w: objective(w)[1],
        method="L-BFGS-B",
    )
    if not result.success:
        raise RuntimeError(f"logistic fit failed: {result.message}")
    return np.asarray(result.x, dtype=float)


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(score, kind="mergesort")
    ys = y[order]
    ranks = np.empty(len(y), dtype=float)
    i = 0
    while i < len(y):
        j = i + 1
        while j < len(y) and score[order[j]] == score[order[i]]:
            j += 1
        ranks[i:j] = np.mean(np.arange(i + 1, j + 1, dtype=float))
        i = j
    pos = float(np.sum(ys))
    neg = float(len(y) - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    pos_rank_sum = float(np.sum(ranks[ys == 1]))
    return (pos_rank_sum - pos * (pos + 1.0) / 2.0) / (pos * neg)


def _auprc(y: np.ndarray, score: np.ndarray) -> float:
    pos = int(np.sum(y))
    if pos == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    y_ord = y[order]
    tp = np.cumsum(y_ord)
    fp = np.cumsum(1 - y_ord)
    recall = tp / max(pos, 1)
    precision = tp / np.maximum(tp + fp, 1)
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * np.concatenate([[1.0], precision])[1:]))


def _balanced_accuracy(y: np.ndarray, score: np.ndarray) -> float:
    pred = score >= 0.5
    pos = y == 1
    neg = ~pos
    if not np.any(pos) or not np.any(neg):
        return float("nan")
    tpr = float(np.mean(pred[pos]))
    tnr = float(np.mean(~pred[neg]))
    return 0.5 * (tpr + tnr)


def _stratified_subject_folds(y: np.ndarray, n_splits: int) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    n_splits = min(n_splits, len(pos), len(neg))
    if n_splits < 2:
        raise ValueError("need at least two subjects in each class for stratified evaluation")
    buckets = [[] for _ in range(n_splits)]
    for i, idx in enumerate(pos):
        buckets[i % n_splits].append(int(idx))
    for i, idx in enumerate(neg):
        buckets[i % n_splits].append(int(idx))
    all_idx = set(range(len(y)))
    return tuple(
        (np.asarray(sorted(all_idx - set(b)), dtype=int), np.asarray(sorted(b), dtype=int))
        for b in buckets
    )


def evaluate_binary_disease_endpoint(
    records: Sequence[SubjectDiseaseRecord],
    *,
    endpoint: str,
    n_splits: int = 5,
    l2: float = 1.0,
    permutation_iterations: int = 0,
    random_seed: int = 17,
) -> DiseaseEvaluationResult:
    if not records:
        raise ValueError("records cannot be empty")
    subject_ids = [r.subject_id for r in records]
    if len(set(subject_ids)) != len(subject_ids):
        raise ValueError("one and only one row per subject is required")
    dims = {len(r.features) for r in records}
    if len(dims) != 1 or 0 in dims:
        raise ValueError("all subjects must share one non-empty feature vector length")
    y = np.asarray([int(r.label) for r in records], dtype=int)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be binary 0/1")
    X = np.asarray([r.features for r in records], dtype=float)
    if not np.all(np.isfinite(X)):
        raise ValueError("features must be finite")

    folds = _stratified_subject_folds(y, n_splits)
    oof = np.full(len(y), np.nan, dtype=float)
    fold_metrics: list[FoldMetrics] = []
    for fold_id, (train, test) in enumerate(folds, start=1):
        mu = np.mean(X[train], axis=0)
        sd = np.std(X[train], axis=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        Xtr = (X[train] - mu) / sd
        Xte = (X[test] - mu) / sd
        w = _fit_logistic(Xtr, y[train], l2=l2)
        p = _sigmoid(np.column_stack([np.ones(len(Xte)), Xte]) @ w)
        oof[test] = p
        fold_metrics.append(
            FoldMetrics(
                fold=fold_id,
                n_train=int(len(train)),
                n_test=int(len(test)),
                positives_test=int(np.sum(y[test])),
                negatives_test=int(np.sum(y[test] == 0)),
                auroc=_auc(y[test], p),
                auprc=_auprc(y[test], p),
                balanced_accuracy=_balanced_accuracy(y[test], p),
                brier=float(np.mean((p - y[test]) ** 2)),
            )
        )

    rng = np.random.default_rng(random_seed)
    null_scores: list[float] = []
    for _ in range(permutation_iterations):
        yp = rng.permutation(y)
        perm_oof = np.full(len(y), np.nan, dtype=float)
        valid = True
        for train, test in folds:
            if len(np.unique(yp[train])) < 2:
                valid = False
                break
            mu = np.mean(X[train], axis=0)
            sd = np.std(X[train], axis=0)
            sd = np.where(sd > 1e-12, sd, 1.0)
            w = _fit_logistic((X[train] - mu) / sd, yp[train], l2=l2)
            p = _sigmoid(
                np.column_stack([np.ones(len(test)), (X[test] - mu) / sd]) @ w
            )
            perm_oof[test] = p
        if valid and np.all(np.isfinite(perm_oof)):
            null_scores.append(_auc(yp, perm_oof))

    pooled_auc = _auc(y, oof)
    pooled_pr = _auprc(y, oof)
    pooled_bal = _balanced_accuracy(y, oof)
    pooled_brier = float(np.mean((oof - y) ** 2))
    limitations = (
        "Disease performance is an endpoint-level research result, not clinical validation.",
        "The evaluator cannot repair confounding, scanner effects, site effects, or label bias; upstream QC must surface them.",
        "External validation on an independent cohort remains mandatory before any generalization claim.",
    )
    status = "PASS" if np.isfinite(pooled_auc) and np.isfinite(pooled_brier) else "INCONCLUSIVE"
    return DiseaseEvaluationResult(
        endpoint=endpoint,
        n_subjects=len(records),
        n_features=X.shape[1],
        positive_subjects=int(np.sum(y)),
        negative_subjects=int(np.sum(y == 0)),
        folds=tuple(fold_metrics),
        pooled_auroc=float(pooled_auc),
        pooled_auprc=float(pooled_pr),
        pooled_balanced_accuracy=float(pooled_bal),
        pooled_brier=pooled_brier,
        null_auroc_mean=float(np.mean(null_scores)) if null_scores else None,
        null_auroc_p95=float(np.quantile(null_scores, 0.95)) if null_scores else None,
        null_auroc_pvalue=(float(1 + np.sum(np.asarray(null_scores) >= pooled_auc) / (len(null_scores) + 1)) if False else (float((1 + np.sum(np.asarray(null_scores) >= pooled_auc)) / (len(null_scores) + 1)) if null_scores else None)),
        permutation_iterations=len(null_scores),
        status=status,
        limitations=limitations,
    )
