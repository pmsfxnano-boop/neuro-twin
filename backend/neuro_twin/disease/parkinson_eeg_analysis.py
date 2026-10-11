"""Parkinson EEG research pipeline with clinical severity and cross-dataset tests.

The primary cohort is NEMAR on007526 / OpenNeuro ds007526 v1.0.2. It contains
subject-level MDS-UPDRS Part III and resting-state EEG. The second cohort
(NEMAR on008768 / OpenNeuro ds008768 v1.0.0) is held out for dataset-level
external validation; it is never used for model fitting, scaling or selection.

This is exploratory research, not diagnosis, treatment guidance or clinical
validation. EEG classification is not interpreted as a P/I/N/Q reconstruction.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import spearmanr

from .coverage import evaluate_module_coverage
from .evaluation import (
    SubjectDiseaseRecord,
    _auc,
    _auprc,
    _balanced_accuracy,
    _fit_logistic,
    _sigmoid,
    evaluate_binary_disease_endpoint,
)
from .eeg_analysis import (
    _bootstrap_auc_ci,
    _fetch,
    _load_segment,
    _recording_candidates,
    _validate_nemar_index,
    extract_eeg_band_features,
    sha256_bytes,
)
from .modules import Disease, get_disease_module

TARGET_SAMPLE_RATE_HZ = 250.0
SEGMENT_SECONDS = 30.0
MIN_SEGMENT_SECONDS = 20.0
FOLD_COUNT = 5
PERMUTATIONS = 250
SPLIT_SENSITIVITY_REPEATS = 20
RIDGE_ALPHA = 1.0
FEATURE_VERSION = "eeg-global-bandpower-v1.0.0"
FEATURE_NAMES = (
    "log10_abs_alpha",
    "log10_abs_beta",
    "log10_abs_delta",
    "log10_abs_gamma",
    "log10_abs_theta",
    "log10_slow_fast_ratio",
    "log10_theta_alpha_ratio",
    "relative_alpha",
    "relative_beta",
    "relative_delta",
    "relative_gamma",
    "relative_theta",
)


@dataclass(frozen=True)
class CohortProfile:
    key: str
    openneuro_id: str
    openneuro_version: str
    openneuro_participants_url: str
    nemar_id: str
    nemar_version: str
    nemar_doi: str
    nemar_license: str
    nemar_participants_url: str
    zarr_index_url: str
    expected_n: int
    expected_groups: dict[str, int]
    group_field: str = "group"
    age_field: str = "age"
    sex_field: str = "sex"
    severity_field: str | None = None
    moca_field: str | None = None
    disease_duration_field: str | None = None
    ledd_field: str | None = None


PRIMARY_COHORT = CohortProfile(
    key="parkinson_pd_eeg_primary",
    openneuro_id="openneuro-ds007526",
    openneuro_version="1.0.2",
    openneuro_participants_url=(
        "https://raw.githubusercontent.com/OpenNeuroDatasets/"
        "ds007526/1.0.2/participants.tsv"
    ),
    nemar_id="on007526",
    nemar_version="1.0.0",
    nemar_doi="10.82901/nemar.on007526",
    nemar_license="CC0",
    nemar_participants_url="https://data.nemar.org/on007526/v1.0.0/participants.tsv",
    zarr_index_url="https://zarr.nemar.org/on007526/zarr/index.json",
    expected_n=144,
    expected_groups={"PD": 116, "CONTROL": 28},
    severity_field="updrs_part_iii",
    moca_field="moca",
    disease_duration_field="disease_duration",
    ledd_field="ledd",
)

EXTERNAL_COHORT = CohortProfile(
    key="parkinson_pd_eeg_external",
    openneuro_id="openneuro-ds008768",
    openneuro_version="1.0.0",
    openneuro_participants_url=(
        "https://raw.githubusercontent.com/OpenNeuroDatasets/"
        "ds008768/1.0.0/participants.tsv"
    ),
    nemar_id="on008768",
    nemar_version="1.0.0",
    nemar_doi="10.82901/nemar.on008768",
    nemar_license="CC0",
    nemar_participants_url="https://data.nemar.org/on008768/v1.0.0/participants.tsv",
    zarr_index_url="https://zarr.nemar.org/on008768/zarr/index.json",
    expected_n=312,
    expected_groups={"PD": 202, "CONTROL": 110},
)


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _number(value: Any) -> float | None:
    text = str(value or "").strip()
    if text.lower() in {"", "n/a", "na", "nan", "none", "null"}:
        return None
    number = float(text)
    if not np.isfinite(number):
        raise ValueError(f"non-finite numeric value: {text!r}")
    return number


def _normalize_group(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"PD", "PARKINSON", "PARKINSONS"}:
        return "PD"
    if text in {"HC", "CONTROL", "CONTROLS", "HEALTHY CONTROL"}:
        return "CONTROL"
    raise ValueError(f"unknown Parkinson cohort group value {value!r}")


def parse_participants(payload: bytes, profile: CohortProfile, *, validate_counts: bool = True) -> dict[str, dict[str, Any]]:
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"), newline=""), delimiter="\t"))
    if not rows:
        raise ValueError(f"{profile.key}: participants.tsv is empty")
    required = {"participant_id", profile.group_field, profile.age_field, profile.sex_field}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{profile.key}: participants.tsv misses required fields {sorted(missing)}")

    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        subject = (row.get("participant_id") or "").strip().lower()
        if not subject or subject in output:
            raise ValueError(f"{profile.key}: empty or duplicate participant_id {subject!r}")
        group = _normalize_group(row.get(profile.group_field))
        age = _number(row.get(profile.age_field))
        sex = str(row.get(profile.sex_field) or "").strip().upper()
        if age is None or not (18.0 <= age <= 110.0):
            raise ValueError(f"{profile.key}: invalid age for {subject}")
        if sex not in {"F", "M"}:
            raise ValueError(f"{profile.key}: unexpected sex value {sex!r} for {subject}")

        output[subject] = {
            "subject": subject,
            "label": group,
            "age_years": age,
            "sex": sex,
            "updrs_part_iii": _number(row.get(profile.severity_field)) if profile.severity_field else None,
            "moca": _number(row.get(profile.moca_field)) if profile.moca_field else None,
            "disease_duration_months": _number(row.get(profile.disease_duration_field)) if profile.disease_duration_field else None,
            "ledd_mg": _number(row.get(profile.ledd_field)) if profile.ledd_field else None,
        }

    if validate_counts:
        counts = {
            label: sum(item["label"] == label for item in output.values())
            for label in ("PD", "CONTROL")
        }
        if len(output) != profile.expected_n or counts != profile.expected_groups:
            raise RuntimeError(
                f"{profile.key}: source schema/count drift; n={len(output)}, "
                f"groups={counts}, expected n={profile.expected_n}, "
                f"groups={profile.expected_groups}"
            )
    return output


def crosscheck_participants(
    nemar_participants: dict[str, dict[str, Any]],
    source_participants: dict[str, dict[str, Any]],
    *,
    nemar_payload: bytes,
    source_payload: bytes,
    profile: CohortProfile,
) -> dict[str, Any]:
    if set(nemar_participants) != set(source_participants):
        raise ValueError(f"{profile.key}: NEMAR and OpenNeuro participant IDs do not match")
    fields = ["label", "age_years", "sex"]
    for optional in ("updrs_part_iii", "moca", "disease_duration_months", "ledd_mg"):
        if profile.severity_field and optional in {"updrs_part_iii", "moca", "disease_duration_months", "ledd_mg"}:
            fields.append(optional)
    differences: list[dict[str, Any]] = []
    for subject in sorted(nemar_participants):
        for field in fields:
            left = nemar_participants[subject].get(field)
            right = source_participants[subject].get(field)
            if left != right:
                differences.append({"subject": subject, "field": field, "nemar": left, "openneuro": right})
                if len(differences) >= 10:
                    break
        if differences:
            break
    if differences:
        raise ValueError(f"{profile.key}: participant metadata mismatch between mirrors: {differences}")
    return {
        "dataset_id": profile.openneuro_id,
        "dataset_version": profile.openneuro_version,
        "retrieval_uri": profile.openneuro_participants_url,
        "openneuro_participants_sha256": sha256_bytes(source_payload),
        "nemar_participants_sha256": sha256_bytes(nemar_payload),
        "participant_count": len(nemar_participants),
        "subject_ids_match": True,
        "diagnosis_labels_match": True,
        "clinical_fields_crosschecked": bool(profile.severity_field),
    }


def _session_number(path: str) -> int:
    match = re.search(r"(?:^|/)ses-(\d+)(?:/|_)", path, flags=re.IGNORECASE)
    return int(match.group(1)) if match else 0


def _select_rest_candidates(candidates: dict[str, list[dict[str, Any]]], subject: str) -> list[dict[str, Any]]:
    values = [
        item for item in candidates.get(subject.lower(), [])
        if "task-rest" in str(item.get("record_path", "")).lower()
        and not item.get("is_processed", False)
    ]
    return sorted(
        values,
        key=lambda item: (
            _session_number(str(item.get("record_path", ""))),
            abs(float(np.log(float(item["sampling_rate_hz"]) / TARGET_SAMPLE_RATE_HZ))),
            str(item.get("record_path", "")),
            str(item.get("group_name", "")),
        ),
    )


def extract_cohort_features(profile: CohortProfile) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    nemar_bytes = _fetch(profile.nemar_participants_url)
    source_bytes = _fetch(profile.openneuro_participants_url)
    index_bytes = _fetch(profile.zarr_index_url)
    participants = parse_participants(nemar_bytes, profile)
    source_participants = parse_participants(source_bytes, profile)
    crosscheck = crosscheck_participants(
        participants,
        source_participants,
        nemar_payload=nemar_bytes,
        source_payload=source_bytes,
        profile=profile,
    )
    index = json.loads(index_bytes.decode("utf-8"))
    index_provenance = _validate_nemar_index(
        index,
        expected_dataset_id=profile.nemar_id,
        expected_doi=profile.nemar_doi,
        expected_license=profile.nemar_license,
    )
    candidate_map = _recording_candidates(
        index,
        expected_dataset_id=profile.nemar_id,
        expected_doi=profile.nemar_doi,
        expected_license=profile.nemar_license,
    )
    root_cache: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for position, (subject, participant) in enumerate(sorted(participants.items()), start=1):
        candidates = _select_rest_candidates(candidate_map, subject)
        if not candidates:
            failures.append({"subject": subject, "reason": "no_raw_task_rest_recording_in_NEMAR_index"})
            continue
        last_error: Exception | None = None
        for candidate in candidates:
            try:
                signal, seconds, signal_hash, channel_count, sample_rate, unit = _load_segment(
                    candidate,
                    root_cache,
                    index_source_commit=index_provenance["source_commit"],
                    expected_dataset_id=profile.nemar_id,
                    expected_doi=profile.nemar_doi,
                    expected_license=profile.nemar_license,
                    target_sampling_rate_hz=TARGET_SAMPLE_RATE_HZ,
                    segment_seconds=SEGMENT_SECONDS,
                    min_segment_seconds=MIN_SEGMENT_SECONDS,
                )
                if not np.isclose(sample_rate, TARGET_SAMPLE_RATE_HZ, rtol=0.0, atol=1e-8):
                    raise ValueError(f"effective rate is not harmonized: {sample_rate}")
                if unit != "uV":
                    raise ValueError(f"physical unit was not normalized to uV: {unit}")
                features = extract_eeg_band_features(signal, sample_rate)
                if tuple(sorted(features)) != FEATURE_NAMES:
                    raise ValueError("EEG feature schema changed from the frozen v1 feature set")
                feature_hash = hashlib.sha256(_canonical(features).encode("utf-8")).hexdigest()
                rows.append({
                    **participant,
                    "features": features,
                    "feature_names": FEATURE_NAMES,
                    "signal_segment_sha256": signal_hash,
                    "feature_sha256": feature_hash,
                    "record_path": candidate["record_path"],
                    "zarr_uri": candidate["uri"],
                    "zarr_group": candidate["group_name"],
                    "segment_seconds": seconds,
                    "channel_count": channel_count,
                    "sampling_rate_hz": sample_rate,
                    "physical_unit": unit,
                    "source_sampling_rate_hz": candidate["sampling_rate_hz"],
                    "session_order": _session_number(candidate["record_path"]),
                })
                last_error = None
                break
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            failures.append({"subject": subject, "reason": f"{type(last_error).__name__}: {last_error}"})
        if position % 25 == 0:
            print(
                f"NEURO_TWIN_PD_EEG cohort={profile.key} progress={position}/{len(participants)} "
                f"success={len(rows)} failures={len(failures)}",
                flush=True,
            )

    success_counts = {
        group: sum(row["label"] == group for row in rows)
        for group in ("PD", "CONTROL")
    }
    excluded_counts = {
        group: participants[subject]["label"]
        for subject in participants
        if subject not in {row["subject"] for row in rows}
        for group in [participants[subject]["label"]]
    }
    observed_sample_rates = sorted({float(row["source_sampling_rate_hz"]) for row in rows})
    observed_units = sorted({str(row["physical_unit"]) for row in rows})
    summary = {
        "profile": profile.key,
        "participant_count": len(participants),
        "successful_signal_count": len(rows),
        "signal_success_fraction": len(rows) / max(1, len(participants)),
        "expected_group_counts": profile.expected_groups,
        "signal_group_counts": success_counts,
        "failed_signal_count": len(failures),
        "failed_subjects": failures[:30],
        "source_sampling_rates_hz": observed_sample_rates,
        "effective_sampling_rate_hz": TARGET_SAMPLE_RATE_HZ,
        "physical_units": observed_units,
        "segment_seconds_requested": SEGMENT_SECONDS,
        "segment_seconds_minimum": MIN_SEGMENT_SECONDS,
        "selection_policy": (
            "one raw/unmarked task-rest recording per participant; earliest session first, "
            "then native sampling rate closest to 250 Hz; no task-walk recording is used"
        ),
        "source_provenance": {
            "openneuro_id": profile.openneuro_id,
            "openneuro_version": profile.openneuro_version,
            "openneuro_participants_sha256": crosscheck["openneuro_participants_sha256"],
            "nemar_id": profile.nemar_id,
            "nemar_version": profile.nemar_version,
            "nemar_doi": profile.nemar_doi,
            "nemar_license": profile.nemar_license,
            "nemar_participants_sha256": crosscheck["nemar_participants_sha256"],
            "zarr_index_sha256": sha256_bytes(index_bytes),
            "nemar_index": index_provenance,
            "mirror_crosscheck": crosscheck,
        },
    }
    if len(rows) < int(0.90 * profile.expected_n):
        raise RuntimeError(
            f"{profile.key}: only {len(rows)}/{profile.expected_n} signal records passed QC; "
            "minimum required is 90% for a cohort report"
        )
    for group, expected_count in profile.expected_groups.items():
        min_success = int(0.80 * expected_count)
        if success_counts[group] < min_success:
            raise RuntimeError(
                f"{profile.key}: signal coverage for group {group} is {success_counts[group]}/"
                f"{expected_count}, below required {min_success}"
            )
    return rows, summary, {
        "participants": participants,
        "index_provenance": index_provenance,
        "index_sha256": sha256_bytes(index_bytes),
        "crosscheck": crosscheck,
    }


def _to_records(
    rows: list[dict[str, Any]],
    positive: str,
    negative: str,
    feature_getter: Callable[[dict[str, Any]], tuple[float, ...]],
) -> list[SubjectDiseaseRecord]:
    subset = [row for row in rows if row["label"] in {positive, negative}]
    return [
        SubjectDiseaseRecord(
            subject_id=row["subject"],
            label=1 if row["label"] == positive else 0,
            features=feature_getter(row),
        )
        for row in sorted(subset, key=lambda item: item["subject"])
    ]


def _classification_summary(result, records: list[SubjectDiseaseRecord], *, seed: int, bootstrap_seed: int) -> dict[str, Any]:
    labels = np.asarray([record.label for record in records], dtype=int)
    scores = np.asarray(result.oof_probabilities, dtype=float)
    baseline = np.asarray(result.oof_prevalence_probabilities, dtype=float)
    return {
        "n_subjects": result.n_subjects,
        "positive_subjects": result.positive_subjects,
        "negative_subjects": result.negative_subjects,
        "n_features": result.n_features,
        "evaluation": "5-fold stratified subject-level OOF; all feature scaling fit within each training fold",
        "pooled_auroc": result.pooled_auroc,
        "conditional_oof_score_bootstrap_ci95": _bootstrap_auc_ci(labels, scores, seed=bootstrap_seed),
        "bootstrap_interpretation": (
            "Resamples the fixed OOF score/label pairs. It is conditional on fitted folds and does not refit "
            "the model; reported repeated-split sensitivity is a separate uncertainty check."
        ),
        "pooled_auprc": result.pooled_auprc,
        "pooled_balanced_accuracy": result.pooled_balanced_accuracy,
        "pooled_brier": result.pooled_brier,
        "fold_local_prevalence_baseline_brier": float(np.mean((labels - baseline) ** 2)),
        "fold_metrics": [
            {
                "fold": fold.fold,
                "n_train": fold.n_train,
                "n_test": fold.n_test,
                "positives_test": fold.positives_test,
                "negatives_test": fold.negatives_test,
                "auroc": fold.auroc,
                "auprc": fold.auprc,
                "balanced_accuracy": fold.balanced_accuracy,
                "brier": fold.brier,
            }
            for fold in result.folds
        ],
        "permutation_null_auroc_mean": result.null_auroc_mean,
        "permutation_null_auroc_p95": result.null_auroc_p95,
        "permutation_pvalue_unadjusted": result.null_auroc_pvalue,
        "permutation_iterations": result.permutation_iterations,
        "random_seed": seed,
        "per_subject_oof_predictions": [
            {"subject_id": record.subject_id, "label": int(record.label), "probability": float(score)}
            for record, score in zip(records, scores)
        ],
        "clinical_validation": False,
    }


def _feature_vectors(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["feature_names"] = FEATURE_NAMES


def _demographic_sensitivity(rows: list[dict[str, Any]], *, seed: int = 41) -> dict[str, Any]:
    all_records: dict[str, list[SubjectDiseaseRecord]] = {}
    all_records["eeg_only"] = _to_records(
        rows, "PD", "CONTROL",
        lambda row: tuple(float(row["features"][name]) for name in FEATURE_NAMES),
    )
    all_records["age_sex_only"] = _to_records(
        rows, "PD", "CONTROL",
        lambda row: (float(row["age_years"]), 1.0 if row["sex"] == "F" else 0.0),
    )
    all_records["eeg_plus_age_sex"] = _to_records(
        rows, "PD", "CONTROL",
        lambda row: tuple(float(row["features"][name]) for name in FEATURE_NAMES)
        + (float(row["age_years"]), 1.0 if row["sex"] == "F" else 0.0),
    )
    result: dict[str, Any] = {
        "target": "Parkinson disease versus healthy control",
        "models_share_subjects_and_fold_seed": True,
        "random_seed": seed,
        "metrics": {},
        "interpretation": (
            "Same-participant, same-fold descriptive sensitivity analysis. Age/sex features are never part "
            "of the EEG-only model; diagnosis, UPDRS, MoCA, disease duration and LEDD remain excluded from "
            "classification inputs."
        ),
    }
    for name, records in all_records.items():
        fit = evaluate_binary_disease_endpoint(
            records, endpoint=f"pd_confound_{name}", n_splits=FOLD_COUNT,
            l2=1.0, permutation_iterations=0, random_seed=seed,
        )
        result["metrics"][name] = {
            "n_subjects": fit.n_subjects,
            "n_features": fit.n_features,
            "pooled_auroc": fit.pooled_auroc,
            "pooled_auprc": fit.pooled_auprc,
            "pooled_balanced_accuracy": fit.pooled_balanced_accuracy,
            "pooled_brier": fit.pooled_brier,
            "fold_metrics": [
                {"fold": f.fold, "auroc": f.auroc, "balanced_accuracy": f.balanced_accuracy, "brier": f.brier}
                for f in fit.folds
            ],
        }
    return result


def _external_metrics(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    feature_getter: Callable[[dict[str, Any]], tuple[float, ...]],
) -> dict[str, Any]:
    ordered_train = sorted(train_rows, key=lambda row: row["subject"])
    ordered_test = sorted(test_rows, key=lambda row: row["subject"])
    X_train = np.asarray([feature_getter(row) for row in ordered_train], dtype=float)
    y_train = np.asarray([1 if row["label"] == "PD" else 0 for row in ordered_train], dtype=int)
    X_test = np.asarray([feature_getter(row) for row in ordered_test], dtype=float)
    y_test = np.asarray([1 if row["label"] == "PD" else 0 for row in ordered_test], dtype=int)
    mean = np.mean(X_train, axis=0)
    scale = np.std(X_train, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    Xtr = (X_train - mean) / scale
    Xte = (X_test - mean) / scale
    parameters = _fit_logistic(Xtr, y_train, l2=1.0)
    probabilities = _sigmoid(np.column_stack([np.ones(len(Xte)), Xte]) @ parameters)
    prevalence = float(np.mean(y_train))
    return {
        "n_train": int(len(ordered_train)),
        "n_test": int(len(ordered_test)),
        "train_positive_count": int(np.sum(y_train)),
        "train_negative_count": int(np.sum(y_train == 0)),
        "test_positive_count": int(np.sum(y_test)),
        "test_negative_count": int(np.sum(y_test == 0)),
        "auroc": _auc(y_test, probabilities),
        "conditional_score_bootstrap_ci95": _bootstrap_auc_ci(y_test, probabilities, iterations=1000, seed=101),
        "auprc": _auprc(y_test, probabilities),
        "balanced_accuracy": _balanced_accuracy(y_test, probabilities),
        "brier": float(np.mean((y_test - probabilities) ** 2)),
        "training_prevalence_baseline_brier": float(np.mean((y_test - prevalence) ** 2)),
        "brier_improvement_vs_training_prevalence": float(np.mean((y_test - prevalence) ** 2) - np.mean((y_test - probabilities) ** 2)),
        "per_subject_predictions": [
            {"subject_id": row["subject"], "label": int(y), "probability": float(probability)}
            for row, y, probability in zip(ordered_test, y_test, probabilities)
        ],
        "clinical_validation": False,
    }


def _regression_folds(y: np.ndarray, n_splits: int = 5, seed: int = 41) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    if y.ndim != 1 or len(y) < n_splits * 2:
        raise ValueError("not enough severity observations to construct regression folds")
    rng = np.random.default_rng(seed)
    tie_breaker = rng.random(len(y))
    order = np.lexsort((tie_breaker, y))
    buckets: list[list[int]] = [[] for _ in range(n_splits)]
    for position, index in enumerate(order):
        buckets[position % n_splits].append(int(index))
    all_indices = set(range(len(y)))
    return tuple(
        (
            np.asarray(sorted(all_indices - set(bucket)), dtype=int),
            np.asarray(sorted(bucket), dtype=int),
        )
        for bucket in buckets
    )


def _ridge_oof_predictions(X: np.ndarray, y: np.ndarray, *, alpha: float = RIDGE_ALPHA, seed: int = 41) -> tuple[np.ndarray, list[dict[str, Any]]]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or y.shape != (len(X),) or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("ridge regression requires finite X[n,p] and y[n]")
    predictions = np.full(len(y), np.nan, dtype=float)
    fold_meta: list[dict[str, Any]] = []
    for fold_id, (train, test) in enumerate(_regression_folds(y, n_splits=FOLD_COUNT, seed=seed), start=1):
        mu = np.mean(X[train], axis=0)
        sd = np.std(X[train], axis=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        X_train = (X[train] - mu) / sd
        X_test = (X[test] - mu) / sd
        design_train = np.column_stack([np.ones(len(train)), X_train])
        design_test = np.column_stack([np.ones(len(test)), X_test])
        penalty = np.eye(design_train.shape[1], dtype=float)
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(
            design_train.T @ design_train + alpha * penalty + 1e-10 * np.eye(design_train.shape[1]),
            design_train.T @ y[train],
        )
        predictions[test] = design_test @ beta
        fold_meta.append({
            "fold": fold_id,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "train_updrs_mean": float(np.mean(y[train])),
            "test_updrs_mean": float(np.mean(y[test])),
        })
    if not np.all(np.isfinite(predictions)):
        raise RuntimeError("ridge OOF predictions contain unfilled values")
    return predictions, fold_meta


def _regression_metrics(y: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    rmse = float(np.sqrt(np.mean((y - predictions) ** 2)))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - np.sum((y - predictions) ** 2) / ss_tot) if ss_tot > 0 else float("nan")
    return {
        "oos_rmse_updrs_points": rmse,
        "oos_mae_updrs_points": float(np.mean(np.abs(y - predictions))),
        "oos_r2": r2,
        "oos_spearman_rho": float(spearmanr(y, predictions).statistic),
    }


def _severity_evaluation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pd_rows = [
        row for row in rows
        if row["label"] == "PD"
        and row["updrs_part_iii"] is not None
        and row["disease_duration_months"] is not None
        and row["ledd_mg"] is not None
    ]
    if len(pd_rows) < 50:
        return {
            "status": "INCONCLUSIVE",
            "reason": "fewer than 50 PD subjects have complete EEG, UPDRS-III, disease-duration and LEDD data",
            "n_subjects": len(pd_rows),
        }
    ordered = sorted(pd_rows, key=lambda row: row["subject"])
    y = np.asarray([row["updrs_part_iii"] for row in ordered], dtype=float)
    eeg = np.asarray([[row["features"][name] for name in FEATURE_NAMES] for row in ordered], dtype=float)
    clinical = np.asarray([
        [
            row["age_years"],
            1.0 if row["sex"] == "F" else 0.0,
            row["disease_duration_months"],
            row["ledd_mg"],
        ]
        for row in ordered
    ], dtype=float)
    median_predictions = np.full(len(y), np.nan, dtype=float)
    folds = _regression_folds(y, n_splits=FOLD_COUNT, seed=41)
    for _, (train, test) in enumerate(folds):
        median_predictions[test] = float(np.median(y[train]))
    eeg_only, fold_meta = _ridge_oof_predictions(eeg, y, alpha=RIDGE_ALPHA, seed=41)
    clinical_only, clinical_folds = _ridge_oof_predictions(clinical, y, alpha=RIDGE_ALPHA, seed=41)
    combined, combined_folds = _ridge_oof_predictions(np.column_stack([eeg, clinical]), y, alpha=RIDGE_ALPHA, seed=41)
    return {
        "status": "EXPLORATORY_OOS_REGRESSION",
        "endpoint": "MDS-UPDRS Part III motor score within Parkinson group",
        "n_subjects": len(pd_rows),
        "complete_case_policy": "PD only; require UPDRS-III, EEG, age, sex, disease duration and LEDD present",
        "evaluation": "5-fold out-of-fold ridge regression; feature standardization fit within training fold; fixed alpha=1.0",
        "models": {
            "training_fold_median_baseline": _regression_metrics(y, median_predictions),
            "eeg_spectral_features_only": _regression_metrics(y, eeg_only),
            "age_sex_duration_ledd_only": _regression_metrics(y, clinical_only),
            "eeg_plus_age_sex_duration_ledd": _regression_metrics(y, combined),
        },
        "incremental_rmse_gain_from_eeg_over_clinical_covariates": (
            _regression_metrics(y, clinical_only)["oos_rmse_updrs_points"]
            - _regression_metrics(y, combined)["oos_rmse_updrs_points"]
        ),
        "folds": fold_meta,
        "clinical_only_folds": clinical_folds,
        "combined_folds": combined_folds,
        "limitations": [
            "Cross-sectional severity estimation is not longitudinal progression prediction.",
            "This small, single-center cohort does not establish clinical utility or a treatment-response model.",
            "LEDD and disease duration are covariates, not ground-truth pathology markers.",
            "Ridge alpha is pre-fixed; hyperparameter tuning and external severity validation have not been performed.",
        ],
        "clinical_validation": False,
    }


def run(output: Path) -> dict[str, Any]:
    primary_rows, primary_summary, primary_aux = extract_cohort_features(PRIMARY_COHORT)
    external_rows, external_summary, external_aux = extract_cohort_features(EXTERNAL_COHORT)
    _feature_vectors(primary_rows)
    _feature_vectors(external_rows)

    primary_records = _to_records(
        primary_rows, "PD", "CONTROL",
        lambda row: tuple(float(row["features"][name]) for name in FEATURE_NAMES),
    )
    labels = np.asarray([record.label for record in primary_records], dtype=int)
    primary_fit = evaluate_binary_disease_endpoint(
        primary_records,
        endpoint="parkinson_vs_control_primary_cohort",
        n_splits=FOLD_COUNT,
        l2=1.0,
        permutation_iterations=PERMUTATIONS,
        random_seed=41,
    )
    primary_classification = _classification_summary(
        primary_fit, primary_records, seed=41, bootstrap_seed=141
    )
    primary_classification["split_sensitivity"] = _split_sensitivity(
        "parkinson_vs_control",
        "PD",
        "CONTROL",
        primary_rows,
        base_seed=41,
        repeats=SPLIT_SENSITIVITY_REPEATS,
    )
    primary_classification["prespecified_endpoint"] = True
    primary_classification["multiple_testing_correction"] = "none required for the single primary hypothesis"
    confounding = _demographic_sensitivity(primary_rows, seed=41)

    external_tests = {
        "eeg_only": _external_metrics(
            primary_rows,
            external_rows,
            lambda row: tuple(float(row["features"][name]) for name in FEATURE_NAMES),
        ),
        "age_sex_only": _external_metrics(
            primary_rows,
            external_rows,
            lambda row: (float(row["age_years"]), 1.0 if row["sex"] == "F" else 0.0),
        ),
        "eeg_plus_age_sex": _external_metrics(
            primary_rows,
            external_rows,
            lambda row: tuple(float(row["features"][name]) for name in FEATURE_NAMES)
            + (float(row["age_years"]), 1.0 if row["sex"] == "F" else 0.0),
        ),
    }
    module = get_disease_module(Disease.PARKINSON)
    coverage = evaluate_module_coverage(module, {"MDS_UPDRS", "EEG", "diagnosis_group", "age", "sex", "disease_duration", "LEDD", "MoCA"})
    severity = _severity_evaluation(primary_rows)

    result: dict[str, Any] = {
        "schema": "neuro-twin.parkinson-eeg-evidence.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "scientific_status": "EXPLORATORY_SINGLE_PRIMARY_COHORT_PLUS_EXTERNAL_DATASET_TEST",
        "clinical_validation": False,
        "runtime_pinq_reconstruction": {
            "status": "NOT_PERFORMED",
            "reason": "EEG spectral features and diagnosis labels are not registered as validated P/I/N/Q observation operators.",
        },
        "disease_module_coverage": {
            "disease": coverage.disease,
            "module_version": coverage.module_version,
            "status": coverage.status,
            "available_features": list(coverage.available_features),
            "covered_required_features": list(coverage.covered_required_features),
            "missing_required_features": list(coverage.missing_required_features),
            "observed_states": list(coverage.observed_states),
            "unobserved_states": list(coverage.unobserved_states),
            "reasons": list(coverage.reasons),
        },
        "primary_cohort": {
            "dataset": primary_summary,
            "diagnosis_group_counts_source": PRIMARY_COHORT.expected_groups,
            "primary_endpoint": primary_classification,
            "demographic_confounding_sensitivity": confounding,
            "clinical_severity_oos_regression": severity,
        },
        "external_dataset_test": {
            "dataset": external_summary,
            "purpose": (
                "Train and scale only on ds007526, then assess the frozen pipeline on ds008768. "
                "No external labels or features are used for tuning, fitting or normalization."
            ),
            "models": external_tests,
            "identity_note": (
                "This is external-dataset validation. The datasets use different cohort-local identifiers; "
                "the available metadata do not provide a person-level cross-dataset identity crosswalk."
            ),
        },
        "pipeline": {
            "feature_version": FEATURE_VERSION,
            "features": list(FEATURE_NAMES),
            "signal_preprocessing": (
                "NEMAR raw/unmarked EEG; select only channel_type=EEG and usable channels; apply channel scales/offsets; "
                "convert physical units to microvolts; resample polyphase to 250 Hz; subtract channel median; "
                "4th-order 0.5-45 Hz Butterworth SOS; median Welch PSD; global median bandpowers."
            ),
            "target_sampling_rate_hz": TARGET_SAMPLE_RATE_HZ,
            "segment_seconds": SEGMENT_SECONDS,
            "minimum_segment_seconds": MIN_SEGMENT_SECONDS,
            "folds": FOLD_COUNT,
            "permutation_iterations_primary": PERMUTATIONS,
            "split_sensitivity_repeats": SPLIT_SENSITIVITY_REPEATS,
            "ridge_alpha_severity": RIDGE_ALPHA,
            "source_data_not_committed": True,
        },
        "limitations": [
            "The primary cohort has 116 PD participants and 28 healthy controls, so uncertainty and class balance remain material.",
            "The external dataset combines multiple source studies and acquisition histories; this is a domain-shift test, not proof of prospective clinical generalization.",
            "Cross-dataset participant identity cannot be established from cohort-local IDs alone.",
            "The current pipeline uses one baseline/rest EEG segment per participant; repeated sessions are deliberately not pooled to avoid within-subject leakage.",
            "Raw recordings are used; the analysis does not reproduce all dataset authors' artifact-rejection/ICA workflows.",
            "Global power features omit spatial/topographic information and may preserve site, medication, vigilance and acquisition effects.",
            "The PD v1 disease module remains only partially observable because alpha-synuclein SAA and NfL are absent.",
            "No diagnosis or progression claim is made from P/I/N/Q; this pipeline evaluates EEG endpoints only.",
        ],
    }
    unsigned = dict(result)
    unsigned["evidence_hash"] = None
    result["evidence_hash"] = hashlib.sha256(_canonical(unsigned).encode("utf-8")).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runtime/parkinson_eeg_evidence.json"))
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps({
        "status": report["status"],
        "scientific_status": report["scientific_status"],
        "evidence_hash": report["evidence_hash"],
        "primary_cohort": {
            "n": report["primary_cohort"]["primary_endpoint"]["n_subjects"],
            "auroc": report["primary_cohort"]["primary_endpoint"]["pooled_auroc"],
            "p_value": report["primary_cohort"]["primary_endpoint"]["permutation_pvalue_unadjusted"],
        },
        "external_dataset_test": {
            key: {
                "n_test": value["n_test"],
                "auroc": value["auroc"],
                "balanced_accuracy": value["balanced_accuracy"],
            }
            for key, value in report["external_dataset_test"]["models"].items()
        },
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
