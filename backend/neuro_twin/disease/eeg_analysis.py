"""Leakage-resistant EEG endpoint analysis using the public NEMAR copy of ds004504.

This is an exploratory signal-level disease evaluation, not clinical validation and
not a P-I-N-Q disease-state reconstruction. Diagnosis labels are targets only.
Only a fixed 60-second segment per participant is processed; preprocessed/denoised
derivatives are preferred when the pinned NEMAR index exposes them.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

from .coverage import evaluate_module_coverage
from .evaluation import SubjectDiseaseRecord, _auc, evaluate_binary_disease_endpoint
from .modules import Disease, get_disease_module

DATASET_KEY = "nemar_on004504_v1.0.0"
DATASET_DOI = "10.82901/nemar.on004504"
DATASET_LICENSE = "CC0"
PARTICIPANTS_URL = "https://data.nemar.org/on004504/v1.0.0/participants.tsv"
ZARR_INDEX_URL = "https://zarr.nemar.org/on004504/zarr/index.json"
SAMPLE_RATE_HZ = 500.0
SEGMENT_SECONDS = 60
MIN_SEGMENT_SECONDS = 20
PERMUTATIONS = 250
BOOTSTRAPS = 1000
FEATURE_VERSION = "eeg-global-bandpower-v1.0.0"
FEATURE_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
GROUP_MAP = {"A": "AD", "F": "FTD", "C": "CONTROL", "AD": "AD",
             "FTD": "FTD", "CN": "CONTROL", "CONTROL": "CONTROL"}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def extract_eeg_band_features(signal_data: np.ndarray, sampling_rate_hz: float = SAMPLE_RATE_HZ) -> dict[str, float]:
    """Extract finite, global robust spectral features from channels × samples."""
    x = np.asarray(signal_data, dtype=float)
    if x.ndim != 2:
        raise ValueError("EEG input must be a 2D channels-by-samples array")
    if x.shape[0] > x.shape[1] and x.shape[0] > 1024:
        x = x.T
    if x.shape[0] < 2 or x.shape[1] < int(MIN_SEGMENT_SECONDS * sampling_rate_hz):
        raise ValueError("EEG segment is too short or has too few channels")
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 90:
        raise ValueError("sampling rate must be greater than 90 Hz for the declared 45 Hz band")

    finite_fraction = np.mean(np.isfinite(x), axis=1)
    channel_median = np.nanmedian(x, axis=1)
    robust_scale = 1.4826 * np.nanmedian(np.abs(x - channel_median[:, None]), axis=1)
    valid = (finite_fraction >= 0.99) & np.isfinite(robust_scale) & (robust_scale > 1e-12)
    x = x[valid]
    if x.shape[0] < 2:
        raise ValueError("fewer than two EEG channels passed basic finite/variance QC")
    x = x - np.median(x, axis=1, keepdims=True)

    sos = butter(4, [0.5, 45.0], btype="bandpass", fs=sampling_rate_hz, output="sos")
    filtered = sosfiltfilt(sos, x, axis=1)
    nperseg = min(2048, filtered.shape[1])
    frequencies, psd = welch(
        filtered, fs=sampling_rate_hz, nperseg=nperseg,
        noverlap=nperseg // 2, axis=1, average="median", scaling="density"
    )
    absolute: dict[str, float] = {}
    for band, (low, high) in FEATURE_BANDS.items():
        mask = (frequencies >= low) & (frequencies < high)
        if int(mask.sum()) < 2:
            raise ValueError(f"insufficient frequency bins for band {band}")
        power_per_channel = np.trapezoid(psd[:, mask], frequencies[mask], axis=1)
        value = float(np.median(power_per_channel[np.isfinite(power_per_channel)]))
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"invalid power estimate in band {band}")
        absolute[band] = value

    total = float(sum(absolute.values()))
    epsilon = max(np.finfo(float).tiny, total * 1e-12)
    result: dict[str, float] = {}
    for band, power in absolute.items():
        result[f"log10_abs_{band}"] = float(np.log10(power + epsilon))
        result[f"relative_{band}"] = float(power / total)
    result["log10_theta_alpha_ratio"] = float(np.log10((absolute["theta"] + epsilon) / (absolute["alpha"] + epsilon)))
    result["log10_slow_fast_ratio"] = float(np.log10(
        (absolute["delta"] + absolute["theta"] + epsilon) /
        (absolute["alpha"] + absolute["beta"] + epsilon)
    ))
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError("EEG feature extraction emitted a non-finite feature")
    return result


def _fetch(url: str, timeout: float = 90.0) -> bytes:
    last_error: Exception | None = None
    for _ in range(3):
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True,
                              headers={"User-Agent": "NEURO-TWIN-public-research/1.0"}) as client:
                response = client.get(url)
                response.raise_for_status()
                return response.content
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"failed to fetch public source {url}: {last_error}")


def _participants(payload: bytes) -> dict[str, dict[str, str]]:
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"), newline=""), delimiter="\t"))
    if not rows:
        raise ValueError("NEMAR participants.tsv is empty")
    if not {"participant_id", "Group"}.issubset(rows[0]):
        raise ValueError(f"participants.tsv columns changed: {sorted(rows[0])}")
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        subject = (row.get("participant_id") or "").strip()
        group_code = (row.get("Group") or "").strip().upper()
        if not subject:
            raise ValueError("public cohort has an empty participant_id")
        if group_code not in GROUP_MAP:
            raise ValueError(f"unknown pinned cohort group code {group_code!r} for {subject}")
        if subject in result:
            raise ValueError(f"duplicate participant record: {subject}")
        result[subject] = {"label": GROUP_MAP[group_code]}
    if len(result) != 88:
        raise ValueError(f"pinned NEMAR cohort expected 88 subjects, found {len(result)}")
    counts = {g: sum(1 for row in result.values() if row["label"] == g) for g in ("AD", "FTD", "CONTROL")}
    if counts != {"AD": 36, "FTD": 23, "CONTROL": 29}:
        raise ValueError(f"pinned diagnostic group counts changed: {counts}")
    return result


def _store_uri(store_meta: dict[str, Any]) -> str:
    direct = store_meta.get("s3_uri") or store_meta.get("uri")
    value = direct or store_meta.get("zarr")
    if not isinstance(value, str) or not value:
        raise ValueError("NEMAR Zarr index store entry has no Zarr location")
    if value.startswith("s3://"):
        return value
    if value.startswith("nemar/"):
        return "s3://" + value.lstrip("/")
    if value.startswith("on004504/zarr/"):
        return "s3://nemar/" + value.lstrip("/")
    return "s3://nemar/on004504/zarr/" + value.lstrip("/")


def _recording_candidates(index: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    stores = index.get("stores")
    if not isinstance(stores, list) or not stores:
        raise ValueError("NEMAR Zarr index does not contain stores[]")
    found: dict[str, list[dict[str, Any]]] = {}
    for store_meta in stores:
        if not isinstance(store_meta, dict):
            continue
        groups = store_meta.get("groups", [])
        if not isinstance(groups, list):
            continue
        try:
            uri = _store_uri(store_meta)
        except ValueError:
            continue
        for group_meta in groups:
            if not isinstance(group_meta, dict) or not isinstance(group_meta.get("name"), str):
                continue
            group_name = group_meta["name"].strip("/")
            path_text = f"{group_name} {uri}".lower()
            match = re.search(r"(sub-\d{3})", group_name, flags=re.IGNORECASE)
            if not match:
                continue
            if "eeg" not in path_text and "eyesclosed" not in path_text and "eyes-closed" not in path_text:
                continue
            subject = match.group(1).lower()
            is_processed = any(token in path_text for token in ("derivatives", "denois", "preprocess", "cleaned"))
            is_eyes_closed = ("eyesclosed" in path_text) or ("eyes-closed" in path_text)
            found.setdefault(subject, []).append({
                "subject": subject,
                "uri": uri,
                "group_name": group_name,
                "store": store_meta.get("zarr", uri),
                "is_processed": bool(is_processed),
                "is_eyes_closed": bool(is_eyes_closed),
            })
    for subject, values in found.items():
        # Prefer known denoised derivatives, then the named eyes-closed task, then stable lexical order.
        values.sort(key=lambda c: (not c["is_processed"], not c["is_eyes_closed"], c["uri"], c["group_name"]))
    return found


def _scale_offset(values: Any, n_channels: int, fallback: float) -> np.ndarray:
    if values is None:
        return np.full(n_channels, fallback, dtype=float)
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 1:
        return np.full(n_channels, float(array[0]), dtype=float)
    if array.size != n_channels:
        raise ValueError(f"Zarr scale/offset has {array.size} entries for {n_channels} channels")
    return array


def _load_segment(candidate: dict[str, Any], root_cache: dict[str, Any]) -> tuple[np.ndarray, float, str, int]:
    # Optional analysis dependencies are imported lazily so the core backend remains lightweight.
    import zarr
    from zarr.storage import FsspecStore

    uri = candidate["uri"]
    if uri not in root_cache:
        remote_store = FsspecStore.from_url(
            uri,
            storage_options={"anon": True, "client_kwargs": {"region_name": "us-east-2"}},
            read_only=True,
        )
        root_cache[uri] = zarr.open_group(store=remote_store, mode="r", zarr_format=3)
    group = root_cache[uri][candidate["group_name"]]
    if "0" not in group:
        raise ValueError(f"Zarr group {candidate['group_name']} lacks level-0 array '0'")
    array = group["0"]
    shape = tuple(int(n) for n in array.shape)
    if len(shape) != 2:
        raise ValueError(f"expected 2D EEG signal array, got shape {shape}")
    if shape[0] <= 64 and shape[1] > 64:
        channels_first = True
        n_channels, n_samples = shape
    elif shape[1] <= 64 and shape[0] > 64:
        channels_first = False
        n_channels, n_samples = shape[1], shape[0]
    else:
        raise ValueError(f"could not identify channels/time axes in shape {shape}")
    sample_count = min(n_samples, int(SEGMENT_SECONDS * SAMPLE_RATE_HZ))
    if sample_count < int(MIN_SEGMENT_SECONDS * SAMPLE_RATE_HZ):
        raise ValueError(f"recording shorter than {MIN_SEGMENT_SECONDS} seconds")
    digital = np.asarray(array[:, :sample_count] if channels_first else array[:sample_count, :].T)
    array_attrs = dict(array.attrs)
    scale = _scale_offset(array_attrs.get("scale"), n_channels, 1.0)
    offset = _scale_offset(array_attrs.get("offset"), n_channels, 0.0)
    if not np.issubdtype(digital.dtype, np.floating) and "scale" not in array_attrs:
        raise ValueError("integer digital EEG lacks required physical-unit scale metadata")
    physical = digital.astype(float, copy=False) * scale[:, None] + offset[:, None]
    if not np.isfinite(physical).all():
        # Keep QC strict rather than silently imputing EEG data.
        raise ValueError("non-finite values in sampled EEG segment")
    signal_hash = sha256_bytes(
        (candidate["group_name"] + "|" + uri + "|").encode("utf-8") +
        np.ascontiguousarray(digital).tobytes() +
        np.ascontiguousarray(scale).tobytes() + np.ascontiguousarray(offset).tobytes()
    )
    return physical, sample_count / SAMPLE_RATE_HZ, signal_hash, n_channels


def _bootstrap_auc_ci(labels: np.ndarray, scores: np.ndarray, iterations: int = BOOTSTRAPS, seed: int = 73) -> list[float] | None:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    n = len(labels)
    for _ in range(iterations):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(labels[idx])) < 2:
            continue
        value = _auc(labels[idx], scores[idx])
        if np.isfinite(value):
            values.append(float(value))
    if len(values) < max(100, iterations // 2):
        return None
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def _holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    previous = 0.0
    for rank, (name, p_value) in enumerate(ordered):
        value = min(1.0, (m - rank) * float(p_value))
        previous = max(previous, value)
        adjusted[name] = previous
    return adjusted


def _endpoint(name: str, positive: str, negative: str,
              feature_rows: list[dict[str, Any]], *, seed: int) -> tuple[dict[str, Any], float]:
    subset = [row for row in feature_rows if row["label"] in {positive, negative}]
    records = [
        SubjectDiseaseRecord(
            subject_id=row["subject"],
            label=1 if row["label"] == positive else 0,
            features=tuple(float(row["features"][key]) for key in row["feature_names"]),
        )
        for row in sorted(subset, key=lambda r: r["subject"])
    ]
    labels = np.asarray([r.label for r in records], dtype=int)
    if len(records) < 30 or len(np.unique(labels)) != 2:
        raise RuntimeError(f"insufficient subjects for endpoint {name}: n={len(records)}")
    result = evaluate_binary_disease_endpoint(
        records,
        endpoint=name,
        n_splits=5,
        l2=1.0,
        permutation_iterations=PERMUTATIONS,
        random_seed=seed,
    )
    scores = np.asarray(result.oof_probabilities, dtype=float)
    ci = _bootstrap_auc_ci(labels, scores, seed=seed + 100)
    prevalence = float(np.mean(labels))
    baseline_brier = float(np.mean((labels - prevalence) ** 2))
    metrics = {
        "endpoint": name,
        "positive_label": positive,
        "negative_label": negative,
        "n_subjects": result.n_subjects,
        "positive_subjects": result.positive_subjects,
        "negative_subjects": result.negative_subjects,
        "n_features": result.n_features,
        "evaluation": "5-fold stratified subject-level out-of-fold",
        "pooled_auroc": result.pooled_auroc,
        "pooled_auroc_bootstrap_ci95": ci,
        "pooled_auprc": result.pooled_auprc,
        "pooled_balanced_accuracy": result.pooled_balanced_accuracy,
        "pooled_brier": result.pooled_brier,
        "prevalence_only_baseline_brier": baseline_brier,
        "brier_improvement_vs_prevalence_baseline": baseline_brier - result.pooled_brier,
        "permutation_null_auroc_mean": result.null_auroc_mean,
        "permutation_null_auroc_p95": result.null_auroc_p95,
        "permutation_pvalue_unadjusted": result.null_auroc_pvalue,
        "permutation_iterations": result.permutation_iterations,
        "random_seed": seed,
        "clinical_validation": False,
    }
    return metrics, float(result.null_auroc_pvalue if result.null_auroc_pvalue is not None else 1.0)


def run(output: Path) -> dict[str, Any]:
    participants_bytes = _fetch(PARTICIPANTS_URL)
    index_bytes = _fetch(ZARR_INDEX_URL)
    participants = _participants(participants_bytes)
    index = json.loads(index_bytes.decode("utf-8"))
    candidates = _recording_candidates(index)
    label_by_subject = {subject.lower(): row["label"] for subject, row in participants.items()}

    # Dataset IDs are fixed by the pinned source; an unknown subject/group is never guessed.
    root_cache: dict[str, Any] = {}
    feature_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for subject in sorted(participants):
        subject_key = subject.lower()
        available = candidates.get(subject_key, [])
        if not available:
            failures.append({"subject": subject, "reason": "no_matching_eeg_group_in_index"})
            continue
        candidate = available[0]
        try:
            signal_data, seconds, signal_hash, channel_count = _load_segment(candidate, root_cache)
            features = extract_eeg_band_features(signal_data, SAMPLE_RATE_HZ)
            feature_rows.append({
                "subject": subject,
                "label": label_by_subject[subject_key],
                "features": features,
                "feature_names": tuple(sorted(features)),
                "feature_hash": signal_hash,
                "source_group": candidate["group_name"],
                "used_processed_derivative": candidate["is_processed"],
                "segment_seconds": seconds,
                "channel_count": channel_count,
            })
        except Exception as exc:
            failures.append({"subject": subject, "reason": f"{type(exc).__name__}: {str(exc)[:240]}"})
    if len(feature_rows) < 60:
        store_entries = index.get("stores", []) if isinstance(index, dict) else []
        group_count = sum(len(s.get("groups", [])) for s in store_entries if isinstance(s, dict) and isinstance(s.get("groups", []), list))
        diagnostic = {
            "index_keys": sorted(index.keys()) if isinstance(index, dict) else str(type(index)),
            "index_format_version": index.get("format_version") if isinstance(index, dict) else None,
            "store_count": len(store_entries),
            "group_count": group_count,
            "first_store_keys": sorted(store_entries[0].keys()) if store_entries and isinstance(store_entries[0], dict) else [],
            "first_store_zarr": str(store_entries[0].get("zarr"))[:160] if store_entries and isinstance(store_entries[0], dict) else None,
            "first_store_group_examples": [g.get("name") for g in store_entries[0].get("groups", [])[:5] if isinstance(g, dict)] if store_entries and isinstance(store_entries[0], dict) else [],
            "candidate_subject_count": len(candidates),
            "candidate_subject_examples": {k: [{"uri": c["uri"], "group_name": c["group_name"], "processed": c["is_processed"]} for c in v[:2]] for k, v in list(sorted(candidates.items()))[:5]},
            "successful_subjects": len(feature_rows),
            "failed_subjects": len(failures),
            "failure_examples": failures[:12],
        }
        print("NEURO_TWIN_EEG_DISCOVERY_DIAGNOSTICS " + json.dumps(diagnostic, sort_keys=True), flush=True)
        raise RuntimeError(f"only {len(feature_rows)}/88 EEG participants produced features; refusing an underpowered report; diagnostics={json.dumps(diagnostic, sort_keys=True)[:7000]}")

    # Three pairwise endpoints are prespecified; FTD is never collapsed into the AD-negative class.
    endpoint_specs = (
        ("alzheimer_vs_control_eeg", "AD", "CONTROL", 17),
        ("ftd_vs_control_eeg", "FTD", "CONTROL", 29),
        ("alzheimer_vs_ftd_eeg", "AD", "FTD", 43),
    )
    endpoints: dict[str, Any] = {}
    p_values: dict[str, float] = {}
    for name, positive, negative, seed in endpoint_specs:
        metrics, p_value = _endpoint(name, positive, negative, feature_rows, seed=seed)
        endpoints[name] = metrics
        p_values[name] = p_value
    adjusted = _holm_adjust(p_values)
    for name in endpoints:
        endpoints[name]["permutation_pvalue_holm_adjusted"] = adjusted[name]
        endpoints[name]["within_cohort_signal_supported_after_holm"] = bool(adjusted[name] <= 0.05)

    module = get_disease_module(Disease.ALZHEIMER)
    coverage = evaluate_module_coverage(module, {"EEG", "MMSE", "cognition", "diagnosis_group"})
    processed_count = sum(1 for row in feature_rows if row["used_processed_derivative"])
    output_data: dict[str, Any] = {
        "schema": "neuro-twin.disease-eeg-evidence.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if len(feature_rows) >= 60 and len(endpoints) == 3 else "INCONCLUSIVE",
        "scientific_status": "EXPLORATORY_SINGLE_COHORT_RESULT",
        "clinical_validation": False,
        "pinq_disease_reconstruction": {
            "status": coverage.status,
            "disease_module": module.disease.value,
            "module_version": module.version,
            "observed_states": list(coverage.observed_states),
            "unobserved_states": list(coverage.unobserved_states),
            "missing_required_features": list(coverage.missing_required_features),
            "interpretation": "EEG spectral classification is an auxiliary endpoint. It does not estimate or clinically validate P/I/N/Q; four core pathology/response/degeneration/cognition bindings are not all available in this public dataset.",
        },
        "source": {
            "dataset_key": DATASET_KEY,
            "dataset_doi": DATASET_DOI,
            "dataset_version": "NEMAR v1.0.0 (derived from OpenNeuro ds004504 v1.0.9)",
            "license": DATASET_LICENSE,
            "participants_uri": PARTICIPANTS_URL,
            "zarr_index_uri": ZARR_INDEX_URL,
            "participants_tsv_sha256": sha256_bytes(participants_bytes),
            "zarr_index_sha256": sha256_bytes(index_bytes),
            "dataset_diagnosis_counts": {
                label: sum(1 for row in participants.values() if row["label"] == label)
                for label in ("AD", "FTD", "CONTROL")
            },
            "label_handling": "source diagnosis labels are targets only; no diagnosis, MMSE, age or sex field enters the EEG feature vector",
        },
        "feature_pipeline": {
            "name": FEATURE_VERSION,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "segment_policy": f"first {SEGMENT_SECONDS} seconds per participant, minimum {MIN_SEGMENT_SECONDS} seconds",
            "preferred_source": "preprocessed/denoised derivative when exposed in the Zarr index; otherwise the listed eyes-closed EEG group",
            "preprocessing": ["dequantize with per-channel Zarr scale/offset", "channel finite/variance QC", "median centering", "4th order zero-phase Butterworth 0.5–45 Hz", "Welch PSD with median averaging"],
            "bands_hz": {name: list(bounds) for name, bounds in FEATURE_BANDS.items()},
            "feature_names": sorted(feature_rows[0]["features"]),
            "features_per_subject": len(feature_rows[0]["features"]),
            "processed_derivative_subjects": processed_count,
            "raw_or_unmarked_source_subjects": len(feature_rows) - processed_count,
            "successful_subjects": len(feature_rows),
            "failed_subjects": len(failures),
            "failure_examples": failures[:10],
            "subjects_have_single_record_per_endpoint": True,
            "split_unit": "subject",
            "folds": 5,
            "permutation_iterations": PERMUTATIONS,
            "bootstrap_iterations": BOOTSTRAPS,
            "multiple_testing": "Holm adjustment across the three prespecified pairwise endpoints",
        },
        "endpoints": endpoints,
        "limitations": [
            "A single public cohort is insufficient for external validation or a clinical claim.",
            "Fixed 60-second prefixes may not represent full-recording variability.",
            "EEG artifacts and acquisition/site effects can remain despite preprocessing; pipeline-level signal QC does not replace expert review.",
            "No participant overlap or independent external replication study has been used to establish generalization.",
            "This endpoint is not a diagnostic device and must not guide care.",
        ],
    }
    unsigned = dict(output_data)
    unsigned["evidence_hash"] = None
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    output_data["evidence_hash"] = sha256_bytes(canonical.encode("utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_data, sort_keys=True, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "status": output_data["status"],
        "science_status": output_data["scientific_status"],
        "successful_subjects": len(feature_rows),
        "failed_subjects": len(failures),
        "endpoints": {name: {
            "n": value["n_subjects"],
            "auroc": value["pooled_auroc"],
            "ci95": value["pooled_auroc_bootstrap_ci95"],
            "p_holm": value["permutation_pvalue_holm_adjusted"],
        } for name, value in endpoints.items()},
        "pinq_status": coverage.status,
        "evidence_hash": output_data["evidence_hash"],
        "output": str(output),
    }, sort_keys=True))
    return output_data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runtime/disease_eeg_evidence.json"))
    args = parser.parse_args()
    report = run(args.output)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
