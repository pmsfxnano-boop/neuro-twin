"""Fetch and persist public disease-cohort evidence.

This first disease integration stage is metadata-level quality control. It
verifies real public cohort metadata, hashes the exact input bytes, and reports
which disease-module observation requirements remain unmet. It does NOT run a
disease classifier or claim clinical validation; raw EEG/MRI signals must first
be processed into versioned, patient-level features.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .public import PUBLIC_COHORTS, fetch_public_cohort
from .readiness import assess_dataset_readiness


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _readiness_payload(spec) -> dict[str, Any]:
    disease_key = "alzheimer" if spec.dataset_id == "openneuro-ds004504" else "parkinson"
    readiness = assess_dataset_readiness(
        dataset_id=spec.dataset_id,
        dataset_version=spec.dataset_version,
        disease=disease_key,
        cohort_label_available=True,
    )
    coverage = readiness.coverage
    return {
        "status": readiness.status,
        "cohort_label_available": readiness.cohort_label_available,
        "available_features": list(readiness.available_features),
        "module_version": coverage.module_version,
        "covered_required_features": list(coverage.covered_required_features),
        "missing_required_features": list(coverage.missing_required_features),
        "observed_states": list(coverage.observed_states),
        "unobserved_states": list(coverage.unobserved_states),
        "reasons": list(coverage.reasons),
        "note": readiness.note,
    }


def run(output: Path) -> dict[str, Any]:
    cohorts: dict[str, Any] = {}
    for key in sorted(PUBLIC_COHORTS):
        spec = PUBLIC_COHORTS[key]
        records, summary, provenance = fetch_public_cohort(spec)
        cohorts[key] = {
            "provenance": provenance,
            "summary": {
                "cohort": summary.cohort,
                "source_version": summary.source_version,
                "raw_sha256": summary.raw_sha256,
                "n_total": summary.n_total,
                "diagnosis_counts": summary.diagnosis_counts,
                "age_mean": summary.age_mean,
                "age_sd": summary.age_sd,
                "severity_summary": summary.severity_summary,
                "assumptions": list(summary.assumptions),
            },
            "readiness": _readiness_payload(spec),
            "record_ids": [r.source_record_id for r in records],
        }

    envelope = {
        "schema": "neuro-twin.disease-cohort-evidence.v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "status_scope": "metadata_cohort_quality_control_only",
        "clinical_validation": False,
        "model_evaluation": {
            "status": "NOT_RUN",
            "performed": False,
            "endpoint_metrics": None,
            "reason": (
                "This report reads participants.tsv metadata only. It does not "
                "extract raw EEG/MRI features or fit/evaluate a disease model."
            ),
            "required_next_step": (
                "Extract quality-controlled, versioned subject-level signal/imaging "
                "features, preserve subject-level splits, then evaluate an explicit "
                "endpoint with internal and independent external validation."
            ),
        },
        "cohorts": cohorts,
    }
    unsigned = dict(envelope)
    unsigned["evidence_hash"] = None
    envelope["evidence_hash"] = hashlib.sha256(
        _canonical(unsigned).encode("utf-8")
    ).hexdigest()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(envelope, sort_keys=True, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return envelope


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runtime/disease_cohort_evidence.json"),
    )
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
