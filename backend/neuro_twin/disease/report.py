"""Fetch and persist public disease-cohort evidence.

This report is intentionally metadata-level in the first disease integration
batch. It verifies that real public cohorts are reachable and their declared
participant counts and schema remain stable before raw neuroimaging features
are allowed into disease-specific inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .public import PUBLIC_COHORTS, fetch_public_cohort


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


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
            "record_ids": [r.source_record_id for r in records],
        }

    envelope = {
        "schema": "neuro-twin.disease-cohort-evidence.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "clinical_validation": False,
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
