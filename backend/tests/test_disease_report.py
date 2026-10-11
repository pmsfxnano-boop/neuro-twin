import hashlib
import json

from neuro_twin.disease import report
from neuro_twin.disease.cohorts import CohortRecord, summarize_cohort
from neuro_twin.disease.public import PUBLIC_COHORTS


def test_report_distinguishes_metadata_qc_from_disease_model_evaluation(tmp_path, monkeypatch):
    def fake_fetch(spec):
        digest = hashlib.sha256(f"fixture:{spec.key}".encode()).hexdigest()
        diagnoses = list(spec.diagnosis_map.values())
        records = [
            CohortRecord(
                subject_id=f"fixture-{index:03d}",
                cohort=spec.key,
                diagnosis=diagnoses[index % len(diagnoses)],
                age_years=65.0 + (index % 10),
                sex="F" if index % 2 else "M",
                severity={"mmse": 24.0} if "mmse" in spec.severity_fields else {},
                source_record_id=f"participants.tsv:row={index + 2}",
            )
            for index in range(spec.expected_n)
        ]
        summary = summarize_cohort(
            records,
            source_version=spec.dataset_version,
            raw_sha256=digest,
        )
        provenance = {
            "dataset_id": spec.dataset_id,
            "dataset_version": spec.dataset_version,
            "source_ref": spec.dataset_version,
            "retrieval_uri": spec.participants_url,
            "raw_sha256": digest,
            "participant_count": len(records),
        }
        return records, summary, provenance

    monkeypatch.setattr(report, "fetch_public_cohort", fake_fetch)
    destination = tmp_path / "cohort-evidence.json"
    evidence = report.run(destination)

    assert destination.exists()
    assert json.loads(destination.read_text(encoding="utf-8")) == evidence
    assert evidence["status"] == "PASS"
    assert evidence["status_scope"] == "metadata_cohort_quality_control_only"
    assert evidence["clinical_validation"] is False
    assert evidence["model_evaluation"]["performed"] is False
    assert evidence["model_evaluation"]["status"] == "NOT_RUN"
    assert evidence["model_evaluation"]["endpoint_metrics"] is None

    ad = evidence["cohorts"]["alzheimer_ds004504_v1.0.6"]
    pd = evidence["cohorts"]["parkinson_ds005892_v1.0.0"]
    assert ad["readiness"]["status"] == "PARTIAL_OBSERVABILITY"
    assert "pTau217" in ad["readiness"]["missing_required_features"]
    assert pd["readiness"]["status"] == "NOT_READY"
    assert pd["readiness"]["unobserved_states"] == ["I", "N", "P", "Q"]

    unsigned = dict(evidence)
    claimed_hash = unsigned["evidence_hash"]
    unsigned["evidence_hash"] = None
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert claimed_hash == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
