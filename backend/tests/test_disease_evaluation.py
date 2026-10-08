import numpy as np

from neuro_twin.disease.evaluation import SubjectDiseaseRecord, evaluate_binary_disease_endpoint


def test_disease_endpoint_is_patient_level_and_oof():
    records = []
    for i in range(12):
        label = int(i >= 6)
        x1 = (2.0 if label else -2.0) + 0.15 * i
        x2 = (0.5 if label else -0.5) + np.sin(i)
        records.append(SubjectDiseaseRecord(subject_id=f"s{i:02d}", label=label, features=(x1, x2)))

    result = evaluate_binary_disease_endpoint(
        records,
        endpoint="parkinson_vs_control",
        n_splits=3,
        permutation_iterations=20,
        random_seed=11,
    )

    assert result.status == "PASS"
    assert result.n_subjects == 12
    assert result.positive_subjects == 6
    assert result.negative_subjects == 6
    assert 0.0 <= result.pooled_auroc <= 1.0
    assert result.pooled_auroc > 0.90
    assert result.pooled_brier < 0.15
    assert result.permutation_iterations == 20
    assert result.null_auroc_p95 is not None
    assert result.null_auroc_pvalue is not None
    assert 0.0 < result.null_auroc_pvalue <= 1.0


def test_duplicate_subjects_are_rejected():
    records = [
        SubjectDiseaseRecord("s1", 0, (0.0,)),
        SubjectDiseaseRecord("s1", 1, (1.0,)),
    ]
    try:
        evaluate_binary_disease_endpoint(records, endpoint="x", n_splits=2)
    except ValueError as exc:
        assert "one and only one row per subject" in str(exc)
    else:
        raise AssertionError("duplicate subject ids should fail")


def test_single_class_is_rejected():
    records = [SubjectDiseaseRecord(f"s{i}", 1, (float(i),)) for i in range(8)]
    try:
        evaluate_binary_disease_endpoint(records, endpoint="x", n_splits=2)
    except ValueError as exc:
        assert "stratified evaluation" in str(exc)
    else:
        raise AssertionError("single-class cohort should fail")
