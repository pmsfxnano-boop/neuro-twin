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



def test_stratified_folds_are_deterministic_randomized_and_balanced():
    from neuro_twin.disease.evaluation import _stratified_subject_folds

    labels = np.asarray([0] * 12 + [1] * 12)
    folds_a = _stratified_subject_folds(labels, 4, random_seed=31)
    folds_b = _stratified_subject_folds(labels, 4, random_seed=31)
    folds_c = _stratified_subject_folds(labels, 4, random_seed=32)

    assert all(
        np.array_equal(a_test, b_test)
        for (_, a_test), (_, b_test) in zip(folds_a, folds_b)
    )
    assert any(
        not np.array_equal(a_test, c_test)
        for (_, a_test), (_, c_test) in zip(folds_a, folds_c)
    )

    all_test_indices = []
    for _, test in folds_a:
        all_test_indices.extend(test.tolist())
        assert set(labels[test]) == {0, 1}
    assert sorted(all_test_indices) == list(range(len(labels)))


def test_permutation_null_rebuilds_folds_for_permuted_labels(monkeypatch):
    import neuro_twin.disease.evaluation as evaluation

    records = [
        SubjectDiseaseRecord(
            subject_id=f"s{i:02d}",
            label=int(i >= 15),
            features=(float(i % 5), float((i * 7) % 11)),
        )
        for i in range(30)
    ]
    original = evaluation._stratified_subject_folds
    observed_labels = []
    observed_seeds = []

    def recording_folds(y, n_splits, *, random_seed=0):
        observed_labels.append(tuple(np.asarray(y, dtype=int).tolist()))
        observed_seeds.append(random_seed)
        return original(y, n_splits, random_seed=random_seed)

    monkeypatch.setattr(evaluation, "_stratified_subject_folds", recording_folds)
    result = evaluation.evaluate_binary_disease_endpoint(
        records,
        endpoint="permutation_fold_rebuild",
        n_splits=3,
        permutation_iterations=8,
        random_seed=101,
    )

    expected_labels = tuple(int(r.label) for r in records)
    assert observed_labels[0] == expected_labels
    assert len(observed_labels) == 1 + result.permutation_iterations
    assert any(labels != expected_labels for labels in observed_labels[1:])
    # All null replicates use the same random seed: the null should vary due
    # to permuted labels, not due to extra Monte Carlo randomness in fold creation.
    assert observed_seeds == [101] * (1 + result.permutation_iterations)
