from dataclasses import replace

import numpy as np

from neuro_twin.disease.parkinson_eeg_analysis import (
    EXTERNAL_COHORT,
    PRIMARY_COHORT,
    _regression_folds,
    _regression_metrics,
    _ridge_oof_predictions,
    _select_rest_candidates,
    crosscheck_participants,
    parse_participants,
)


def _tiny_primary_payload():
    return (
        "participant_id\tgroup\tage\tsex\tupdrs_part_iii\tupdrs_total\tmoca\tdisease_duration\tledd\n"
        "sub-001\tHC\t64\tF\t0\t0\t29\tn/a\tn/a\n"
        "sub-002\tPD\t70\tM\t24\t41\t25\t60\t600\n"
    ).encode()


def test_parkinson_source_profiles_are_version_pinned_and_expected_counts_are_explicit():
    assert PRIMARY_COHORT.openneuro_id == "openneuro-ds007526"
    assert PRIMARY_COHORT.openneuro_version == "1.0.2"
    assert PRIMARY_COHORT.expected_n == 144
    assert PRIMARY_COHORT.expected_groups == {"PD": 116, "CONTROL": 28}
    assert EXTERNAL_COHORT.openneuro_id == "openneuro-ds008768"
    assert EXTERNAL_COHORT.openneuro_version == "1.0.0"
    assert EXTERNAL_COHORT.expected_n == 312
    assert EXTERNAL_COHORT.expected_groups == {"PD": 202, "CONTROL": 110}
    assert "/main/" not in PRIMARY_COHORT.openneuro_participants_url
    assert "/main/" not in EXTERNAL_COHORT.openneuro_participants_url


def test_parkinson_participant_parser_maps_group_but_keeps_clinical_fields_separate():
    profile = replace(PRIMARY_COHORT, expected_n=2, expected_groups={"CONTROL": 1, "PD": 1})
    payload = _tiny_primary_payload()
    parsed = parse_participants(payload, profile)

    assert set(parsed) == {"sub-001", "sub-002"}
    assert parsed["sub-001"]["label"] == "CONTROL"
    assert parsed["sub-002"]["label"] == "PD"
    assert parsed["sub-002"]["age_years"] == 70.0
    assert parsed["sub-002"]["updrs_part_iii"] == 24.0
    assert parsed["sub-002"]["disease_duration_months"] == 60.0
    assert parsed["sub-002"]["ledd_mg"] == 600.0
    assert parsed["sub-001"]["updrs_part_iii"] == 0.0
    assert parsed["sub-001"]["disease_duration_months"] is None


def test_nemar_and_openneuro_metadata_crosscheck_hard_fails_on_source_drift():
    profile = replace(PRIMARY_COHORT, expected_n=2, expected_groups={"CONTROL": 1, "PD": 1})
    payload = _tiny_primary_payload()
    nemar = parse_participants(payload, profile)
    source = parse_participants(payload, profile)
    valid = crosscheck_participants(
        nemar,
        source,
        nemar_payload=payload,
        source_payload=payload,
        profile=profile,
    )
    assert valid["participant_count"] == 2
    assert valid["subject_ids_match"] is True
    assert valid["clinical_fields_crosschecked"] is True
    assert len(valid["openneuro_participants_sha256"]) == 64

    source["sub-002"]["updrs_part_iii"] = 99.0
    try:
        crosscheck_participants(
            nemar,
            source,
            nemar_payload=payload,
            source_payload=payload,
            profile=profile,
        )
    except ValueError as exc:
        assert "metadata mismatch" in str(exc)
    else:
        raise AssertionError("metadata mirror drift must fail closed")


def test_recording_selection_prefers_earliest_raw_rest_recording_not_walking_or_processed():
    candidates = {
        "sub-001": [
            {
                "record_path": "sub-001/ses-02/eeg/sub-001_ses-02_task-rest_eeg.eeg",
                "group_name": "eeg_250hz",
                "sampling_rate_hz": 250.0,
                "is_processed": False,
            },
            {
                "record_path": "sub-001/ses-01/eeg/sub-001_ses-01_task-rest_eeg.eeg",
                "group_name": "eeg_500hz",
                "sampling_rate_hz": 500.0,
                "is_processed": False,
            },
            {
                "record_path": "sub-001/ses-01/eeg/sub-001_ses-01_task-walk_eeg.eeg",
                "group_name": "eeg_250hz",
                "sampling_rate_hz": 250.0,
                "is_processed": False,
            },
            {
                "record_path": "sub-001/ses-01/derivatives/sub-001_ses-01_task-rest_cleaned.zarr",
                "group_name": "eeg_250hz",
                "sampling_rate_hz": 250.0,
                "is_processed": True,
            },
        ]
    }
    selected = _select_rest_candidates(candidates, "sub-001")
    assert len(selected) == 2
    assert "ses-01" in selected[0]["record_path"]
    assert "task-rest" in selected[0]["record_path"]
    assert not selected[0]["is_processed"]


def test_subject_level_regression_folds_partition_every_subject_once():
    y = np.asarray([float(x) for x in range(31)])
    folds = _regression_folds(y, n_splits=5, seed=12)
    test_indices = [int(index) for _, test in folds for index in test]
    assert sorted(test_indices) == list(range(len(y)))
    assert all(len(train) + len(test) == len(y) for train, test in folds)
    assert max(len(test) for _, test in folds) - min(len(test) for _, test in folds) <= 1


def test_oos_ridge_severity_regression_uses_finite_train_fold_predictions():
    rng = np.random.default_rng(23)
    X = rng.normal(size=(80, 4))
    y = 18.0 + 4.0 * X[:, 0] - 2.5 * X[:, 1] + rng.normal(scale=1.5, size=80)
    predicted, fold_info = _ridge_oof_predictions(X, y, alpha=1.0, seed=41)
    metrics = _regression_metrics(y, predicted)

    assert predicted.shape == y.shape
    assert np.isfinite(predicted).all()
    assert len(fold_info) == 5
    assert metrics["oos_rmse_updrs_points"] > 0
    assert metrics["oos_rmse_updrs_points"] < 5
    assert -1.0 <= metrics["oos_r2"] <= 1.0
    assert -1.0 <= metrics["oos_spearman_rho"] <= 1.0
