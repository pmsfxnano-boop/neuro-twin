from neuro_twin.disease.cohorts import parse_participants_tsv, summarize_cohort
from neuro_twin.disease.public import PUBLIC_COHORTS


def test_ad_participants_parser_keeps_diagnosis_outside_observations():
    payload = (
        "participant_id\tGender\tAge\tGroup\tMMSE\n"
        "sub-001\tF\t57\tA\t16\n"
        "sub-002\tM\t70\tC\t30\n"
    ).encode()
    records, raw_sha = parse_participants_tsv(
        payload,
        cohort="alzheimer_test",
        source_version="1.0.0",
        diagnosis_field="Group",
        diagnosis_map={"A": "AD", "C": "CONTROL"},
        severity_fields={"mmse": "MMSE"},
        age_field="Age",
        sex_field="Gender",
    )
    assert len(records) == 2
    assert records[0].diagnosis == "AD"
    assert records[0].severity["mmse"] == 16.0
    assert len(raw_sha) == 64

    summary = summarize_cohort(
        records,
        source_version="1.0.0",
        raw_sha256=raw_sha,
    )
    assert summary.n_total == 2
    assert summary.diagnosis_counts == {"AD": 1, "CONTROL": 1}
    assert summary.severity_summary["mmse"]["mean"] == 23.0


def test_public_specs_are_explicit_and_expected_sizes_are_pinned():
    ad = PUBLIC_COHORTS["alzheimer_ds004504_v1.0.6"]
    pd = PUBLIC_COHORTS["parkinson_ds005892_v1.0.0"]

    assert ad.expected_n == 88
    assert pd.expected_n == 55
    assert ad.diagnosis_map["A"] == "AD"
    assert pd.diagnosis_map["PD-MCI"] == "PD_MCI"
