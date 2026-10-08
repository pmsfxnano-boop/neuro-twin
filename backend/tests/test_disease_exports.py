from neuro_twin.disease.exports import parse_authorized_export
from neuro_twin.disease.sources import ADNI_PLASMA_V1, PPMI_CORE_V1


def test_authorized_export_parser_is_strict_and_hashes_source():
    payload = (
        "PTID,VISCODE,AB42,AB40,pTau217,NfL,GFAP\n"
        "sub1,bl,1.0,2.0,3.0,4.0,5.0\n"
    ).encode()
    records, digest = parse_authorized_export(
        payload,
        subject_field=ADNI_PLASMA_V1.subject_field,
        visit_field=ADNI_PLASMA_V1.visit_field,
        field_map={name: name for name in ADNI_PLASMA_V1.canonical_fields},
        required_fields=ADNI_PLASMA_V1.canonical_fields,
    )
    assert records[0].values["pTau217"] == 3.0
    assert len(digest) == 64


def test_authorized_export_rejects_duplicate_subject_visit():
    payload = (
        "subject_id,visit,alphaSyn_SAA,NfL,MDS_UPDRS\n"
        "p1,baseline,1,2,3\n"
        "p1,baseline,1,2,4\n"
    ).encode()
    try:
        parse_authorized_export(
            payload,
            subject_field=PPMI_CORE_V1.subject_field,
            visit_field=PPMI_CORE_V1.visit_field,
            field_map={name: name for name in PPMI_CORE_V1.canonical_fields if name in {"alphaSyn_SAA", "NfL", "MDS_UPDRS"}},
            required_fields=PPMI_CORE_V1.canonical_fields,
        )
    except ValueError as exc:
        assert "duplicate subject/visit" in str(exc)
    else:
        raise AssertionError("duplicate subject/visit was accepted")
