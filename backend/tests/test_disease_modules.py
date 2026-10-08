from neuro_twin.disease.coverage import evaluate_module_coverage
from neuro_twin.disease.modules import Disease, get_disease_module
from neuro_twin.disease.readiness import assess_dataset_readiness


def test_ad_requires_pathology_inflammation_neurodegeneration_and_cognition():
    module = get_disease_module(Disease.ALZHEIMER)
    result = evaluate_module_coverage(
        module,
        {"pTau217", "Abeta42_40", "GFAP", "NfL", "cognition"},
    )
    assert result.status == "READY_FOR_FULL_OBSERVABILITY"
    assert result.observed_states == ("I", "N", "P", "Q")


def test_pd_diagnosis_label_is_not_an_observation():
    module = get_disease_module(Disease.PARKINSON)
    result = evaluate_module_coverage(module, {"diagnosis_group", "MDS_UPDRS"})
    assert result.status == "PARTIAL_OBSERVABILITY"
    assert "Q" in result.observed_states
    assert "N" in result.unobserved_states


def test_public_ad_cohort_is_not_ready_for_full_reconstruction():
    result = assess_dataset_readiness(
        dataset_id="openneuro-ds004504",
        dataset_version="1.0.6",
        disease=Disease.ALZHEIMER,
    )
    assert result.status == "PARTIAL_OBSERVABILITY"
    assert "Q" in result.coverage.observed_states
    assert "P" in result.coverage.unobserved_states


def test_public_pd_imaging_cohort_is_not_ready_for_full_reconstruction():
    result = assess_dataset_readiness(
        dataset_id="openneuro-ds005892",
        dataset_version="1.0.0",
        disease=Disease.PARKINSON,
    )
    assert result.status == "NOT_READY"
    assert result.coverage.observed_states == ()
