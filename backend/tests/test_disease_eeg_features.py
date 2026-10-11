import numpy as np
import pytest

from neuro_twin.disease.eeg_analysis import (
    DATASET_DOI,
    DATASET_LICENSE,
    _crosscheck_openneuro_labels,
    _recording_candidates,
    _validate_nemar_index,
    _validate_nemar_store_provenance,
    extract_eeg_band_features,
)


def test_eeg_feature_extractor_identifies_alpha_dominant_signal():
    fs = 500.0
    time = np.arange(int(fs * 30)) / fs
    base = 8.0 * np.sin(2 * np.pi * 10.0 * time) + 0.25 * np.sin(2 * np.pi * 20.0 * time)
    signals = np.vstack([base + 0.02 * np.sin(time * (i + 1)) for i in range(8)])
    features = extract_eeg_band_features(signals, fs)
    assert len(features) == 12
    assert all(np.isfinite(value) for value in features.values())
    assert features["relative_alpha"] > features["relative_delta"]
    assert features["relative_alpha"] > features["relative_beta"]


def test_eeg_feature_extractor_rejects_short_or_invalid_segments():
    with pytest.raises(ValueError, match="too short"):
        extract_eeg_band_features(np.ones((4, 1000)), 500.0)
    with pytest.raises(ValueError, match="2D"):
        extract_eeg_band_features(np.ones(5000), 500.0)


def test_eeg_feature_extractor_reports_relative_band_powers_that_sum_to_one():
    fs = 500.0
    time = np.arange(int(fs * 30)) / fs
    signals = np.vstack([
        np.sin(2 * np.pi * 3.0 * time) + 0.4 * np.sin(2 * np.pi * 10.0 * time)
        for _ in range(4)
    ])
    features = extract_eeg_band_features(signals, fs)
    assert sum(features[f"relative_{band}"] for band in ("delta", "theta", "alpha", "beta", "gamma")) == pytest.approx(1.0, abs=1e-10)



def _valid_nemar_index():
    return {
        "format": "nemar-zarr-index",
        "format_version": 3,
        "dataset_id": "on004504",
        "doi": DATASET_DOI,
        "license": DATASET_LICENSE,
        "source_commit": "a" * 40,
        "updated_utc": "2026-09-06T02:27:31Z",
        "s3_uri": "s3://nemar/on004504/zarr/",
        "s3_region": "us-east-2",
        "s3_anonymous": True,
        "stores": [
            {
                "path": "sub-001/eeg/sub-001_task-eyesclosed_eeg.set",
                "zarr": "sub-001/eeg/sub-001_task-eyesclosed_eeg.zarr",
                "derived": False,
                "groups": [{"name": "eeg_250hz", "rate": 250.0}],
            }
        ],
    }


def test_nemar_index_and_store_provenance_are_validated():
    index = _valid_nemar_index()
    validated = _validate_nemar_index(index)
    assert validated["source_commit"] == "a" * 40

    candidates = _recording_candidates(index)
    selected = candidates["sub-001"][0]
    assert selected["sampling_rate_hz"] == 250.0
    assert selected["uri"] == (
        "s3://nemar/on004504/zarr/sub-001/eeg/sub-001_task-eyesclosed_eeg.zarr"
    )

    store_provenance = {
        "dataset_id": "on004504",
        "doi": DATASET_DOI,
        "license": DATASET_LICENSE,
        "source_commit": "a" * 40,
        "source_tree": "raw",
        "derived": False,
        "contract_url": "https://zarr.nemar.org/on004504/zarr/sub-001/eeg/sub-001_task-eyesclosed_eeg.zarr/",
    }
    assert _validate_nemar_store_provenance(
        store_provenance, expected_source_commit="a" * 40
    )["source_commit"] == "a" * 40

    mismatched = dict(store_provenance, source_commit="b" * 40)
    with pytest.raises(ValueError, match="source_commit"):
        _validate_nemar_store_provenance(
            mismatched, expected_source_commit="a" * 40
        )


def test_openneuro_labels_are_crosschecked_against_nemar_mirror():
    nemar = {
        "sub-001": {"label": "AD"},
        "sub-002": {"label": "FTD"},
        "sub-003": {"label": "CONTROL"},
    }
    payload = (
        "participant_id\tGender\tAge\tGroup\tMMSE\n"
        "sub-001\tF\t57\tA\t16\n"
        "sub-002\tM\t70\tF\t22\n"
        "sub-003\tM\t70\tC\t30\n"
    ).encode()
    result = _crosscheck_openneuro_labels(nemar, payload)
    assert result["subject_ids_match"] is True
    assert result["diagnosis_labels_match"] is True
    assert result["participant_count"] == 3
    assert len(result["participants_tsv_sha256"]) == 64

    incorrect = payload.replace(b"sub-002\tM\t70\tF", b"sub-002\tM\t70\tA")
    with pytest.raises(ValueError, match="diagnosis labels differ"):
        _crosscheck_openneuro_labels(nemar, incorrect)
