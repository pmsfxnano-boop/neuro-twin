import numpy as np
import pytest

from neuro_twin.disease.eeg_analysis import extract_eeg_band_features


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
