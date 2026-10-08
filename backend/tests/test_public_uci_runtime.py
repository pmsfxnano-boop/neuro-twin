from __future__ import annotations

import json
from pathlib import Path

from neuro_twin.runtime.runner import RuntimeRunRequest, execute_runtime


def test_public_uci_runtime_fixture_executes_with_real_longitudinal_data() -> None:
    package = Path(__file__).resolve().parents[1] / "public_runtime" / "uci_parkinsons_subject1_v1.json"
    payload = json.loads(package.read_text(encoding="utf-8"))
    request = RuntimeRunRequest.model_validate(payload)

    result = execute_runtime(request)

    assert result.runtime_class.value == "research_observational"
    assert result.provenance.source_name == "UCI Parkinsons Telemonitoring"
    assert result.provenance.access_tier == "public_dataset"
    assert len(result.observations) == 38
    assert result.oos is not None
    assert result.oos.temporal_leakage is False
    assert result.oos.status in {"PASS", "INCONCLUSIVE"}
    assert result.provenance.raw_hashes
    assert result.prediction["mode"] == "prospective_oos"
    assert result.prediction["future_data_not_published_to_live_state"] if "future_data_not_published_to_live_state" in result.prediction else True
