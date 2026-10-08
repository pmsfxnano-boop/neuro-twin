"""Numerical regression tests only; fixtures are not production data sources."""
from datetime import datetime, timedelta, timezone

import numpy as np
from fastapi.testclient import TestClient

from neuro_twin.api_server import app
from neuro_twin.runtime.runner import RuntimeRunRequest, execute_runtime


def _payload(subject='TEST'):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    features = ['P_obs', 'I_obs', 'N_obs', 'Q_obs']
    values = [[.80, .20, .10, .05], [.79, .20, .10, .05], [.79, .20, .11, .05]]
    observations = []
    for i, row in enumerate(values):
        t = base + timedelta(days=120 * i)
        for j, (feature, value) in enumerate(zip(features, row)):
            observations.append({
                'subject_id': subject, 'event_id': f'V{i+1}',
                'acquisition_time': t.isoformat(), 'result_time': (t + timedelta(hours=1)).isoformat(),
                'ingest_time': (t + timedelta(hours=2)).isoformat(), 'source': 'unit-test', 'center': 'C1',
                'modality': 'research', 'observation_kind': 'clinical', 'feature': feature, 'value': value,
                'unit': 'a.u.', 'uncertainty': 0.05, 'missing': False,
                'quality': {'provenance_complete': True},
                'provenance': {'source_name': 'unit-test', 'source_version': '1', 'raw_hash': 'a' * 64,
                               'processing_pipeline': 'test', 'processing_version': '1', 'access_tier': 'public_dataset'},
            })
    return {
        'dataset_id': 'unit-test', 'dataset_version': '1', 'source_name': 'unit-test', 'source_version': '1',
        'access_tier': 'public_dataset', 'subject_id': subject, 'event_id': 'V3',
        'processing_pipeline': 'test', 'processing_version': '1', 'raw_hashes': ['a' * 64],
        'observations': observations,
        'operator': {'name': 'explicit-test-operator', 'version': '1.0', 'specs': [
            {'name': f, 'weights': [1 if k == j else 0 for k in range(4)], 'sigma': 0.05}
            for j, f in enumerate(features)
        ]},
        'initial_state': [.8, .2, .1, .05],
        'initial_state_covariance': np.eye(4).tolist(),
        'initial_theta': [.02, .05, .04, .03, .06, .02, .01, .05, .03, .02, .04],
        'oos_fraction': .8, 'analysis_mode': 'prospective_oos',
    }


def test_runtime_executes_and_persists_identifiability_diagnostics():
    result = execute_runtime(RuntimeRunRequest.model_validate(_payload()))
    assert result.runtime_class.value == 'research_observational'
    assert result.oos is not None and result.oos.status == 'PASS'
    assert len(result.observations) == 8  # train envelope only: 2 visits × 4 channels
    assert result.oos.metrics['identifiability_rank'] > 0
    assert result.oos.metrics['identifiability_parameter_count'] == 15.0
    assert 0.0 < result.oos.metrics['identifiability_rank_fraction'] <= 1.0
    assert result.evidence['run']['identifiability']['effective_rank'] > 0
    assert result.evidence['run']['identifiability']['status'] in {'FULL_RANK', 'RANK_DEFICIENT'}


def test_http_run_publishes_live_result():
    r = TestClient(app).post('/v1/runtime/run', json=_payload('HTTP'))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['status'] == 'PUBLISHED'
    assert body['summary']['state_status'] == 'LIVE_RESEARCH'
    assert body['summary']['provenance']['result_hash']


def test_reference_runtime_is_hard_disabled():
    r = TestClient(app).post('/v1/run/reference')
    assert r.status_code == 410
    assert 'removed' in r.json()['detail'].lower()
