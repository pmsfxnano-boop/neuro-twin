# NEURO-TWIN Runtime Input Contract v1

The web application accepts one JSON object.

Required top-level fields:

```text
dataset_id
dataset_version
source_name
source_version
access_tier
subject_id
event_id
processing_pipeline
processing_version
observations[]
operator
initial_state[4]
initial_state_covariance[4][4]
initial_theta[11]
```

`observations[]` uses the canonical `Observation` contract already present in NEURO-TWIN.

Each observation must carry:

```text
subject_id
event_id
acquisition_time
result_time (optional)
ingest_time
source
modality
observation_kind
feature
value
quality.provenance_complete=true
provenance.source_version
provenance.raw_hash
provenance.access_tier
```

The `operator` must explicitly map observed features to the four latent state variables:

```json
{
  "name": "explicit-model-name",
  "version": "1.0.0",
  "specs": [
    {
      "name": "feature_name",
      "weights": [0,0,0,0],
      "bias": 0,
      "sigma": 1
    }
  ]
}
```

No biological mapping is inferred from the feature name.

For prospective OOS mode, observations after the training cutoff are used only for OOS scoring and are not published into the live posterior envelope.
