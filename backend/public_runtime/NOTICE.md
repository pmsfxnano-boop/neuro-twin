# Public data attribution

NEURO-TWIN includes a small, derived runtime excerpt from the **UCI Parkinsons Telemonitoring** dataset (UCI Machine Learning Repository, dataset 189; DOI 10.24432/C5ZS3N), licensed CC BY 4.0.

Source dataset:
https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring

The excerpt contains subject 1's first 24 longitudinal recordings and two derived features:
- motor_UPDRS / 100
- total_UPDRS / 100

The calendar timestamps in the runtime package are deterministic computational anchors derived from the source's study-relative `test_time` values; they are not claimed to be the original calendar acquisition timestamps.

The P/Q observation projection is a **research-only coordinate mapping** and is not a clinically validated mapping from voice-derived UPDRS measurements to biological P/I/N/Q states.
