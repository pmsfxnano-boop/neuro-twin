"""Multimodal observation assembly.

A model configuration explicitly chooses which canonical features are observed.
The assembler keeps modality, assay, unit and provenance attached to every
observation and never silently maps a biomarker to a latent P/I/N/Q state.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np

from neuro_twin.data.schema import Observation


@dataclass(frozen=True)
class ObservationPanel:
    subject_id: str
    event_id: str
    timestamp: datetime
    feature_names: tuple[str, ...]
    values: np.ndarray
    uncertainties: np.ndarray
    source_ids: tuple[str, ...]


def assemble_panel(observations: Iterable[Observation], *, features: tuple[str, ...], require_all: bool = False) -> ObservationPanel:
    obs = list(observations)
    if not obs:
        raise ValueError("no observations")
    subject_ids = {o.subject_id for o in obs}
    event_ids = {o.event_id for o in obs}
    if len(subject_ids) != 1 or len(event_ids) != 1:
        raise ValueError("panel must contain one subject and one event")
    by_feature = {o.feature: o for o in obs}
    if require_all and any(f not in by_feature for f in features):
        missing = [f for f in features if f not in by_feature]
        raise ValueError(f"missing required observations: {missing}")
    selected = [by_feature[f] for f in features if f in by_feature]
    if not selected:
        raise ValueError("none of requested features are present")
    if any(isinstance(o.value, list) for o in selected):
        raise ValueError("panel assembly requires scalar observations")
    values = np.array([float(o.value) for o in selected], dtype=float)
    uncertainties = np.array([float(o.uncertainty or 0.0) for o in selected], dtype=float)
    timestamp = max((o.result_time or o.acquisition_time) for o in selected)
    return ObservationPanel(
        subject_id=selected[0].subject_id,
        event_id=selected[0].event_id,
        timestamp=timestamp,
        feature_names=tuple(o.feature for o in selected),
        values=values,
        uncertainties=uncertainties,
        source_ids=tuple(o.provenance.source_record_id or o.event_id for o in selected),
    )


def group_by_subject_event(observations: Iterable[Observation]) -> dict[tuple[str, str], list[Observation]]:
    groups: dict[tuple[str, str], list[Observation]] = defaultdict(list)
    for observation in observations:
        groups[(observation.subject_id, observation.event_id)].append(observation)
    return dict(groups)
