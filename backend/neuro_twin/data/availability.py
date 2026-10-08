"""Point-in-time availability and temporal integrity guards.

A measurement is admissible to a historical state only if it was available at
that inference cutoff. Acquisition time alone is insufficient because delayed
laboratory/processing results can arrive after acquisition.
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterable

from neuro_twin.data.schema import Observation


def available_at(observation: Observation, cutoff: datetime) -> bool:
    available = observation.result_time or observation.acquisition_time
    return available <= cutoff


def point_in_time_filter(
    observations: Iterable[Observation],
    cutoff: datetime,
) -> tuple[Observation, ...]:
    """Return only observations that were genuinely available at ``cutoff``."""
    return tuple(sorted(
        (obs for obs in observations if available_at(obs, cutoff)),
        key=lambda x: (x.acquisition_time, x.subject_id, x.event_id, x.feature),
    ))


def assert_no_future_information(
    observations: Iterable[Observation],
    cutoff: datetime,
) -> None:
    """Raise if any observation would leak information past the cutoff."""
    offenders = []
    for obs in observations:
        available = obs.result_time or obs.acquisition_time
        if available > cutoff:
            offenders.append((obs.subject_id, obs.event_id, obs.feature, available.isoformat()))
    if offenders:
        raise ValueError(f"point-in-time leakage detected: {offenders[:5]}")
