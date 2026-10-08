"""Leakage-resistant temporal splitting with point-in-time availability."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from neuro_twin.data.availability import assert_no_future_information
from neuro_twin.data.schema import Observation


@dataclass(frozen=True)
class TemporalSplit:
    train: tuple[Observation, ...]
    validation: tuple[Observation, ...]
    test_oos: tuple[Observation, ...]


def temporal_split(
    observations: list[Observation] | tuple[Observation, ...],
    *,
    train_end: datetime,
    validation_end: datetime,
    use_result_time: bool = True,
) -> TemporalSplit:
    if validation_end <= train_end:
        raise ValueError("validation_end must be after train_end")

    key = (lambda x: ((x.result_time or x.acquisition_time) if use_result_time else x.acquisition_time,
                      x.acquisition_time, x.subject_id, x.event_id, x.feature))
    ordered = sorted(observations, key=key)
    def available_time(x: Observation) -> datetime:
        return (x.result_time or x.acquisition_time) if use_result_time else x.acquisition_time

    train = tuple(x for x in ordered if available_time(x) <= train_end)
    validation = tuple(x for x in ordered if train_end < available_time(x) <= validation_end)
    test = tuple(x for x in ordered if available_time(x) > validation_end)

    if train and max(available_time(x) for x in train) > train_end:
        raise AssertionError("temporal leakage into training set")
    if validation and min(available_time(x) for x in validation) <= train_end:
        raise AssertionError("temporal leakage into validation set")
    if validation and max(available_time(x) for x in validation) > validation_end:
        raise AssertionError("temporal leakage out of validation set")
    return TemporalSplit(train=train, validation=validation, test_oos=test)
