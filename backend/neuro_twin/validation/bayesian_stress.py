"""Posterior-predictive stress testing.

Stress tests reuse a baseline posterior while perturbing the synthetic data-
generating process.  They answer: "does the posterior predictive distribution
remain useful under controlled shifts?" They do not prove transportability or
clinical robustness.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import copy

import numpy as np

from neuro_twin.hierarchical.joint_map import SubjectSeries
from neuro_twin.hierarchical.bayesian_hierarchical import (
    posterior_predictive_simulation,
    posterior_predictive_checks,
)


@dataclass(frozen=True)
class StressScenario:
    name: str
    measurement_noise_scale: float = 1.0
    process_noise_scale: float = 1.0
    missingness_rate: float = 0.0


def _stressed_subject(subject: SubjectSeries, scenario: StressScenario, rng: np.random.Generator) -> SubjectSeries:
    s = copy.deepcopy(subject)
    R = np.asarray(s.R, dtype=float) * float(scenario.measurement_noise_scale)
    q = None if s.process_noise_spectral_density is None else np.asarray(s.process_noise_spectral_density, dtype=float) * float(scenario.process_noise_scale)
    y = np.asarray(s.observations, dtype=float).copy()
    rate = float(scenario.missingness_rate)
    if not (0.0 <= rate < 1.0):
        raise ValueError("missingness_rate must be in [0,1)")
    if rate > 0:
        miss = rng.random(y.shape) < rate
        y[miss] = np.nan
    return SubjectSeries(
        subject_id=s.subject_id,
        cohort=s.cohort,
        time=s.time,
        observations=y,
        observation_specs=s.observation_specs,
        R=R,
        initial_state_prior=s.initial_state_prior,
        initial_state_covariance=s.initial_state_covariance,
        initial_state_mean_for_filter=s.initial_state_mean_for_filter,
        process_noise_spectral_density=q,
        random_effect_design=s.random_effect_design,
        random_effect_covariance=s.random_effect_covariance,
    )


def posterior_predictive_stress_test(
    subjects: list[SubjectSeries],
    posterior_samples: dict[str, np.ndarray],
    scenarios: list[StressScenario],
    *,
    seed: int = 0,
    n_draws: int | None = None,
    interval: float = 0.95,
) -> dict[str, dict[str, Any]]:
    """Evaluate baseline posterior under controlled perturbations."""
    if not scenarios:
        raise ValueError("at least one scenario is required")
    results: dict[str, dict[str, Any]] = {}
    for i, sc in enumerate(scenarios):
        rng = np.random.default_rng(seed + i)
        stressed = [_stressed_subject(s, sc, rng) for s in subjects]
        ppc = posterior_predictive_checks(
            stressed,
            posterior_samples,
            seed=seed + 100 + i,
            n_draws=n_draws,
            interval=interval,
        )
        results[sc.name] = {
            "scenario": sc,
            "ppc": ppc,
            "coverage_delta_from_baseline": None,
        }
    baseline = results.get("baseline")
    if baseline is not None:
        base_cov = float(baseline["ppc"]["aggregate_coverage"])
        for rec in results.values():
            rec["coverage_delta_from_baseline"] = float(rec["ppc"]["aggregate_coverage"] - base_cov)
    return results


def stress_gate(results: dict[str, dict[str, Any]], *, minimum_coverage: float = 0.80) -> dict[str, Any]:
    """Flag stress scenarios with severe predictive under-coverage."""
    failed = []
    for name, rec in results.items():
        cov = float(rec["ppc"]["aggregate_coverage"])
        if np.isfinite(cov) and cov < minimum_coverage:
            failed.append(name)
    return {
        "status": "PASS" if not failed else "FAIL",
        "minimum_coverage": float(minimum_coverage),
        "failed_scenarios": failed,
        "scope": "predictive stress gate; not clinical validation",
    }
