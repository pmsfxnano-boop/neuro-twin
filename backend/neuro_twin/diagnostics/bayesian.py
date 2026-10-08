"""Backend-agnostic Bayesian diagnostics.

These routines consume arrays exported by NumPyro/ArviZ (or synthetic arrays in
unit tests) and intentionally do not require NumPyro at import time.  They are
engineering diagnostics, not clinical validation criteria.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BayesianDiagnostics:
    parameter_names: tuple[str, ...]
    rhat: dict[str, float]
    ess_bulk: dict[str, float]
    ess_tail: dict[str, float]
    divergence_count: int
    energy_ebfmi: float | None
    warnings: tuple[str, ...]


def _rank_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    flat = x.reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(flat) + 1, dtype=float)
    # Blom transform; finite for 1..N.
    p = (ranks - 0.375) / (len(flat) + 0.25)
    from scipy.stats import norm
    return norm.ppf(p).reshape(x.shape)


def _rhat(chains: np.ndarray) -> float:
    """Split-free classical R-hat for shape (chain, draw)."""
    a = np.asarray(chains, dtype=float)
    if a.ndim != 2 or a.shape[0] < 2 or a.shape[1] < 4:
        return float("nan")
    m, n = a.shape
    chain_means = a.mean(axis=1)
    W = a.var(axis=1, ddof=1).mean()
    B = n * np.var(chain_means, ddof=1)
    var_hat = ((n - 1) / n) * W + B / n
    return float(np.sqrt(var_hat / W)) if W > 0 else (1.0 if B == 0 else float("inf"))


def _ess_bulk(chains: np.ndarray) -> float:
    """Geyer's initial-positive-sequence ESS approximation."""
    a = np.asarray(chains, dtype=float)
    if a.ndim != 2 or a.shape[1] < 4:
        return float("nan")
    # Center globally, then average autocovariances across chains.
    a = a - np.mean(a, axis=1, keepdims=True)
    m, n = a.shape
    var = float(np.mean(np.sum(a * a, axis=1) / max(n - 1, 1)))
    if var <= 0 or not np.isfinite(var):
        return float(m * n)
    acov = []
    for lag in range(1, n):
        covs = np.sum(a[:, : n - lag] * a[:, lag:], axis=1) / max(n - lag, 1)
        acov.append(float(np.mean(covs) / var))
    tau = 1.0
    for k in range(0, len(acov) - 1, 2):
        pair = acov[k] + acov[k + 1]
        if not np.isfinite(pair) or pair < 0:
            break
        tau += 2.0 * pair
    return float(max(1.0, m * n / tau))


def _ess_tail(chains: np.ndarray) -> float:
    z = _rank_normalize(chains)
    return _ess_bulk(z)


def _ebfmi(energy: np.ndarray) -> float | None:
    e = np.asarray(energy, dtype=float)
    if e.ndim == 2:
        e = e.reshape(-1)
    if e.size < 4:
        return None
    delta = np.diff(e)
    denom = np.var(e)
    if denom <= 0 or not np.isfinite(denom):
        return None
    return float(np.mean(delta**2) / denom)


def compute_mcmc_diagnostics(
    posterior_by_parameter_and_chain: dict[str, np.ndarray],
    *,
    divergences: np.ndarray | None = None,
    energy: np.ndarray | None = None,
    rhat_warn: float = 1.01,
    ess_warn: float = 400.0,
) -> BayesianDiagnostics:
    """Compute R-hat/ESS/divergence/E-BFMI diagnostics.

    Arrays must have shape ``(chains, draws)``.  Divergences may be boolean
    arrays matching ``(chains, draws)``.  The thresholds are conservative
    engineering defaults and are not a substitute for scientific judgment.
    """
    names = tuple(sorted(posterior_by_parameter_and_chain))
    rhat: dict[str, float] = {}
    ess_bulk: dict[str, float] = {}
    ess_tail: dict[str, float] = {}
    warnings: list[str] = []
    for name in names:
        arr = np.asarray(posterior_by_parameter_and_chain[name], dtype=float)
        r = _rhat(arr)
        eb = _ess_bulk(arr)
        et = _ess_tail(arr)
        rhat[name], ess_bulk[name], ess_tail[name] = r, eb, et
        if np.isfinite(r) and r > rhat_warn:
            warnings.append(f"{name}: R-hat={r:.4f} > {rhat_warn}")
        if np.isfinite(eb) and eb < ess_warn:
            warnings.append(f"{name}: bulk ESS={eb:.1f} < {ess_warn:.1f}")
        if np.isfinite(et) and et < ess_warn:
            warnings.append(f"{name}: tail ESS={et:.1f} < {ess_warn:.1f}")
    div_count = int(np.sum(np.asarray(divergences, dtype=bool))) if divergences is not None else 0
    if div_count:
        warnings.append(f"divergences={div_count}")
    ebfmi = None if energy is None else _ebfmi(energy)
    if ebfmi is not None and ebfmi < 0.3:
        warnings.append(f"E-BFMI={ebfmi:.3f} < 0.3")
    return BayesianDiagnostics(names, rhat, ess_bulk, ess_tail, div_count, ebfmi, tuple(warnings))


def svi_diagnostics(elbo_losses: np.ndarray, *, tail_fraction: float = 0.1) -> dict[str, Any]:
    """Summarize SVI convergence without assuming monotonic ELBO."""
    losses = np.asarray(elbo_losses, dtype=float).reshape(-1)
    if losses.size < 5 or not np.all(np.isfinite(losses)):
        raise ValueError("elbo_losses must contain at least five finite values")
    tail = losses[max(0, int((1.0 - tail_fraction) * losses.size)) :]
    slope = float(np.polyfit(np.arange(tail.size), tail, 1)[0]) if tail.size >= 2 else float("nan")
    scale = float(np.std(tail))
    center = float(np.mean(tail))
    rel_slope = float(abs(slope) / max(abs(center), 1e-12))
    return {
        "steps": int(losses.size),
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "best_loss": float(np.min(losses)),
        "tail_mean": center,
        "tail_std": scale,
        "tail_slope": slope,
        "relative_tail_slope": rel_slope,
        "stable": bool(rel_slope < 1e-4 or scale < max(1e-6, abs(center) * 1e-4)),
    }


def compare_backend_scores(records: dict[str, dict[str, float]], *,
                           lower_is_better: tuple[str, ...] = ("rmse", "mae", "nll"),
                           higher_is_better: tuple[str, ...] = ("coverage", "elpd")) -> dict[str, Any]:
    """Create a transparent normalized scorecard across MAP/NUTS/SVI/etc.

    Missing metrics are left unscored rather than imputed.  Each metric gets
    equal weight; this is a comparison aid, not a statistical model-selection
    theorem.
    """
    if not records:
        raise ValueError("records cannot be empty")
    metric_names = [*lower_is_better, *higher_is_better]
    per_metric: dict[str, dict[str, float]] = {}
    for metric in metric_names:
        vals = {name: rec[metric] for name, rec in records.items() if metric in rec and np.isfinite(rec[metric])}
        if len(vals) < 2:
            continue
        arr = np.array(list(vals.values()), dtype=float)
        lo, hi = float(np.min(arr)), float(np.max(arr))
        span = hi - lo
        scores: dict[str, float] = {}
        for name, v in vals.items():
            if span == 0:
                scores[name] = 1.0
            elif metric in lower_is_better:
                scores[name] = (hi - v) / span
            else:
                scores[name] = (v - lo) / span
        per_metric[metric] = scores
    totals = {name: 0.0 for name in records}
    counts = {name: 0 for name in records}
    for scores in per_metric.values():
        for name, score in scores.items():
            totals[name] += score
            counts[name] += 1
    mean_scores = {name: (totals[name] / counts[name] if counts[name] else float("nan")) for name in records}
    ranking = sorted(mean_scores, key=lambda n: (-mean_scores[n] if np.isfinite(mean_scores[n]) else float("inf")))
    return {"per_metric_scores": per_metric, "mean_scores": mean_scores, "ranking": ranking}


def diagnostics_from_numpyro(mcmc: Any) -> BayesianDiagnostics:
    """Extract standard diagnostics from a NumPyro MCMC object when available."""
    samples = mcmc.get_samples(group_by_chain=True)
    selected = {k: np.asarray(v) for k, v in samples.items() if np.asarray(v).ndim >= 2}
    extra = mcmc.get_extra_fields(group_by_chain=True)
    divergences = extra.get("diverging")
    energy = extra.get("energy")
    # Flatten non-scalar tensor dimensions by treating the last dimensions as
    # independent parameter coordinates. Diagnostics are returned under an
    # explicit coordinate suffix to avoid silently aggregating dimensions.
    scalar_chains: dict[str, np.ndarray] = {}
    for name, arr in selected.items():
        if arr.ndim == 2:
            scalar_chains[name] = arr
        else:
            for idx in np.ndindex(arr.shape[2:]):
                scalar_chains[f"{name}{idx}"] = arr[(slice(None), slice(None), *idx)]
    return compute_mcmc_diagnostics(scalar_chains, divergences=divergences, energy=energy)
