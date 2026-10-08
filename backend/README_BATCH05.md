# NEURO-TWIN — Batch 05: Exact Longitudinal State-Space Inference

## Objective

Add a longitudinal inference layer on top of the canonical P-I-N-Q ODE. The
model remains unchanged. Because the canonical dynamics are affine-linear,
the continuous-discrete state transition can be computed exactly for irregular
observation intervals and used in a Kalman filter + Rauch-Tung-Striebel (RTS)
smoother.

## Added

- Exact affine transition `Phi, offset` from matrix exponentials.
- Exact discrete process covariance `Qd` from a continuous covariance spectral
density `Qc` using the Van Loan construction.
- Continuous-discrete Kalman filtering for irregular time grids.
- Missing-channel handling through finite-value masks per visit.
- Prediction-only visits when every observation at a time point is missing.
- Joseph-form covariance update and PSD symmetrization for numerical stability.
- RTS backward smoothing for retrospective latent-trajectory reconstruction.
- Innovation vectors, innovation covariances, observed-channel indices and
  filter log-likelihood for downstream diagnostics/model comparison.

## Scientific boundary

This is an inference layer, not a new biological model. The canonical ODE and
its parameterization remain the same. A future nonlinear/SDE backend can use
the same observation and evidence contracts without changing the data fabric.

## What is deliberately not claimed

- No biomarker is assigned a mechanistic P/I/N/Q meaning by this batch.
- No causal effect is inferred.
- A smoothed trajectory is retrospective; it must not be used as the historical
  state in a point-in-time OOS evaluation when future observations were already
  available to the smoother.

## Verification

Batch 05 adds tests for:

1. exact discrete transition versus the canonical ODE solver;
2. irregular sampling + partial missingness + a fully missing visit;
3. PSD covariance preservation and finite log-likelihood;
4. deterministic prediction-only equivalence when process noise is zero.
