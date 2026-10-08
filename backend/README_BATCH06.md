# NEURO-TWIN Batch 06 — Differentiable Marginal Inference

## Objective

Add a statistically principled parameter-estimation layer above the exact
continuous-discrete state-space engine from Batch 05. The biological model
remains the canonical P-I-N-Q system; this batch changes only the inference
machinery.

## Added

- Exact Kalman marginal likelihood for irregular sampling.
- Joint optimisation of `x0` and `theta` by maximum marginal likelihood.
- JAX reverse-mode differentiation through:
  - the affine matrix-exponential transition;
  - the Van Loan process-noise covariance discretisation;
  - the masked Kalman recursion.
- Profile likelihood for individual parameters with chi-square likelihood-ratio
  thresholding.
- Optional NumPyro/NUTS posterior sampling using the same differentiable
  likelihood.

## Independent numerical oracle

The existing SciPy Kalman implementation remains the reference computation.
The JAX backend is required to reproduce its log-likelihood on the same data
before it can be trusted as the accelerated differentiation backend.

A 16-timepoint synthetic test with 9.375% missing entries produced:

- SciPy reference log-likelihood: `40.013831207649694`
- JAX log-likelihood: `40.013831207649744`
- absolute difference: approximately `5e-14`

The result is persisted in `evidence/batch06_benchmark.json`.

## Scientific boundaries

- Bayesian inference is conditional on the specified prior and initial-state
  conditioning.
- A posterior interval is uncertainty quantification, not proof of a
  mechanistic parameter's biological truth.
- Identifiability remains a separate gate and should be reported together
  with parameter uncertainty.
- Retrospective smoothing remains prohibited as a historical feature for
  point-in-time OOS evaluation.

## Optional dependencies

- `jax` is required for the differentiable backend.
- `numpyro` is additionally required for NUTS sampling.

The base deterministic/SciPy implementation remains functional without these
optional packages.
