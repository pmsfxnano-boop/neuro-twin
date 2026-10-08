# NEURO-TWIN — Batch 10

## Full Bayesian hierarchical nonlinear state-space layer

Batch 10 adds an optional Bayesian backend above the existing Batch 09 joint MAP architecture.

### What changed

The posterior model explicitly samples:

- global population theta
- cohort-specific theta
- subject-specific theta
- subject initial latent state
- persistent observation random effects when configured
- process innovations when continuous process noise is configured
- observed channels under the nonlinear observation operator

The P-I-N-Q continuous-time dynamics are not changed. The affine transition is evaluated exactly with a matrix exponential, and process noise is discretised with Van Loan.

### Important boundary

The Batch 09 backend used an EKF marginal likelihood. Batch 10 does **not** use that EKF approximation in its posterior graph. Instead, it samples latent trajectories explicitly, so the nonlinear observation map is represented directly.

This is still a research inference engine, not clinical validation. The biological meaning of each observation channel remains whatever is explicitly specified in the observation registry; no biomarker-to-latent-state mapping is invented here.

### NUTS and SVI

The code supports:

```python
sample_hierarchical_nuts(...)
fit_hierarchical_svi(...)
```

NumPyro is intentionally optional. In the current execution environment NumPyro is not installed, so NUTS and SVI were not falsely reported as executed.

### Posterior predictive checks

`posterior_predictive_checks(...)` produces predictive intervals, coverage and RMSE. `calibration_gate(...)` reports PASS / FAIL / INCONCLUSIVE and refuses to declare a calibration pass when the sample is too small.

### Identifiability

`posterior_parameter_identifiability(...)` computes the pooled posterior covariance eigen-spectrum, condition number and entropy effective rank. This complements the local Fisher/SVD machinery already present in earlier batches.

### Validation

Batch 10 includes tests for:

- exact affine transition shape/finite values
- posterior predictive simulation with missing observations
- calibration gate behavior
- posterior information geometry
- truthful NumPyro dependency gating
- NumPyro graph contract reporting when available
