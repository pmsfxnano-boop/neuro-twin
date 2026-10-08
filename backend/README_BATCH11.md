# NEURO-TWIN — Batch 11

## Bayesian diagnostics, non-centered hierarchy, and posterior-predictive stress testing

### Added

- Backend-agnostic MCMC diagnostics: R-hat, bulk ESS, tail ESS, divergence count, E-BFMI.
- SVI convergence diagnostics from ELBO loss trajectories.
- Transparent MAP/SVI/NUTS scorecard; missing metrics are never imputed.
- Non-centered hierarchical parameterization contract using Cholesky transforms.
- Posterior-predictive stress testing under controlled measurement-noise, process-noise, and missingness shifts.
- Predictive stress gate with explicit non-clinical scope.

### Scientific boundaries

1. NUTS/SVI remain optional and were **not executed** if NumPyro is unavailable.
2. R-hat/ESS/E-BFMI are computational diagnostics, not evidence of clinical validity.
3. Stress testing reuses a baseline posterior under controlled shifts; it does not establish transportability.
4. No biomarker-to-state mapping is introduced beyond the explicit observation registry.

### Validation

Batch 11 tests exercise the new diagnostics and stress modules using synthetic data. They do not claim execution of NUTS/SVI when NumPyro is absent.

The non-centered backend is provided as `build_hierarchical_model_noncentered(...)` and uses explicit standard-normal latent variables plus Cholesky factors. It is algebraically equivalent to the centered hierarchy when the same covariance matrices are used.

`diagnostics_from_numpyro(...)` can extract R-hat/ESS/divergence/energy diagnostics directly from a NumPyro `MCMC` object once NumPyro is installed.
