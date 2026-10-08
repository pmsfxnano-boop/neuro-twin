# NEURO-TWIN Batch 09

## Joint hierarchical raw-observation MAP + Laplace

This batch moves the population layer one level closer to the raw longitudinal observation model.

### Model boundary

For subject `s` in cohort `c`:

```text
y_s | x0_s, theta_s  -> nonlinear EKF marginal likelihood

theta_s | mu_c        -> N(mu_c, Sigma_within)
mu_c | mu0            -> N(mu0, Sigma_cohort)
mu0                   -> N(mu_prior, Sigma_global)
x0_s                  -> N(x0_prior, Sigma_x0)
```

The fitted objective is therefore a **joint MAP built on the nonlinear EKF marginal likelihood**. It is not an exact nonlinear Bayesian posterior. A future HMC/NUTS backend can replace the MAP/Laplace approximation while retaining the same hierarchy.

### Added capabilities

- joint optimization of subject-specific initial states and ODE parameters;
- cohort means with partial pooling through explicit Gaussian priors;
- global population mean with a separate prior;
- JAX autodiff gradient of the complete hierarchical objective;
- SciPy L-BFGS-B deterministic optimizer;
- Laplace covariance from the full Hessian at the MAP, with eigenvalue flooring for numerical stability;
- persistent random effects now actually enter the nonlinear observation operator as `Z @ u` in the JAX EKF backend;
- explicit random-effect prior covariance is required whenever a random-effect design is supplied.

### Important scientific correction

Batch 08 introduced a `Z` path in the EKF state augmentation, but the observation function did not yet add `Z @ u` to the predicted observation. Batch 09 corrects this inconsistency. The corrected model is:

\[
h(x, u, \theta) = g(W_x x + W_\theta\theta + b) + Zu.
\]

This change is covered by a regression test and requires an explicit positive-definite random-effect covariance.

### Validation boundary

The batch benchmark is synthetic and is used only to verify numerical behavior and parameter recovery. It must not be interpreted as evidence of clinical validity. The original validation ladder remains mandatory: equations → synthetic recovery → experimental → retrospective → longitudinal → external → prospective. See the core specification in the project PDF.
