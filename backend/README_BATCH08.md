# NEURO-TWIN Batch 08

## Population hierarchy + external validation + higher-order nonlinear inference

This batch adds three research-grade capabilities without changing the canonical P-I-N-Q dynamics:

1. **UKF**: higher-order sigma-point alternative to the EKF for nonlinear observation operators, with exact affine P-I-N-Q state transition. It is a higher-order approximation, not a claim of universal superiority over EKF.
2. **Hierarchical empirical-Bayes partial pooling**: subject-level Gaussian likelihood summaries are pooled through cohort means, a within-cohort covariance and a cohort-level covariance, with explicit shrinkage toward the population.
3. **Strict leave-one-cohort-out validation**: held-out cohorts are never included in the training index; predictive RMSE/MAE/bias, 95% coverage and log-score are available.

### Scientific boundary

The hierarchical layer currently consumes `(theta_hat_s, V_s)` subject-level sufficient-statistic-like outputs. It is not misrepresented as a raw-observation hierarchical Bayesian model. A future raw-data joint JAX/NumPyro backend can replace this boundary while retaining the same population semantics.

The UKF is an approximation, not an exact posterior filter. It must be benchmarked against synthetic ground truth and, when needed, particle/SMC or smoother references.

## Tests

Batch 08 extends the existing regression suite and verifies:

- UKF nonlinear filtering with irregular sampling and missingness;
- EKF/UKF agreement in a near-linear regime;
- partial-pooling shrinkage;
- leave-one-cohort-out separation;
- predictive scoring and coverage.

## Batch benchmark

The included synthetic benchmark records EKF versus UKF latent RMSE and likelihood, UKF 95% state coverage, and partial-pooling convergence. In the seeded benchmark used for this batch, UKF was not better than EKF on latent RMSE; this is retained as evidence rather than hidden. The correct architectural conclusion is that UKF is an additional estimator selected by a validation gate, not an automatic replacement for EKF.
