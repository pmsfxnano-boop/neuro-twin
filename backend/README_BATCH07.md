# NEURO-TWIN Batch 07 — Nonlinear observation + persistent hierarchical effects

Batch 07 adds a configurable nonlinear observation operator and a reference EKF
without changing the canonical P-I-N-Q dynamics. Observation channels may use
identity, log, exp, softplus, or sigmoid links, with optional explicit theta
loadings. Persistent Gaussian random effects (e.g. center/assay/scanner) are
represented as static augmented states with prior covariance. This preserves
longitudinal correlation instead of incorrectly adding independent per-visit
noise.

The JAX backend autodifferentiates the nonlinear observation and the exact
continuous-time affine state transition. It is validated against the SciPy EKF
reference on synthetic data with irregular sampling and missing channels.

Important: EKF is an approximation for nonlinear h. It is not promoted to a
scientific truth claim. Nonlinear operators must be registered/configured and
validated against synthetic recovery or a higher-order inference method.
