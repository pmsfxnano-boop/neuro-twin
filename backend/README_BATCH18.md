# NEURO-TWIN — Batch 18

## AIF uncertainty propagation for dynamic PET

### Scientific objective

Batch 18 adds an explicit nuisance-distribution layer around the deterministic
frame-integrated PET model from Batch 17.  The objective is to quantify how
uncertainty in total plasma activity, plasma parent fraction and arterial delay
propagates into kinetic parameters, without silently treating the measured AIF
as exact.

### Model

For an AIF represented by sampled total plasma `Cp_total(t)` and parent fraction
`f_parent(t)`, the parent input is

\[
C_p(t)=C_{p,total}(t)f_{parent}(t).
\]

The nuisance model uses:

- multiplicative lognormal uncertainty for positive total plasma activity;
- logistic-normal uncertainty for parent fraction, preserving `0 < f < 1`;
- exponential temporal correlation `rho(dt)=exp(-|dt|/ell)`;
- non-negative truncated-normal uncertainty for a fixed arterial delay.

The tissue model itself remains the exact frame-integrated Batch 17 forward model.

### Outputs

`AIFPropagationResult` stores:

- successful parameter draws;
- propagated parameter mean/SD/covariance;
- AIF and fit success rates;
- delay distribution diagnostics;
- explicit assumptions.

Two modes are distinguished:

1. **AIF-only propagation**: the observed tissue TAC is held fixed, isolating AIF-induced uncertainty.
2. **Joint parametric Monte Carlo**: AIF uncertainty and frame observation noise are both resampled when an observation SD vector is supplied.

### Scientific scope and limitations

This is a parametric uncertainty propagation engine, not an assertion that the selected nuisance distributions are universally correct.  The priors/uncertainty model must be calibrated from the actual blood-sampling protocol or validated derivative outputs.

BIDS PET derivatives explicitly support blood-processing outputs containing interpolated whole-blood, plasma and metabolite-corrected AIF information, together with configuration describing the parent-fraction and AIF modeling procedure.  The current implementation is designed to consume such products upstream, while keeping their uncertainty model explicit rather than treating the reported AIF as noise-free.

Literature has shown that AIF uncertainty can materially propagate into PET kinetic parameter estimates, and that noisy/sparse parent-fraction measurements can compromise reconstructed AIFs. See:

- https://pubmed.ncbi.nlm.nih.gov/21127711/
- https://pubmed.ncbi.nlm.nih.gov/23108277/
- https://pubmed.ncbi.nlm.nih.gov/25873424/

### Validation

Batch 18 tests cover:

- reproducible AIF ensemble generation;
- physical support constraints for total plasma, parent fraction and delay;
- zero-uncertainty recovery;
- non-zero propagated kinetic uncertainty;
- joint-noise contract requiring tissue observation SD.

The affected test file is executed independently as part of the batch acceptance process.
