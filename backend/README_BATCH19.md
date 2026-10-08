# NEURO-TWIN — Batch 19

## Joint AIF + kinetic inference with latent delay and dispersion

### Scientific objective

Batch 19 upgrades the Batch 18 Monte-Carlo nuisance propagation into an explicit
joint local Bayesian model. Kinetic parameters and selected low-dimensional AIF
nuisance parameters are estimated together from the same frame-integrated PET
likelihood instead of being fit in two independent stages.

The model adds a causal first-order dispersion filter:

\[
\frac{dD}{dt}=\frac{C_{p,parent}(t-\delta)-D(t)}{\tau},
\]

where `delta` is an explicit latent arterial delay and `tau` is the dispersion
time constant. The tissue compartment then receives `D(t)` as its input.

For a 1TCM:

\[
\dot C_t = K_1 D-k_2 C_t.
\]

For a 2TCM:

\[
\dot C_1=K_1D-(k_2+k_3)C_1+k_4C_2,\qquad
\dot C_2=k_3C_1-k_4C_2.
\]

The PET measurement remains the exact frame average over each `[start, end]`
interval, using a piecewise-linear parent-plasma source and augmented matrix
exponentials. Therefore the new nuisance layer does not revert to a midpoint
approximation.

### Joint nuisance model

Optional nuisance parameters are:

- `plasma_log_scale`: common multiplicative plasma correction;
- `parent_fraction_logit_shift`: common parent-fraction calibration shift;
- `delay`: non-negative latent arterial delay;
- `dispersion_tau`: positive exponential dispersion time constant.

Priors are explicit:

- Gaussian on `plasma_log_scale`;
- Gaussian on `parent_fraction_logit_shift`;
- truncated-support Gaussian on delay;
- lognormal prior on dispersion time constant.

The current implementation is **MAP + local Laplace**. It is not claimed to be
full NUTS/MCMC. A rejection-filtered multivariate-normal ensemble from the
Laplace approximation is included only for local posterior predictive
Diagnostics.

### Identifiability result

The synthetic benchmark deliberately includes the optional plasma-scale
nuisance. The local correlation between `K1` and `plasma_log_scale` was:

`-0.9998425996565353`

and the condition number increased from approximately `651` without the
plasma-scale nuisance to approximately `5748` when that nuisance was enabled.

This is not a failure of the software. It exposes a real structural
confounding: a common multiplicative change in the plasma input can be traded
against `K1`. The system therefore reports the correlation and conditioning
instead of treating the fitted parameters as independently identified.

The local posterior SD of `K1` increased by a factor of approximately `56.36`
in the synthetic comparison when the uncertain plasma scale was released under
its stated prior. This is benchmark evidence about the specified model, not a
clinical claim.

### Current BIDS blood contract correction

During Batch 19 review, the current official BIDS PET specification was checked.
The normative raw blood-recording form is `*_blood.tsv` with a corresponding
JSON sidecar and the `recording-<label>` entity. The table includes `time` and,
when available, `plasma_radioactivity`, `metabolite_parent_fraction`, and other
blood measurements. The sidecar includes required availability flags such as
`PlasmaAvail`, `MetaboliteAvail`, `WholeBloodAvail`, and `DispersionCorrected`.
The adapter `read_bids_blood_tsv` now validates this contract.

The earlier `read_bloodproc_tsv` adapter remains for compatibility with the
previous Batch 18 derivative/legacy contract; it is no longer presented as the
normative raw BIDS naming scheme.

This distinction follows the current BIDS specification. citeturn744008search2turn744008search1

### Verification

Targeted scientific regression suite:

- Batch 16 PET kinetics: 5/5 PASS
- Batch 17 frame models: 6/6 PASS
- Batch 18 AIF uncertainty: 5/5 PASS
- Batch 19 joint inference + normative BIDS blood contract: 5/5 PASS
- compileall: PASS

Total pytest tests currently collected in the cumulative repository: `106`.
The complete monolithic suite is not used as the sole acceptance criterion when
execution-time limits can truncate it; affected scientific layers are executed
independently as above.

### Benchmark

Ground truth:

```text
K1              = 0.12
k2              = 0.20
arterial delay  = 0.75
spread tau      = 1.35
```

Joint MAP:

```text
K1              = 0.1199999976
k2              = 0.1999999977
delay           = 0.7500001676
dispersion_tau  = 1.3499998824
```

Local Jacobian rank: `4/4`.
Condition number: approximately `651.24`.
Laplace ensemble accepted: `2000/2000` in the benchmark.

A posterior-predictive smoke test with measurement noise achieved 100% point
coverage for the synthetic truth across the tested frames, with mean predictive
RMSE approximately `0.00190`. This is a software/benchmark diagnostic, not a
calibration study.

### Scope boundary

Batch 19 does not yet claim:

- tracer-specific AIF priors;
- validated population priors;
- automated arterial delay/dispersion selection;
- full NUTS/SVI execution in this environment;
- clinical calibration or prospective validity;
- automatic propagation all the way from PET nuisance parameters into the
  longitudinal P-I-N-Q posterior.

The latter is the next integration boundary: PET kinetic posterior → canonical
observation distribution → Neuro-Twin state-space inference.
