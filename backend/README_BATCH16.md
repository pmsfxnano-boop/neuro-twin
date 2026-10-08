# NEURO-TWIN — Batch 16

## Formal PET kinetic modeling: plasma input, graphical analysis, and reference tissue

### Objective

Convert the PET layer from static quantitative measures (SUV/SUVR) into an explicit kinetic-modeling layer without silently choosing tracer-specific assumptions.

### Implemented

- `ArterialInputFunction` with explicit total plasma, measured parent fraction, fixed delay, non-negative validation, and PCHIP interpolation.
- Metabolite-corrected parent plasma input construction: `C_parent(t) = C_total_plasma(t) * f_parent(t)`.
- Reversible plasma-input 1-tissue compartment model (1TCM).
- Reversible plasma-input 2-tissue compartment model (2TCM).
- Constrained nonlinear least-squares fitting with positivity bounds.
- Optional fixed vascular blood-volume fraction; it is not estimated automatically because that can materially increase identifiability burden.
- Analytical distribution-volume reporting:
  - 1TCM: `VT = K1/k2`
  - 2TCM: `VT = (K1/k2) * (1 + k3/k4)`
- Plasma-input Logan graphical analysis with explicit caller-supplied `t*`.
- Reference-region Logan graphical analysis returning DVR with explicit caller-supplied `t*`.
- Standard 3-parameter SRTM (`R1`, `k2`, `BP_ND`) with explicit operational equation and nonlinear fitting.
- Parameter covariance when the Jacobian is full rank and enough observations are available.
- Jacobian rank/condition number diagnostics.
- AIC/AICc under a stated Gaussian observation-error assumption; no automatic model selection across non-equivalent estimands.
- Graphical-analysis stability utility across multiple candidate `t*` values; no silent `t*` optimization.

### Scientific boundaries

Not automated in Batch 16:

- arterial blood sampling;
- arterial-to-plasma metabolite correction estimation;
- delay estimation;
- dispersion correction;
- image-derived input function extraction;
- tracer-specific reference-region selection;
- partial-volume correction;
- irreversible Patlak modeling;
- full frame-integration likelihood; compartment fitting in this batch uses TAC sample times (typically frame midpoints) rather than integrating the predicted concentration over each PET frame.

These are intentionally separate because kinetic-model assumptions, tracer kinetics, acquisition protocol, and reference-tissue validity materially affect quantitative bias and identifiability.

### Why this boundary is necessary

The PET quantitative literature emphasizes that simplified models are assumption-dependent and can be biased by noise or kinetic mismatch. Logan analysis can exhibit noise-induced bias, while reference-tissue methods depend on a valid reference region and appropriate model assumptions. See the quantitative PET roadmap and reference-tissue literature: https://pmc.ncbi.nlm.nih.gov/articles/PMC9358699/ and https://pmc.ncbi.nlm.nih.gov/articles/PMC3677108/.

The implemented Logan equations follow the standard plasma-input and reference-input graphical formulations. The SRTM implementation follows the common three-parameter operational equation described in the reference-tissue literature.

### Validation

Deterministic synthetic tests cover:

- 1TCM parameter recovery;
- 2TCM forward-model recovery;
- plasma Logan sanity check;
- SRTM parameter recovery;
- reference Logan sanity check;
- parameter-rank/covariance behavior through the fitting API.

### Important interpretation rule

Kinetic endpoints are observations/features, not direct assignments to P/I/N/Q. A model configuration must explicitly declare how a kinetic endpoint enters `h(x, theta)`.
