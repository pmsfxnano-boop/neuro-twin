# NEURO-TWIN — cumulative research engine, Batch 19

## Batch 19: joint AIF/kinetic inference + latent delay/dispersion

Batch 19 adds a joint local Bayesian layer around the exact frame-integrated PET
forward model. Kinetic parameters and selected AIF nuisance parameters are now
fit together under an explicit likelihood and explicit priors, with MAP + local
Laplace uncertainty.

### Core additions

- `neuro_twin/pet/joint_inference.py`
  - causal exponential AIF dispersion state;
  - latent arterial delay;
  - optional plasma scale nuisance;
  - optional parent-fraction calibration nuisance;
  - joint MAP estimation;
  - Laplace covariance / local credible intervals;
  - posterior predictive sampling from the local Laplace approximation;
  - local parameter-correlation diagnostics.
- `neuro_twin/pet/blood.py`
  - current BIDS raw blood-recording contract via `read_bids_blood_tsv`;
  - compatibility retained for the previous Batch 18 `read_bloodproc_tsv` adapter.

### Critical scientific finding

The synthetic benchmark exposed near-perfect local anti-correlation between
`K1` and an uncertain common plasma scale (`r ≈ -0.99984`). The engine records
this as an identifiability warning rather than presenting the parameters as
independently measured. Releasing that nuisance increased local `K1` uncertainty
by approximately 56x in the benchmark.

### Cumulative architecture

```text
DATA FABRIC → PIT / PROVENANCE → BIDS / CANONICAL
→ MRI / PET FEATURES → PET FRAME-INTEGRATED MODEL
→ AIF / BLOOD CONTRACT → AIF UNCERTAINTY
→ JOINT AIF + KINETIC INFERENCE → IDENTIFIABILITY
→ STATE / PARAMETER INFERENCE → HIERARCHICAL BAYES
→ UNCERTAINTY → OOS / EXTERNAL VALIDATION → EVIDENCE STORE
```

### Normative BIDS blood recording alignment

Current official BIDS PET documentation defines raw blood recordings using a
`*_blood.tsv` table and a corresponding JSON sidecar; the `recording-<label>`
entity distinguishes acquisition methods. Relevant fields include `time`,
`plasma_radioactivity`, `metabolite_parent_fraction`, and required availability
flags such as `PlasmaAvail`, `MetaboliteAvail`, `WholeBloodAvail`, and
`DispersionCorrected`. citeturn744008search2turn744008search1

This cumulative package preserves the previous Batch 18 compatibility adapter
but uses `read_bids_blood_tsv` for the normative current contract.

This remains subordinate to the mathematical and pipeline contract defined in
`NEURO_TWIN_Nucleo_Motor_Pipeline_Profesional.pdf`.
