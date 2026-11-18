# NEURO-TWIN — Six-domain dynamic network (research architecture v1.0)

## Status

This is an additive research architecture. It does not replace the deployed or
existing P-I-N-Q implementation, disease modules, API contracts, or runtime.
It is not a clinically validated model and is not a diagnostic system.

The current software baseline remains four-dimensional P-I-N-Q. This package
introduces a separate six-dimensional mathematical interface so that new
work can be tested without silently changing the established runtime.

## State vector

The six latent domains are ordered as:

```
x(t) = [P(t), I(t), M(t), N(t), Q(t), Pr(t)]^T
```

| State | Working construct | Working sign convention |
|---|---|---|
| P | Neurodegenerative progression | Higher = greater progression burden |
| I | Inflammation | Higher = greater inflammatory burden |
| M | Metabolism | Higher = greater metabolic dysregulation |
| N | Neuronal integrity | Higher = greater preserved integrity |
| Q | Functional capacity | Higher = greater functional capacity |
| Pr | Proteostasis | Higher = greater proteostasis impairment |

These sign conventions are latent-coordinate conventions. They are not
biomarker definitions, clinical cut-offs, nor validated measurements. They
must be reviewed before any disease-specific observation binding is registered.

Each coordinate is latent. It is not equivalent to a directly measured
concentration, imaging intensity, cognitive score, or clinical diagnosis.

## Hierarchical domain ontology

The top-level state vector is coarse-grained. A separate ontology in
`backend/neuro_twin/network/domain_ontology_v1.json` decomposes the domains
into candidate subconstructs without expanding them into extra fitted states
or assigning automatic composite scores.

Key distinctions are enforced:

- `M` contains separate candidate constructs for mitochondrial bioenergetics,
  glucose/lactate metabolism, lipid metabolism, and redox homeostasis.
- `Pr` contains folding/chaperone systems, ubiquitin-proteasome function,
  autophagy-lysosome clearance, and aggregation/trafficking.
- A tau, amyloid-beta or alpha-synuclein pathology assay does not directly
  measure total proteostasis capacity or flux.
- `N` (neuronal/circuit integrity) is distinct from `Q` (realized functional
  capacity); reserve, compensation, treatment and comorbidity may separate them.
- `P` is a latent progression-burden state, not its derivative. Its
  instantaneous velocity is `dP/dt`; it is not a diagnosis, clinical stage,
  nor a composite sum of the other five domains.

The ontology explicitly leaves parent-domain aggregation
`not_operationalized`. Equal weighting, averaging or collapsing subconstructs
into a single score is not allowed until a measurement and aggregation model
has been specified and validated.

## Dynamic model

The v1.0 reference model is a transparent affine continuous-time network:

```
dx/dt = b + A x
```

- `b[i]` is the drift of state `i`, in latent-coordinate units per time unit.
- `A[i,j]` is the directed influence of source state `j` on the derivative
  of target state `i`; diagonal terms are self-dynamics.
- The permitted graph is an explicit list of directed candidate edges
  `(source, target)`. Non-zero matrix entries outside that list are rejected.
  An edge in this list is a candidate parameter, not evidence of causality.
- Time units are mandatory metadata. Coefficients in `A` therefore have
  inverse-time units.
- Initial states, drift, candidate edges, and coupling coefficients must be
  explicit inputs. No empirical coefficient values or default causal edges
  are supplied.
- The current parameter vector has six drift terms plus one coefficient per
  declared candidate edge. This is a structural parameterization contract,
  not a recommendation to fit every parameter at once. Sparse graph design,
  regularization, and identifiability gates must limit the fitted dimension.

For any interval `dt`, the affine transition is evaluated with a matrix
exponential:

```
x(t + dt) = Phi(dt) x(t) + offset(dt)
```

This is exact for the affine v1.0 equations up to numerical precision and
supports irregular longitudinal visit times.

A richer nonlinear model may ultimately be necessary. Nonlinear terms are
not enabled by default: their form should be justified by biological
mechanisms, sufficient observations, and identifiability analysis rather
than selected for mathematical sophistication alone.

## Observation model

Observed data are connected only through a versioned, explicit operator:

```
y(t) = H x(t) + bias + epsilon(t)
epsilon(t) ~ N(0, R)
```

- `H` must be supplied explicitly and has six columns.
- `R` must be positive definite.
- Feature names and modality labels do not automatically create biological
  mappings.
- MRI, PET, fluid biomarkers, EEG, cognition, and clinical scales retain
  their own measurement model, units, uncertainty, provenance, and acquisition
  timing outside this latent-coordinate contract.

This preserves the repository invariant that the scientific engine never
guesses a biomarker-to-state mapping.

## Longitudinal inference

The package includes:

- exact irregular-interval state transitions;
- continuous-time process-noise discretization using the Van Loan method;
- a six-dimensional continuous-discrete Kalman filter;
- missing-channel handling via NaN observations;
- Joseph-form covariance updates;
- a retrospective Rauch–Tung–Striebel smoother.

The forward filter uses only current and prior observations and is the
appropriate foundation for prospective point-in-time (PIT) inference. The RTS
smoother uses later observations and is retrospective only; it must not be
used to publish a prospective state estimate.

## Disease separation

Alzheimer and Parkinson are explicit, separate model scopes. The architecture
can share numerical code, contracts, and validation machinery, but each
disease requires its own candidate graph, parameter vector, observation
operators, priors, and validation. The package rejects other disease labels
rather than silently widening the agreed scope.

## Parameter status and evidence

A model defaults to `parameter_status = hypothesis`. Its SHA-256 configuration
hash covers disease scope, time unit, version, drift, coupling matrix,
candidate graph, and parameter status. A matrix that can be simulated is not
thereby estimated, identifiable, causal, or clinically validated. Promotion to estimated or validated status must be performed by
an audited calibration/validation workflow and must preserve dataset versions,
PIT cut-offs, preprocessing versions, hashes, uncertainty, and evaluation
results.

The architecture itself does not claim that all six states are identifiable
from currently available datasets. Observability and identifiability are
empirical gates to be evaluated per disease, cohort, modality combination,
and observation schedule.

## Observation and measurement engine

The initial source-measurement inventory is stored in
`backend/neuro_twin/network/measurement_catalog_v1.json` and validated by
`network/measurements.py`. It is separate from the latent observation
operator `H` and from the existing four-state production observation
registry.

Each definition records disease scope, canonical feature, modality,
observation kind, specimen, analyte or metric, value type, canonical unit,
permitted source units, value direction, target subconstruct, relation type,
uncertainty policy, assay requirements, required acquisition/processing
metadata, provenance requirements, limitations, and references.

The v1.0 inventory contains 11 initial measurement definitions:

- **Alzheimer disease:** plasma p-tau217, CSF Aβ42/40, plasma GFAP, plasma
  NfL, regional FDG-PET SUVR, amyloid-PET Centiloid, MRI hippocampal volume,
  and ADAS-Cog total score.
- **Parkinson disease:** CSF alpha-synuclein seed-amplification result,
  MDS-UPDRS Part III, and plasma NfL.

This list is an initial measurement contract, not a claim that every variable
is available in every ADNI or PPMI release. Source-specific variable names,
assay platforms, visit schedules and access conditions must be confirmed
against the currently downloaded source data dictionary and snapshot.

A validated measurement envelope carries the existing canonical
`Observation` plus typed uncertainty semantics (standard deviation,
standard error, coefficient of variation or confidence-interval half-width),
catalog version, and optional harmonization identity. A numeric uncertainty
without its statistical meaning is not accepted as a typed uncertainty.
Accepted non-canonical units and source feature aliases require an explicit,
versioned harmonization step; the validator does not relabel or rescale values.

A measurement can pass the measurement-layer checks while remaining
**ineligible for latent inference**. In v1.0 every measurement definition has
`latent_binding_status = not_registered`; no feature-to-`H` weights are
created. A subconstruct proxy is not a direct measurement of P, I, M, N, Q
or Pr.

Measurement validation checks disease scope, canonical feature, modality,
observation kind, canonical unit, assay identity where required, time-zone
aware timestamps, provenance source version and raw hash, preanalytical and
processing metadata, quality flags, typed uncertainty, and scalar/binary
value constraints. Scanner and center effects remain explicit warnings for
downstream harmonization and sensitivity analysis; they are not silently
corrected.

The coverage report summarizes which subconstructs have at least one
measurement proxy, which measurement types are blocked, and which are absent
from a dataset. It deliberately labels this as **measurement-proxy coverage**.
It does not call a domain observed, compute latent observability, infer missing
measurements, or authorize state estimation. Six-domain latent observability
continues to be evaluated only through an explicit versioned latent operator
and the separate network diagnostics.

The catalog is validated against the hierarchical ontology. In particular,
regional MRI morphometry has a separate N subconstruct
`regional_structure`; it is not treated as a neuronal cell count.

### Source governance

PPMI provides data dictionaries, code lists, user guidance and biomarker
resources. Its individual-level data access process requires registration and
a Data Use Agreement; its biospecimens follow a separate request and
verification process. The measurement catalog therefore does not assume
that a public website listing implies unrestricted access to participant-level
files or physical specimens.

## Literature interaction register

`backend/neuro_twin/network/edge_evidence_v1.json` contains a versioned,
disease-separated inventory of literature-level interactions. Its schema is
validated by `network/evidence.py`. The initial records cover provisional
metabolism–proteostasis interactions in Alzheimer disease and Parkinson
disease, an unresolved inflammation–proteostasis interaction in Alzheimer
disease, and selected Parkinson candidate mechanisms involving
proteostasis–inflammation, inflammation–neuronal integrity, and
metabolism–neuronal integrity.

The initial register is based on review syntheses, including the 2025
Alzheimer molecular-neurobiology review (Hein et al., DOI
10.1007/s12035-025-05602-0), a 2026 Alzheimer systems-biology scoping review
(Bice et al., DOI 10.1186/s13024-026-00934-4), the 2024 review of
mitochondrial/lysosomal/synaptic cross-talk in Parkinson disease (Brooker et
al., DOI 10.1016/j.conb.2024.102841), and a 2025 Parkinson signaling-pathways
review (Khan et al., DOI 10.3390/ijms26136416). The registry also references
the 2024 autophagy–lysosome review (DOI 10.1038/s41580-024-00757-5) and
other disease-specific reviews.

Every entry is explicitly marked `literature_candidate_only`, has no
coefficient sign constraint, and is not eligible for parameter fitting.
Undirected evidence is not converted into a directed edge. Candidate
mechanisms do not establish causality, patient-level temporal ordering, or
parameter values. The registry is a source-traceable hypothesis inventory,
not a validated biological graph.


## Cumulative persistent-damage sidecar

The Alzheimer systems decomposition adds a separate damage-accumulation model in
`backend/neuro_twin/network/damage.py`. It tracks four persistent-damage
compartments: synaptic/structural, axonal/neuritic, neuronal-somatic, and
neurovascular. Its rate is the positive excess of explicitly mapped damaging
exposures over explicitly mapped repair/resilience capacity, integrated across
irregular time points.

This module is intentionally not a seventh coordinate in the six-state
`P, I, M, N, Q, Pr` vector. It does not infer mappings from biomarkers or
from those six latent domains. Its input normalization, coefficients, initial
damage and time units must be supplied explicitly. It models persistent
damage as non-decreasing; reversible dysfunction and repairable synaptic
plasticity require separate states. The implementation is a research
hypothesis component, not a calibrated or clinically validated damage score.

The Alzheimer mechanism inventory is stored in
`backend/neuro_twin/network/alzheimer_mechanism_map_v1.json` and validated by
`network/mechanisms.py`. The v1 catalogue contains 28 multiscale nodes and
30 candidate edges with evidence maturity, causal-status labels, limitations
and references. It is not an executable or validated causal graph.

See `docs/ALZHEIMER_SYSTEMS_REVERSE_ENGINEERING_V1.md` for the full systems
decomposition and the research questions that should guide the next engine
increments.


## Mechanism-level dynamics, intervention and model comparison

The Alzheimer systems-research increment adds
`backend/neuro_twin/network/mechanistic_dynamics.py`. It is a nonlinear
mechanism-level simulator, separate from the six-domain affine baseline.
Each modeled mechanism node has an explicitly supplied normalized activity
set-point and relaxation rate. Each executable interaction has an explicitly
supplied coefficient, sign assumption and Hill saturation parameters:

\[
\dot z_i = \lambda_i(b_i-z_i)
+\sum_{e\in A_i}\beta_e h_e(z_{source})(1-z_i)
-\sum_{e\in I_i}\beta_e h_e(z_{source})z_i
\]

The state coordinate \(z_i\) is normalized activity in [0,1], not a raw
biomarker, clinical score or universal measure of pathogenicity. The equation
is a model family; it is not a claim that Alzheimer biology follows this exact
kinetic law.

Scientific gates:
- Every run must validate its architecture and edge selection against the
  machine-readable Alzheimer mechanism and origin registries.
- Only edges marked `directed_candidate` are executable. Bidirectional or
  unresolved relationships are blocked until a specific direction is
  justified as a new explicit hypothesis.
- Edge signs are not inferred from the prose, evidence grade or source paper.
  Activation/inhibition is an explicit context-specific modeling assumption.
- Every baseline, kinetic rate, coefficient, Hill parameter, initial activity
  and damage mapping is caller-supplied. No biological defaults are created.
- A model with `parameter_status=hypothesis` produces only a hypothesis
  scenario. Estimated/validated labels require a separate auditable workflow.

Virtual intervention protocols:
- `clamp` performs an explicit time-bounded do-like state clamp;
- `additive_drive` adds a signed rate input to a target mechanism;
- intervention start/end boundaries are inserted into the integration and
  damage-accumulation schedule;
- baseline and perturbed trajectories use the same initial state and
  parameter set;
- outputs include component-wise final damage prevented and the difference
  in integrated damage burden over time;
- multiple supplied parameter draws can be summarized with quantiles.

A virtual clamp is not a drug dose or pharmacokinetic model. The result is
labelled `model_conditional_counterfactual_not_empirical_causal_effect`.
It says what the supplied equations imply under a specified intervention,
not what would happen in a person. Parameter uncertainty summaries only
reflect the supplied parameter ensemble and omit unmodeled structural,
measurement and cohort uncertainty.

A separate versioned mechanism observation operator is implemented in
backend/neuro_twin/network/mechanism_observation.py. It requires an explicit
matrix H, bias, positive-definite observation covariance, mechanism-node
order, feature names/units, source-catalog hash and transformation identity.
It produces prediction means from mechanism trajectories and can summarize
prediction mean/dispersion across caller-supplied parameter draws. It does not
automatically bind p-tau, A-beta, GFAP, NfL, PET or MRI to mechanisms. A source
adapter must first identify the actual dataset feature, assay, transform,
units, timing and provenance; a domain expert must specify the observation
relationship and its uncertainty.

The separate `network/model_comparison.py` module scores predictive
distributions on a declared subject holdout, temporal holdout or external
cohort. It requires the same observed outcomes and availability mask for all
forecasts and reports Gaussian mean log predictive density, RMSE and 50%/90%
interval coverage. A superior held-out predictive score does not prove the
architecture's origin hypothesis is causal. The prediction arrays must be
produced by an explicit, versioned observation operator before this scoring
function is used on biological data.

This architecture is therefore a controlled executable hypothesis framework,
not yet a calibrated mechanistic model. It must be fit to authorized
longitudinal human data, checked for parameter identifiability, evaluated on
held-out cohorts, and triangulated against genetic and perturbation evidence
before any intervention ranking is scientifically interpretable.

## Validation boundary

The included tests are deterministic numerical/software tests using
mathematical fixtures. They do not establish neurobiological validity or
predictive performance.

Required next research gates:

1. Define disease-specific biological hypotheses and candidate directed edges.
2. Register modality-specific observation operators with traceable evidence.
3. Evaluate structural and practical identifiability before fitting all
   coupling parameters.
4. Use simulated parameter-recovery tests and posterior predictive checks.
5. Validate with subject-level splits, temporal PIT/walk-forward evaluation,
   and independent out-of-sample cohorts.
6. Calibrate uncertainty and report failure/abstention when observability is
   insufficient.
7. Keep clinical diagnosis and treatment decisions outside this research
   runtime.

## Code location

- `backend/neuro_twin/network/model.py`
- `backend/neuro_twin/network/observation.py`
- `backend/neuro_twin/network/state_space.py`
- `backend/neuro_twin/network/diagnostics.py`
- `backend/neuro_twin/network/evidence.py`
- `backend/neuro_twin/network/edge_evidence_v1.json`
- `backend/neuro_twin/network/ontology.py`
- `backend/neuro_twin/network/domain_ontology_v1.json`
- `backend/neuro_twin/network/measurements.py`
- `backend/neuro_twin/network/measurement_catalog_v1.json`
- `backend/tests/test_six_domain_network.py`
- `backend/tests/test_six_domain_network_diagnostics.py`
- `backend/tests/test_six_domain_evidence_registry.py`
- `backend/tests/test_six_domain_domain_ontology.py`
- `backend/tests/test_six_domain_measurements.py`

The package is additive and does not yet connect to the production API or UI.

## Selected evidence sources

- Bice et al. (2026), *Systems biology of Alzheimer's Disease: a scoping review of key pathways and mechanisms*, DOI [10.1186/s13024-026-00934-4](https://doi.org/10.1186/s13024-026-00934-4).
- Zhao et al. (2026), *Noninvasive diagnosis of Alzheimer's disease: a review of emerging biomarkers and multimodal approaches*, DOI [10.3389/fcell.2026.1964029](https://doi.org/10.3389/fcell.2026.1964029).
- Brooker et al. (2024), *Cell biology of Parkinson's disease: Mechanisms of synaptic, lysosomal, and mitochondrial dysfunction*, DOI [10.1016/j.conb.2024.102841](https://doi.org/10.1016/j.conb.2024.102841).
- Nixon and Rubinsztein (2024), *Mechanisms of autophagy-lysosome dysfunction in neurodegenerative diseases*, DOI [10.1038/s41580-024-00757-5](https://doi.org/10.1038/s41580-024-00757-5).

These reviews support the construct inventory and candidate mechanism register; they do not validate the Neuro-Twin state variables, parameters, or predictions.

## Measurement source documentation

- Alzheimer's Association Workgroup (2024), *Revised criteria for diagnosis and staging of Alzheimer's disease*, DOI [10.1002/alz.13859](https://doi.org/10.1002/alz.13859). This provides a framework for Core 1 amyloid and phosphorylated-tau biomarkers and later-changing tau biomarkers; fluid and PET staging must not be treated as interchangeable.
- Alzheimer's Association (2024), *DETeCD-ADRD clinical practice guideline*, DOI [10.1002/alz.14333](https://doi.org/10.1002/alz.14333). It discusses plasma/CSF p-tau species, GFAP, NfL, MRI and FDG-PET in distinct measurement categories.
- PPMI official [Guidance Resources](https://www.ppmi-info.org/access-data-specimens/guidance-resources) publish data dictionaries, code lists, data user guides, biomarker resources and SAA guidance.
- PPMI official [Access Data](https://bank.ppmi-info.org/access-data-specimens/download-data) documents individual-level data access, registration and Data Use Agreement requirements.
- Morris and Lees (2024), *Limitations of the α-Synuclein Seed Amplification Assay in Clinical Practice*, DOI [10.1001/jamaneurol.2024.2381](https://doi.org/10.1001/jamaneurol.2024.2381), cautions against overgeneralizing assay results across the pathological diversity of Parkinson syndromes.
- (2026), *Seed Amplification Assays for Parkinson's Disease: A Review of α-Synuclein Assays in Body Fluids and Tissues*, DOI [10.1111/jnc.70435](https://doi.org/10.1111/jnc.70435), notes continuing protocol standardization and multicenter validation needs.

These references support measurement categories and limitations. They do not validate the Neuro-Twin latent-state mappings or confer clinical validity.
