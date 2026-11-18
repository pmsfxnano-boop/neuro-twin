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
- `backend/tests/test_six_domain_network.py`
- `backend/tests/test_six_domain_network_diagnostics.py`
- `backend/tests/test_six_domain_evidence_registry.py`
- `backend/tests/test_six_domain_domain_ontology.py`

The package is additive and does not yet connect to the production API or UI.

## Selected evidence sources

- Bice et al. (2026), *Systems biology of Alzheimer's Disease: a scoping review of key pathways and mechanisms*, DOI [10.1186/s13024-026-00934-4](https://doi.org/10.1186/s13024-026-00934-4).
- Zhao et al. (2026), *Noninvasive diagnosis of Alzheimer's disease: a review of emerging biomarkers and multimodal approaches*, DOI [10.3389/fcell.2026.1964029](https://doi.org/10.3389/fcell.2026.1964029).
- Brooker et al. (2024), *Cell biology of Parkinson's disease: Mechanisms of synaptic, lysosomal, and mitochondrial dysfunction*, DOI [10.1016/j.conb.2024.102841](https://doi.org/10.1016/j.conb.2024.102841).
- Nixon and Rubinsztein (2024), *Mechanisms of autophagy-lysosome dysfunction in neurodegenerative diseases*, DOI [10.1038/s41580-024-00757-5](https://doi.org/10.1038/s41580-024-00757-5).

These reviews support the construct inventory and candidate mechanism register; they do not validate the Neuro-Twin state variables, parameters, or predictions.
