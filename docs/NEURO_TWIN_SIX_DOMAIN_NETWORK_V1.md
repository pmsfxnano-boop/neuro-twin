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
- `backend/tests/test_six_domain_network.py`

The package is additive and does not yet connect to the production API or UI.
