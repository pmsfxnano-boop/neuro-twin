# NEURO-TWIN — Batch 17

## Frame-integrated dynamic PET likelihood and kinetic design control

### Scientific objective

Replace the midpoint observation approximation with an explicit frame-average observation model for dynamic PET:

\[
 y_i = \frac{1}{\Delta_i}\int_{t_i}^{t_i+\Delta_i} C_{PET}(t)\,dt + \varepsilon_i.
\]

The production forward model is deterministic and exact conditional on a **piecewise-linear parent plasma input**. Segment boundaries include PET frame boundaries and shifted AIF knots. The implementation propagates both the latent compartment state and the frame integral through an augmented matrix exponential.

This is deliberately separate from the existing Batch 16 graphical methods and midpoint-based legacy compartment fitting.

### Implemented

- `PETFrameSchedule` with strict non-overlap and positive-duration validation.
- Explicit BIDS-seconds → kinetic-time-unit conversion (`frame_schedule_from_bids`).
- Exact frame-average forward model for reversible 1TCM.
- Exact frame-average forward model for reversible 2TCM.
- Optional fixed vascular fraction inside the frame-integrated observation.
- Constrained least-squares fitting against **frame averages**, not midpoint samples.
- Weighted likelihood using per-frame observation SD when supplied.
- Jacobian rank and condition diagnostics.
- AIC/AICc comparison only when the observation/error model is held comparable.
- Quantification of the discrepancy induced by replacing a frame average with its midpoint value.

### Mathematical construction

For a linear compartment model

\[
\dot x = Ax + B u(t),
\]

and piecewise-linear input

\[
u(t)=a+b\tau,
\]

the implementation augments the state with `u(t)`, a constant-one state, and an integral accumulator:

\[
\frac{d}{dt}\begin{bmatrix}x\\u\\1\\I\end{bmatrix}
=
\begin{bmatrix}
A & B & 0 & 0\\
0 & 0 & b & 0\\
0 & 0 & 0 & 0\\
C & D & 0 & 0
\end{bmatrix}
\begin{bmatrix}x\\u\\1\\I\end{bmatrix}.
\]

At the end of each frame, `I / Delta` is the model prediction for the reconstructed frame activity.

### Why the midpoint distinction matters

Dynamic PET measures are inherently associated with a finite acquisition interval. Literature has shown that treating a frame average as an instantaneous midpoint value can introduce non-negligible kinetic bias, with the magnitude dependent on tracer, framing and endpoint. See:

- https://pmc.ncbi.nlm.nih.gov/articles/PMC3049528/
- https://pmc.ncbi.nlm.nih.gov/articles/PMC4232966/

The implementation therefore makes the observation model explicit rather than hiding the approximation.

### BIDS timing contract

BIDS PET specifies `FrameTimesStart` and `FrameDuration` as arrays in seconds. The adapter converts them explicitly to the requested kinetic time unit so rate constants cannot silently be combined with seconds-based acquisition timing.

Reference: https://bids-specification.readthedocs.io/en/v1.11.2/modality-specific-files/positron-emission-tomography.html

### Validation

Batch 17 tests cover:

- frame schedule validation;
- BIDS time-unit conversion;
- 1TCM parameter recovery;
- 2TCM parameter recovery;
- agreement against a high-accuracy ODE reference under the same piecewise-linear input assumption;
- measurable midpoint-vs-frame-average discrepancy;
- AICc model ranking under a common observation model.

Batch 17: **6/6 PASS**.

The broader repository contains 96 collected tests. They were executed as individual test-file runs during this batch; the monolithic `pytest` invocation is not used as the acceptance criterion because it can exceed the execution-time budget even when the affected test files pass independently.

### Scientific boundaries

Not silently automated here:

- tracer-specific reference-region choice;
- arterial delay estimation;
- dispersion correction;
- plasma metabolite model fitting;
- frame-level count-noise likelihood derived from scanner physics;
- PET reconstruction correction factors;
- automatic selection between kinetic models based on AICc alone.

The frame-integrated engine is an observation-model improvement, not a claim of clinical validity.
