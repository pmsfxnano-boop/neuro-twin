# Contributing to NEURO-TWIN

## Change policy

Scientific changes require tests that demonstrate numerical behavior, not only API availability.

For changes affecting:

- dynamics,
- observation models,
- PET kinetic models,
- uncertainty propagation,
- temporal validation,
- identifiability,
- runtime contracts,

add or update a scientific regression test before merging.

## Pull requests

Every PR should state:

1. scientific hypothesis or engineering problem;
2. affected modules;
3. validation performed;
4. changes to assumptions or contracts;
5. whether any PIT/OOS behavior changed.

Never commit subject-level data, credentials, API tokens or protected datasets.
