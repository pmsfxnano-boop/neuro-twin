# NEURO-TWIN Runtime Production v3

This package contains the production runtime boundary only. It contains **no synthetic runtime payload** and the bridge rejects synthetic/reference publication.

## Start

```bash
cd backend
PYTHONPATH=. ../RUN_PRODUCTION.sh
```

Open `http://127.0.0.1:8000/`.

The UI initially shows `WAITING FOR LIVE RUNTIME` until the scientific pipeline publishes a validated result.

## Publish a real result

### In-process

```python
from neuro_twin.pipeline.runtime_publish import publish_runtime_result
publish_runtime_result("/path/to/backend", runtime_result)
```

### HTTP

```text
POST /v1/runtime/publish
```

### CLI

```text
PYTHONPATH=backend python -m neuro_twin.runtime.cli /path/to/runtime_result.json --root backend
```

## PET uncertainty → Neuro-Twin posterior

The runtime adapter requires a registered PET→Neuro observation model. It computes

`R_eff = R_kin + Htheta Sigma_theta Htheta^T`

and updates the state posterior through the registered `Hx` Jacobian. The adapter
never invents a mapping from PET kinetic parameters to P/I/N/Q.

## Runtime guarantees

- point-in-time chronology enforced;
- provenance/raw hash required;
- future information relative to the PIT cutoff rejected;
- temporal leakage in OOS rejected;
- synthetic/reference runtime publication rejected;
- content-addressed runtime result written to `backend/runtime/`;
- REST + WebSocket publishing available;
- 3D UI reads only live runtime state.
