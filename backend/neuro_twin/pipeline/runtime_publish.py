"""One-call publication hook for the scientific pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from neuro_twin.runtime.adapter import RuntimeEngineAdapter
from neuro_twin.runtime.contracts import RuntimeResult


def publish_runtime_result(runtime_root: str | Path, result: RuntimeResult) -> dict[str, Any]:
    """Publish a validated, real pipeline result to the research bridge outbox."""
    return RuntimeEngineAdapter(runtime_root).publish(result)
