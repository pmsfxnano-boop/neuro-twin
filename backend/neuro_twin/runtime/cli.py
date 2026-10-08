"""CLI to publish a real RuntimeResult JSON produced by the scientific pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapter import RuntimeEngineAdapter
from .contracts import RuntimeResult


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish a real NEURO-TWIN RuntimeResult")
    parser.add_argument("runtime_json", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    payload = json.loads(args.runtime_json.read_text(encoding="utf-8"))
    result = RuntimeResult.model_validate(payload)
    published = RuntimeEngineAdapter(args.root).publish(result)
    print(json.dumps({"status": published["publication_status"], "runtime_id": result.runtime_id, "result_hash": result.provenance["result_hash"]}, indent=2))


if __name__ == "__main__":
    main()
