#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec python -m neuro_twin.api_server --host "${NEURO_TWIN_HOST:-127.0.0.1}" --port "${NEURO_TWIN_PORT:-8000}"
