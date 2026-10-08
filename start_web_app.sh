#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
if [ ! -d "$ROOT/frontend-app/node_modules" ]; then
  echo "Frontend dependencies are not installed. Run:"
  echo "  cd '$ROOT/frontend-app' && npm install"
  echo "Then start the Vite frontend with: npm run dev"
  echo "The Python runtime can be started independently with:"
  echo "  cd '$ROOT/backend' && PYTHONPATH=. python -m neuro_twin.api_server --host 127.0.0.1 --port 8000"
  exit 2
fi
(cd "$ROOT/backend" && PYTHONPATH=. python -m neuro_twin.api_server --host 127.0.0.1 --port 8000) &
BACKEND_PID=$!
trap 'kill $BACKEND_PID 2>/dev/null || true' EXIT
(cd "$ROOT/frontend-app" && npm run dev)
