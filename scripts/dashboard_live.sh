#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_FILTER="${1:-prod}"
INTERVAL="${2:-5}"
LIMIT="${3:-10}"
CITY="${4:-}"

PY_BIN="python3"
if [[ -x ".venv/bin/python" ]]; then
  PY_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PY_BIN="python"
fi

"$PY_BIN" scripts/dashboard_tui.py --env "$ENV_FILTER" --interval "$INTERVAL" --limit "$LIMIT" --city "$CITY"

