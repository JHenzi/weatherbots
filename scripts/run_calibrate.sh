#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p Data/logs

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

WINDOW_DAYS="${WT_WINDOW_DAYS:-14}"

TRADE_DATE="$(python - <<'PY'
import datetime as dt
print((dt.date.today() - dt.timedelta(days=1)).isoformat())
PY
)"

python calibrate_sources.py --trade-date "$TRADE_DATE" --window-days "$WINDOW_DAYS"

