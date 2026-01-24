#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p Data/logs

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

# Settle yesterday's event date (CLI posts next day).
TRADE_DATE="$(python - <<'PY'
import datetime as dt
print((dt.date.today() - dt.timedelta(days=1)).isoformat())
PY
)"

python settle_eval.py --trade-date "$TRADE_DATE"
python daily_metrics.py --as-of-date "$TRADE_DATE"
python update_city_metadata.py --as-of-date "$TRADE_DATE"

