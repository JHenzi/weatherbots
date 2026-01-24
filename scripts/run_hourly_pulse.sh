#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p Data/logs

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

# Pulse is for the same trade_date that the trade job uses (tomorrow's markets).
TRADE_DATE="$(python - <<'PY'
import datetime as dt
print((dt.date.today() + dt.timedelta(days=1)).isoformat())
PY
)"

python hourly_pulse.py --trade-date "$TRADE_DATE"

#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p Data/logs

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

python hourly_pulse.py

