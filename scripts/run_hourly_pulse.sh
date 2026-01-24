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
try:
    from zoneinfo import ZoneInfo
    tz = ZoneInfo((__import__("os").environ.get("TZ") or "America/New_York").strip() or "America/New_York")
except Exception:
    tz = dt.datetime.now().astimezone().tzinfo
today = dt.datetime.now(tz=tz).date()
print((today + dt.timedelta(days=1)).isoformat())
PY
)"

echo "[hourly_pulse] $(date -Is) trade_date=${TRADE_DATE}"
python hourly_pulse.py --trade-date "$TRADE_DATE"
