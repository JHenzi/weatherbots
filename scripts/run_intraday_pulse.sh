#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p Data/logs

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

# Pulse is for the same trade_date that the trade job uses (**today's** markets).
TRADE_DATE="$(python - <<'PY'
import datetime as dt, os
try:
    from zoneinfo import ZoneInfo
    tz = ZoneInfo((os.environ.get("TZ") or "America/New_York").strip() or "America/New_York")
except Exception:
    tz = dt.datetime.now().astimezone().tzinfo
today = dt.datetime.now(tz=tz).date()
print(today.isoformat())
PY
)"

WT_BANDIT_MODE="${WT_BANDIT_MODE:-live}"

echo "[intraday_pulse] $(date -Is) trade_date=${TRADE_DATE}"
python intraday_pulse.py --trade-date "$TRADE_DATE" --decision-role monitoring --bandit-mode "$WT_BANDIT_MODE" --write-predictions

# Snapshot all bucket prices for the market-making price history (graceful: skips if no API key).
python log_market_prices.py --trade-date "$TRADE_DATE" --env "${WT_ENV:-demo}" 2>&1 || true