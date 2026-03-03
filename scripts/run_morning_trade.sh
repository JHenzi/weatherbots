#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p Data/logs

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

WT_ENV="${WT_ENV:-demo}"
WT_SEND_ORDERS="${WT_SEND_ORDERS:-false}"
WT_MORNING_BUDGET="${WT_MORNING_BUDGET:-10}"
WT_BANDIT_MODE="${WT_BANDIT_MODE:-live}"

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

echo "[morning_trade] $(date -Is) trade_date=${TRADE_DATE}"

# Refresh predictions (7AM pulse already ran; this ensures predictions_latest.csv is current).
python intraday_pulse.py \
  --trade-date "$TRADE_DATE" \
  --env "$WT_ENV" \
  --decision-role monitoring \
  --bandit-mode "$WT_BANDIT_MODE" \
  --write-predictions

# Snapshot all market prices (start building the history for learned exit targets).
python log_market_prices.py --trade-date "$TRADE_DATE" --env "$WT_ENV" 2>&1 || true

# Morning Kelly scan and entry.
ARGS=(
  python morning_trader.py
  --env "$WT_ENV"
  --trade-date "$TRADE_DATE"
  --morning-budget "$WT_MORNING_BUDGET"
  --max-entry-price 25
  --kelly-fraction 0.25
  --min-kelly 0.05
  --max-buckets-per-city 2
)

case "$(echo "$WT_SEND_ORDERS" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|y) ARGS+=(--send-orders) ;;
esac

"${ARGS[@]}"
