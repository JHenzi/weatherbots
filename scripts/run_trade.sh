#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p Data/logs

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

WT_DAILY_BUDGET="${WT_DAILY_BUDGET:-50}"
WT_ENV="${WT_ENV:-demo}"
WT_SEND_ORDERS="${WT_SEND_ORDERS:-false}"
WT_MAX_CONTRACTS_PER_ORDER="${WT_MAX_CONTRACTS_PER_ORDER:-500}"

TRADE_DATE="$(python - <<'PY'
import datetime as dt, os
try:
    from zoneinfo import ZoneInfo
    tz = ZoneInfo((os.environ.get("TZ") or "America/New_York").strip() or "America/New_York")
except Exception:
    tz = dt.datetime.now().astimezone().tzinfo
today = dt.datetime.now(tz=tz).date()
print((today + dt.timedelta(days=1)).isoformat())
PY
)"

echo "[trade] $(date -Is) trade_date=${TRADE_DATE} (final 22:00 fetch)"

# Final 22:00 fetch (writes intraday row + predictions_latest + appends predictions_history)
python intraday_pulse.py \
  --trade-date "$TRADE_DATE" \
  --env "$WT_ENV" \
  --write-predictions

# Execute trades from the latest intraday-based predictions (kalshi_trader enforces additional gates).
ARGS=(
  python kalshi_trader.py
  --env "$WT_ENV"
  --trade-date "$TRADE_DATE"
  --predictions-csv Data/predictions_latest.csv
  --trades-log Data/trades_history.csv
  --decisions-log Data/decisions_history.csv
  --eval-log Data/eval_history.csv
  --min-confidence 0.75
  --max-spread 3.0
  --orderbook-depth 25
  --sigma-floor 2.0
  --sigma-mult 1.0
  --max-dollars-total "$WT_DAILY_BUDGET"
  --max-dollars-per-city "$WT_DAILY_BUDGET"
  --max-contracts-per-order "$WT_MAX_CONTRACTS_PER_ORDER"
)

case "$(echo "$WT_SEND_ORDERS" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|y) ARGS+=(--send-orders) ;;
esac

"${ARGS[@]}"

