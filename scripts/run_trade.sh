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
import datetime as dt
print((dt.date.today() + dt.timedelta(days=1)).isoformat())
PY
)"

ARGS=(
  python run_daily.py
  --env "$WT_ENV"
  --trade-date "$TRADE_DATE"
  --prediction-mode forecast
  --refresh-history
  --min-confidence 0.5
  --max-spread 3.0
  --min-ev-cents 3
  --max-yes-spread-cents 6
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

