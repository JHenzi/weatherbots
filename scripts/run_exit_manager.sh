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

echo "[exit_manager] $(date -Is) trade_date=${TRADE_DATE} cleanup=${CLEANUP:-false}"

ARGS=(
  python exit_manager.py
  --env "$WT_ENV"
  --trade-date "$TRADE_DATE"
)

# Pass --cleanup flag when script is called for the 12:30 cleanup pass.
if [[ "${CLEANUP:-false}" == "true" ]] || [[ "${1:-}" == "--cleanup" ]]; then
  ARGS+=(--cleanup)
fi

case "$(echo "$WT_SEND_ORDERS" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|y) ARGS+=(--send-orders) ;;
esac

"${ARGS[@]}"
