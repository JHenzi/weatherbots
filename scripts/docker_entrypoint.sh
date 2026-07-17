#!/bin/bash
set -euo pipefail

cd /app

mkdir -p Data/logs

# Make it obvious the container is alive, even before the first cron run.
echo "weather-trader container started: $(date -Is)" | tee -a /app/Data/logs/container.log

# THE REAL FIX: Export current environment variables to a shell-safe file for cron.
# This ensures cron jobs see WT_ENV, WT_SEND_ORDERS, and any keys passed via Docker.
declare -p | grep -E ' (WT_|KALSHI_|TZ|API_KEY|GOOGLE|TOMORROW|WEATHERAPI|OPENWEATHERMAP|PIRATE|NWS)' > /app/container.env

touch /app/Data/logs/trade.cron.log /app/Data/logs/calibrate.cron.log
touch /app/Data/logs/intraday_pulse.cron.log /app/Data/logs/dashboard.cron.log

# Optional: run once on startup (disabled by default).
WT_RUN_TRADE_ON_START="${WT_RUN_TRADE_ON_START:-false}"
WT_RUN_CALIBRATE_ON_START="${WT_RUN_CALIBRATE_ON_START:-false}"
WT_ALLOW_LIVE_ON_START="${WT_ALLOW_LIVE_ON_START:-false}"

_bool() { [[ "$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')" =~ ^(1|true|yes|y)$ ]]; }

if _bool "$WT_RUN_TRADE_ON_START"; then
  if _bool "${WT_SEND_ORDERS:-false}" && ! _bool "$WT_ALLOW_LIVE_ON_START"; then
    echo "Refusing to live-trade on startup. Set WT_ALLOW_LIVE_ON_START=true to allow." | tee -a /app/Data/logs/container.log
  else
    echo "Running trade job once on startup..." | tee -a /app/Data/logs/container.log
    # Load the captured env just in case
    /bin/bash -c ". /app/container.env && /app/scripts/run_trade.sh" >> /app/Data/logs/trade.cron.log 2>&1 || true
  fi
fi

if _bool "$WT_RUN_CALIBRATE_ON_START"; then
  echo "Running calibrate job once on startup..." | tee -a /app/Data/logs/container.log
  /bin/bash -c ". /app/container.env && /app/scripts/run_calibrate.sh" >> /app/Data/logs/calibrate.cron.log 2>&1 || true
fi

# Install the crontab for this container, but first ensure it sources the container env
sed "s|cd /app && |cd /app \&\& . /app/container.env \&\& |g" /app/ops/docker/crontab > /tmp/crontab.final
crontab /tmp/crontab.final

# Web dashboard (default on). Set WT_RUN_DASHBOARD_ON_START=false to disable.
WT_RUN_DASHBOARD_ON_START="${WT_RUN_DASHBOARD_ON_START:-true}"

DASH_PID=""

start_dashboard() {
  echo "Starting web dashboard on port 8080..." | tee -a /app/Data/logs/container.log
  . /app/container.env
  python /app/scripts/web_dashboard_api.py >> /app/Data/logs/dashboard.cron.log 2>&1 &
  DASH_PID=$!
  echo "dashboard pid=$DASH_PID at $(date -Is)" | tee -a /app/Data/logs/container.log
}

# Health probe: returns 0 if the dashboard answers on :8080, non-zero otherwise.
# Uses python because the base image ships no curl/wget.
dashboard_healthy() {
  python - <<'PY' >/dev/null 2>&1
import sys, urllib.request
try:
    urllib.request.urlopen("http://127.0.0.1:8080/", timeout=5)
except Exception:
    sys.exit(1)
sys.exit(0)
PY
}

# Dashboard supervisor: keep the dashboard alive AND responsive. Runs as a
# background subshell so cron can stay the foreground PID 1 (the original,
# proven setup). This catches the failure `restart: unless-stopped` cannot:
# the dashboard process wedging (alive but not answering) without the
# container ever exiting.
supervise_dashboard() {
  # set -e is unhelpful in a long-lived supervisor; tolerate transient errors.
  set +e
  local fail_count=0
  start_dashboard
  sleep 15
  while true; do
    if [ -z "$DASH_PID" ] || ! kill -0 "$DASH_PID" 2>/dev/null; then
      echo "dashboard process gone; relaunching at $(date -Is)" | tee -a /app/Data/logs/container.log
      wait "$DASH_PID" 2>/dev/null   # reap the exited child so it never zombies
      start_dashboard
      fail_count=0
    elif dashboard_healthy; then
      fail_count=0
    else
      fail_count=$((fail_count + 1))
      echo "dashboard health check failed ($fail_count) at $(date -Is)" | tee -a /app/Data/logs/container.log
      if [ "$fail_count" -ge 3 ]; then
        echo "dashboard unresponsive; killing pid=$DASH_PID and relaunching" | tee -a /app/Data/logs/container.log
        kill "$DASH_PID" 2>/dev/null
        sleep 3
        kill -9 "$DASH_PID" 2>/dev/null
        wait "$DASH_PID" 2>/dev/null
        start_dashboard
        fail_count=0
      fi
    fi
    sleep 30
  done
}

if _bool "$WT_RUN_DASHBOARD_ON_START"; then
  supervise_dashboard &
fi

# Cron stays the foreground process (PID 1), exactly as before. If it ever
# exits, the container exits and `restart: unless-stopped` brings it back.
exec cron -f

