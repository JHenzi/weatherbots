#!/bin/bash
set -euo pipefail

cd /app

mkdir -p Data/logs

# Make it obvious the container is alive, even before the first cron run.
echo "weather-trader container started: $(date -Is)" | tee -a /app/Data/logs/container.log
touch /app/Data/logs/trade.cron.log /app/Data/logs/calibrate.cron.log
touch /app/Data/logs/intraday_pulse.cron.log

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
    /bin/bash /app/scripts/run_trade.sh >> /app/Data/logs/trade.cron.log 2>&1 || true
  fi
fi

if _bool "$WT_RUN_CALIBRATE_ON_START"; then
  echo "Running calibrate job once on startup..." | tee -a /app/Data/logs/container.log
  /bin/bash /app/scripts/run_calibrate.sh >> /app/Data/logs/calibrate.cron.log 2>&1 || true
fi

# Install the crontab for this container
crontab /app/ops/docker/crontab

# Start cron in foreground
exec cron -f

