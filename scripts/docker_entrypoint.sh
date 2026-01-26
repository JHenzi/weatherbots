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

# Start cron in foreground
exec cron -f

