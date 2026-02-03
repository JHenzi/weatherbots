# Docker setup (autonomous scheduler)

Recommended setup: a single container runs **cron** internally (trade, calibrate, settle). No need for host cron or launchd.

## 1) Prepare `.env`

Copy `.env.example` to `.env` and set at least: `KALSHI_API_KEY_ID`, `KALSHI_PRIVATE_KEY_PATH`, `KALSHI_ENV` (e.g. `prod` or `demo`), and `NWS_USER_AGENT`. Add forecast provider keys as needed.

**→ Full list: [Environment variables](environment_variables.md)**

## 2) Mount your Kalshi private key

Edit `docker-compose.yml` to mount your PEM key file and set:

- `KALSHI_PRIVATE_KEY_PATH=/run/secrets/kalshi_key.pem`

Example volume (uncomment and set the correct local path):

```yaml
volumes:
  - ./gooony.txt:/run/secrets/kalshi_key.pem:ro
```

> **Warning:** Cron runs inside the container. The path in `.env` must exist **inside** the container. A common failure is `KALSHI_PRIVATE_KEY_PATH` pointing to a host-only path that does not exist in the container.

## 3) Run it (one command)

From the repo root:

```bash
docker compose up -d --build
```

> Jobs run on the schedule in `ops/docker/crontab`. If you just started the container, the next run may not be until the next scheduled time.

### One-off runs (manual)

```bash
docker exec weather-trader /bin/bash /app/scripts/run_trade.sh
docker exec weather-trader /bin/bash /app/scripts/run_calibrate.sh
docker exec weather-trader /bin/bash /app/scripts/run_settle.sh
```

To force a safe dry-run (no orders placed):

```bash
docker exec -it -e WT_ENV=prod -e WT_SEND_ORDERS=false weather-trader /bin/bash /app/scripts/run_trade.sh
```

Optional: enable “run once on startup” by adding to `docker-compose.yml` environment:

- `WT_RUN_TRADE_ON_START=true`
- `WT_RUN_CALIBRATE_ON_START=true`

## 4) Logs and outputs

- Container logs: `docker logs -f weather-trader`
- Cron logs (persisted in `Data/`): `Data/logs/trade.cron.log`, `Data/logs/calibrate.cron.log`, `Data/logs/settle.cron.log`
- Decisions/trades/eval CSVs: `Data/decisions_history.csv`, `Data/trades_history.csv`, `Data/eval_history.csv`
- Learning + rollups: `Data/source_performance.csv`, `Data/weights.json`, `Data/weights_history.csv`, `Data/daily_metrics.csv`

## 5) Data persistence and risk of data loss

The system uses **both** CSV files under `Data/` and, when configured, **Postgres**. Postgres is an optional mirror when `ENABLE_PG_WRITE` is set, and an optional read source (with CSV fallback) when `ENABLE_PG_READ` is set.

- **Container filesystem**: If `./Data` is **not** mounted into the container, all state (CSVs, models, weights, logs, caches) is **ephemeral** and lost when the container is recreated. **Always mount `./Data`** (or an equivalent host directory).
- **CSV is still primary**: Scripts write to CSV first; Postgres is a mirror when enabled. If you lose the `Data/` directory, you lose history even if Postgres is attached.
- **Dual-write inconsistency**: When both CSV and Postgres are used, they can get out of sync if one write fails. Rely on CSV for operational truth until the Postgres transition is complete.

**Recommendations:** (1) Always mount `./Data` in Docker. (2) Back up the host directory that holds `Data/`. (3) If you use Postgres, set `ENABLE_PG_WRITE` and ensure the DB is durable and backed up.

## 6) Schedule / timezone

Cron times are defined in `ops/docker/crontab`. Container timezone is set by `TZ` in `docker-compose.yml` (e.g. `America/New_York`).

- **Intraday pulse**: hourly at :00 and at :30 for hours 0–16 (41 runs/day).
- **Trade**: 07:00 (early refresh), **13:00 ET** (ny/fl trade at 13:00 local), **14:00 ET** (il/tx trade at 13:00 CT).
- **Calibrate**: 02:15, retry 06:30.
- **Settle/metrics**: 03:15, retry 07:15.

**13:00 local gate:** `kalshi_trader.py` only executes the trade for a city when it is **13:00 (1 PM)** in that city's local time (ny/fl: America/New_York, il/tx: America/Chicago). At 13:00 ET, ny and fl get fresh forecasts (from the same run) then trade; at 14:00 ET, il and tx get fresh forecasts then trade. Do not pass `--skip-13h-gate` for this use.
