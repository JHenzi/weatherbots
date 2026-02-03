# Operational runbook

How to run the system autonomously, control budget and live trading, and perform common tasks.

## Autonomous operation (Docker, recommended)

Goal: you do a **one-time setup**, then the container runs on its own:

- **Daily trade job**: runs `run_trade.sh` (intraday_pulse + kalshi_trader) at 07:00, 13:00, and 14:00 ET (13:00 local gate: ny/fl at 13:00 ET, il/tx at 14:00 ET).
- **Nightly calibration job**: runs `calibrate_sources.py` for *yesterday* to update per-source error logs + learned weights when NWS CLI truth is available.
- **Nightly settlement/metrics job**: runs `settle_eval.py` + `daily_metrics.py` for *yesterday* to backfill realized outcomes and roll up MAE/RMSE + hit-rate + PnL.

### One-time setup

1) Ensure `.env` has the required values:

- **Kalshi (required)**: `KALSHI_API_KEY_ID`, `KALSHI_PRIVATE_KEY_PATH`, `KALSHI_ENV`
- **weather.gov forecasts (recommended)**: `NWS_USER_AGENT=weather-trader (contact: you@example.com)`
- Optional: provider keys for forecasts (`GOOGLE`, `TOMORROW`, `WEATHERAPI`, `PIRATE_WEATHER_API_KEY`, `VISUAL_CROSSING_API_KEY`)

2) Install dependencies once:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

3) Start the Docker scheduler (runs automatically thereafter):

```bash
docker compose up -d --build
```

See **[Docker setup](docker_setup.md)** for mounting the Kalshi key, logs, and schedule details.

### Controlling budget and live trading

The scheduler wrapper scripts in `scripts/` support environment variables:

- **`WT_DAILY_BUDGET`**: total daily budget in dollars (default `50`)
- **`WT_ENV`**: `demo` or `prod` (default `demo`)
- **`WT_SEND_ORDERS`**: `true` to actually place orders (default: **dry-run**)

### Idempotency (one live trade per city per date)

When live trading is enabled, the bot enforces **one live trade per city per trade date**. It checks `Data/trades_history.csv` for an existing row with `send_orders=true` for that `(env, trade_date, city)`; if found, it skips with reason `already_traded` and does not submit another order.

To enable live trading, set in `docker-compose.yml`:

- `WT_ENV=prod`
- `WT_SEND_ORDERS=true`
- (optional) `WT_DAILY_BUDGET=50`

Then restart the container: `docker compose up -d`

### Where to look for logs

- `Data/logs/trade.cron.log`
- `Data/logs/calibrate.cron.log`
- `Data/logs/settle.cron.log`
- Decisions/trades/eval CSVs in `Data/`:
  - `Data/decisions_history.csv`
  - `Data/trades_history.csv`
  - `Data/eval_history.csv`
  - `Data/daily_metrics.csv`
  - `Data/source_performance.csv`
  - `Data/weights.json` / `Data/weights_history.csv`

---

## Autonomous operation (macOS, legacy / optional)

If you prefer `launchd` on macOS, see the old LaunchAgents under `ops/launchd/`. Docker is the recommended path on macOS due to more predictable scheduling and fewer permission surprises.

---

## Generate predictions (recommended: forecast or blend)

```bash
source .venv/bin/activate

# Forecast consensus (Open-Meteo + Visual Crossing)
python daily_prediction.py --trade-date 2026-01-23 --prediction-mode forecast --skip-fetch

# Blend forecast with LSTM (useful when you want some historical smoothing)
python daily_prediction.py --trade-date 2026-01-23 --prediction-mode blend --blend-forecast-weight 0.9 --skip-fetch
```

## Refresh local historical pickles (for LSTM input continuity)

```bash
python daily_prediction.py --trade-date 2026-01-23 --prediction-mode lstm
```

This fetches a recent observed window ending at `trade_date - 1` and regenerates `Data/prediction_data_cleaned_<city>.pkl`.

## Dry-run trade selection (no orders)

```bash
python kalshi_trader.py --env demo --trade-date 2026-01-23 --predictions-csv Data/predictions_latest.csv
```

This performs **orderbook/quote-aware trade selection** per city:

- Computes bucket probabilities from a Normal approximation around the voting consensus.
- Pulls market pricing (prefers orderbook; falls back to `Get Market` quotes).
- Computes **EV/edge** and picks the **best-EV YES** bucket per city (or skips).
- Writes audit logs:
  - `Data/decisions_history.csv` (trade vs skip + reason)
  - `Data/eval_history.csv` (model + market context for later scoring)

Key knobs (conservative defaults):

- `--min-ev-cents` (default 3)
- `--max-yes-spread-cents` (default 6)
- `--max-dollars-per-city` (default 50)
- `--max-dollars-total` (default 150)
- `--max-contracts-per-order` (default 500)
- `--sigma-floor` (default 2.0) and `--sigma-mult` (default 1.0)

For **hourly 13:00 local gate**: do not pass `--skip-13h-gate`. To trade all cities in one run (e.g. one-shot), pass `--skip-13h-gate`.

## Place orders (demo first)

Run with `--env demo` first. Once satisfied, use `--env prod` and `--send-orders` for live trading.

```bash
python kalshi_trader.py --env demo --trade-date 2026-01-23 --send-orders
```
