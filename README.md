## Weather-trader (operational)

This repo predicts **daily maximum temperature** for 4 locations and maps the prediction into Kalshi “high temperature” markets for automated (or dry-run) trading.

### Inspiration and origin

This project is **inspired by and builds on** the [LSTM-Automated-Trading-System](https://github.com/pranavgoyanka/LSTM-Automated-Trading-System) repo (Kalshi Weather Prediction Common Task for BU's CS542 Spring 2024). It reuses and extends the same core ideas:

- **LSTM models** for daily max-temperature prediction (one model per city, trained on historical weather data)
- **Data pipeline**: multiple weather APIs (Open-Meteo, Visual Crossing, Meteostat, NOAA NCEI), merged and cleaned into per-city DataFrames, then pickled for training and inference
- **Kalshi trading**: automated (or dry-run) trading on Kalshi daily high-temperature markets using predictions

The original repo used a 10-day timestep LSTM, Adam optimizer, and 80 epochs; this repo keeps the same LSTM architecture and data sources while adding provider-forecast blending, calibration, Docker scheduling, and richer operational tooling.

**Mapping from the LSTM-Automated-Trading-System repo:** data fetching → `data_fetcher_new.ipynb` (or `daily_prediction.py` for daily ingest); model training → `train_models.py` (cf. `data_lstm.ipynb`); daily prediction → `daily_prediction.py`; Kalshi trading → `kalshi_trader.py`. Trained models live in `Data/model_<city>.keras`; cleaned data in `Data/prediction_data_cleaned_<city>.pkl`.

## Documentation

Project docs live in `documentation/`:

- **System architecture**: `documentation/system_architecture.md`
- **Mathematical foundations (weights, sigma, EV)**: `documentation/mathematical_foundations.md`
- **Data files + schemas**: `documentation/data_reference.md`
- **Audit report (risks/strengths)**: `documentation/audit_results.md`
- **Improvement roadmap**: `documentation/improvement_roadmap.md`

### Cities / coordinates
- **NYC (Central Park / Belvedere Castle)**: `40.79736,-73.97785` (`ny`)
- **Chicago (Midway Intl)**: `41.78701,-87.77166` (`il`)
- **Austin (Bergstrom Intl)**: `30.14440,-97.66876` (`tx`)
- **Miami**: `25.77380,-80.19360` (`fl`)

### Features
- **Prediction modes**: LSTM-only, provider-forecast-only, or blended (configurable weight). Multiple providers: Open-Meteo, Visual Crossing, Tomorrow.io, WeatherAPI, OpenWeatherMap, Pirate Weather, NWS.
- **Calibration**: Nightly job updates per-source MAE and writes `Data/weights.json`; consensus prediction is a weighted average of provider forecasts (and optional LSTM).
- **Intraday pulse**: Cron runs prediction snapshots (e.g. 09:00 / 15:00 / 21:00) and persists today + tomorrow so the dashboard always has the next trade date.
- **Trading**: Orderbook-aware trade selection (EV-based), per-city sigma from spread + historical MAE, budget allocation by confidence + historical MAE/bucket hit-rate.
- **Dashboard**: TUI (terminal) and web UI — live NWS observations, next-trade predictions, Risk/Sell advisor, at-risk brackets (BUY NO), positions, and analytics.
- **Measurements**: Per-source and consensus MAE, bucket hit rate, realized PnL; all logged to CSVs and optionally mirrored to Postgres.

### Measurements (what we track)
| Artifact | Purpose | Key fields |
|----------|---------|------------|
| `Data/source_performance.csv` | Per-source prediction error vs NWS actual | `date`, `city`, `source_name`, `predicted_tmax`, `actual_tmax`, `absolute_error` |
| `Data/daily_metrics.csv` | Rollups for allocation & scoring | `trade_date`, `city`, `metric_type` (e.g. `mae_f`, `bucket_hit_rate`), `value` |
| `Data/eval_history.csv` | Per-trade outcome and market state | `trade_date`, `city`, `settlement_tmax_f`, `bucket_hit`, `realized_pnl_dollars`, `yes_ask`, `ev_cents` |
| `Data/city_metadata.json` | Per-city historical MAE (used for σ) | `cities.<code>.historical_MAE` |

- **MAE**: Mean absolute error (°F) of predictions vs NWS CLI actual; computed per source and for consensus over a rolling window.
- **Bucket hit rate**: Fraction of trades where the chosen Kalshi temperature bucket contained the settled max temp.
- **PnL**: Realized profit/loss per trade (and rolled up per day/city) after settlement.

---

## Quick start

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### Configure secrets
- Copy `.env.example` → `.env`
- Fill in:
  - `VISUAL_CROSSING_API_KEY`
  - `GOOGLE` (Google Weather API key; optional)
  - `KALSHI_API_KEY_ID`
  - `KALSHI_PRIVATE_KEY_PATH` (points to your RSA PEM file; for Docker we recommend `/run/secrets/kalshi_key.pem`)
  - `KALSHI_ENV` (`demo` or `prod`)

> [!IMPORTANT]
> `.env` and private-key files are gitignored. Never commit secrets.

---

## Docker (autonomous scheduler)

If you don’t want `launchd`/cron on macOS, the recommended setup is Docker Compose:
- a single container runs **cron** internally (trade + calibrate)
- `./Data` is mounted for persistence (models/logs/histories/caches)
- `./.env` is mounted read-only
- your Kalshi private key PEM is mounted as a read-only secret file

### 1) Prepare `.env`
Make sure your `.env` contains (at minimum):
- `KALSHI_API_KEY_ID=...`
- `KALSHI_ENV=prod` (or `demo` while testing)
- any forecast provider keys you want
- `NWS_USER_AGENT=weather-trader (contact: you@example.com)` (for api.weather.gov)

### 2) Mount your Kalshi private key
Edit `docker-compose.yml` to mount your PEM key file and set:
- `KALSHI_PRIVATE_KEY_PATH=/run/secrets/kalshi_key.pem`

> [!WARNING]
> Cron runs inside the container. The path in `.env` must exist **inside** the container. A common failure is `KALSHI_PRIVATE_KEY_PATH` pointing to a host-only path (e.g. `gooony.txt`) that does not exist in the container.

### 3) Run it (one command)
From the repo root:

```bash
docker compose up -d --build
```

> [!NOTE]
> Jobs run on the schedule in `ops/docker/crontab`. If you just started the container, the next run may not be until the next scheduled time.

To trigger a one-off run immediately (manual):

```bash
docker exec weather-trader /bin/bash /app/scripts/run_trade.sh
docker exec weather-trader /bin/bash /app/scripts/run_calibrate.sh
docker exec weather-trader /bin/bash /app/scripts/run_settle.sh
```

> [!TIP]
> To force a safe dry-run (no orders placed), run with `WT_SEND_ORDERS=false`:

To force a safe manual dry-run regardless of container defaults:

```bash
docker exec -it -e WT_ENV=prod -e WT_SEND_ORDERS=false weather-trader /bin/bash /app/scripts/run_trade.sh
```

Or enable “run once on startup” by adding these to `docker-compose.yml` environment:
- `WT_RUN_TRADE_ON_START=true`
- `WT_RUN_CALIBRATE_ON_START=true`

### 4) Logs and outputs
- Container logs: `docker logs -f weather-trader`
- Cron logs (persisted): `Data/logs/trade.cron.log`, `Data/logs/calibrate.cron.log`, `Data/logs/settle.cron.log`
- Decisions/trades/eval CSVs: `Data/decisions_history.csv`, `Data/trades_history.csv`, `Data/eval_history.csv`
- Learning + rollups:
  - `Data/source_performance.csv` (per-source absolute errors when truth exists)
  - `Data/weights.json` (current learned weights)
  - `Data/weights_history.csv` (weight drift over time)
  - `Data/daily_metrics.csv` (MAE/RMSE + bucket hit-rate + daily PnL)

### 5) Schedule / timezone
Cron times are defined in `ops/docker/crontab` (defaults: intraday pulses at 09:00/15:00/21:00, trade at 22:00, calibrate at 02:15, settle/metrics at 03:15).
If you want cron to run in your local timezone, set `TZ` in `docker-compose.yml` (e.g. `America/New_York`).

---

## To Get Predictions

Run intraday_pulse.py:

```bash
docker exec -it weather-trader /bin/bash -lc '
d=$(date +%F)
python /app/intraday_pulse.py --trade-date "$d" --env "${WT_ENV:-prod}" --write-predictions
'
```

Problem is, this writes to disk, we can't see them unless we run the dashboard (see **Dashboard (how-to)** below). Or you can print output:

```bash
docker exec -it weather-trader /bin/bash -lc '
d=$(date +%F)
python /app/intraday_pulse.py --trade-date "$d" --env "${WT_ENV:-prod}" --print --no-write
'
```

## Dashboard (how-to)

### Web dashboard (recommended)
Single process: background NWS observation fetch + FastAPI API + HTML dashboard. Serves live station data, next-trade predictions, Risk/Sell advisor, at-risk brackets, and positions.

**Run from repo root:**
```bash
python scripts/web_dashboard_api.py
```
Then open **http://localhost:8080/** in a browser.

**Pages:**
- **/** — Main dashboard: city cards (current temp, observed high today, projected high, trend °/hr, sun progress), next-trade predictions table (consensus, sources, spread, confidence), Risk/Sell advisor (hedge signals + desktop notifications), at-risk brackets (BUY NO suggestions), and open positions with P/L.
- **/markets** — Kalshi markets feed: upcoming high-temp markets with filters and links to Kalshi.
- **/analytics** — Forecasts vs actuals (MAE per source and per city/source), observation projected_high vs actual (MAE by city), and "when to lock in" (by hour and trend bucket). Data from `source_performance.csv` and `observations_history.csv`.

> [!NOTE]
> Observations are fetched from **api.weather.gov** every 2 minutes. The dashboard uses the **latest** observation per station; the [NWS Time Series Viewer](https://www.weather.gov/wrh/timeseries) (hourly mode) can show slightly different values because it uses only hourly observations (e.g. :51–:59). Both are from the same station.

Observations are written to `Data/observations_latest.json` and `Data/observations_history.csv`.

**Optional (Docker):** Expose port 8080 in `docker-compose.yml` and run the dashboard inside a sidecar or on the host pointing at the same `Data/` directory so it can read predictions and trades.

### TUI (terminal dashboard)
Ncurses-style status view: file row counts (predictions, trades, decisions, eval, intraday), latest predictions table, forecast comparison (all sources), recent trades, next cron runs.

```bash
./scripts/dashboard_live.sh [env] [interval_sec] [limit] [city]
```
- **env**: `prod` (default) or `demo` — filters trades by environment.
- **interval_sec**: Refresh interval in seconds (default `5`).
- **limit**: Number of recent rows to show (default `10`).
- **city**: Optional — focus forecast comparison on one city (`ny` / `il` / `tx` / `fl`). Omit to show all cities.

Example: `./scripts/dashboard_live.sh prod 5 10` — refresh every 5 s, show 10 rows, all cities. Press **q** to quit.

---

## Weights + math (what's actually running)

### A) Learned provider weights (`Data/weights.json`)
Updated by the nightly calibration job (`scripts/run_calibrate.sh` → `calibrate_sources.py`) once NWS CLI “truth” is available.

- **Inputs**
  - Predictions for a trade date (from `Data/predictions_history.csv`)
  - Actual max temp from NWS CLI (via `truth_engine.py`)
  - Logged errors in `Data/source_performance.csv`
- **Per-city per-source error window (inclusive)**
  - last \(N\) days ending at `as_of` (the calibrated `trade_date`)
  - \([as\_of-(N-1),\ as\_of]\)
- **MAE + weights**
  - \(MAE_i = \text{mean}(|pred_i - actual|)\) over the window
  - \(w_i \propto 1/MAE_i^2\), normalized so \(\sum_i w_i = 1\)

### B) Intraday forecast “consensus” (`intraday_pulse.py`)
Cron runs `scripts/run_intraday_pulse.sh` at **09:00 / 15:00 / 21:00** local container time. It targets the configured `trade_date` (e.g. tomorrow in cron, or today for the 07:00 run):

- For each city, fetches provider forecasts (Open‑Meteo / Visual Crossing / Tomorrow / WeatherAPI / Google / OpenWeatherMap / PirateWeather / weather.gov).
- **Mean forecast (\(\mu\))**
  - uses `Data/weights.json` **if it contains weights for those providers**
  - otherwise falls back to **equal weights** across available providers
  - \(\mu = \sum_i w_i x_i\)
- **Snapshot spread (“sigma”)**
  - \(\sigma_{\text{snapshot}} = \text{pstdev}(\{x_i\})\) across available provider forecasts
- **Confidence score**
  - A blended measure combining:
    - spread-based agreement (low \(\sigma_{\text{snapshot}}\) ⇒ higher raw confidence), and
    - an entropy-based skill score derived from the learned provider weights for that city.
  - Final confidence is capped below 1.0 and is written as `confidence_score` alongside `spread_f`.

Outputs:
- `Data/intraday_forecasts.csv` (append-only snapshots)
- optionally `Data/predictions_latest.csv` + `Data/predictions_history.csv` when run with `--write-predictions`

### C) Trading distribution sigma (`kalshi_trader.py`)
When trading, the bot uses:

- `spread_f` from the predictions row (typically the intraday snapshot std dev)
- `historical_MAE` from `Data/city_metadata.json`

and sets:

- \(\sigma = \max(spread\_f,\ historical\_MAE)\)

This is the \(\sigma\) used for bucket probabilities and EV logging.


---

## Data flow (end-to-end)

### 1) Historical + daily weather ingestion
The ingestion logic lives in `daily_prediction.py` (and historically in `data_fetcher_new.ipynb`), following the same data-sources approach as the [LSTM-Automated-Trading-System](https://github.com/pranavgoyanka/LSTM-Automated-Trading-System) repo.

**Data sources used** (inspired by that repo):

| Source | Features used |
|--------|----------------|
| [Open-Meteo](https://open-meteo.com/) | Maximum temperature, precipitation |
| [Visual Crossing](https://www.visualcrossing.com/) | Maximum temperature, humidity |
| [Meteostat](https://meteostat.net/en/) | Maximum temperature, minimum temperature |
| [NOAA NCEI](https://www.ncei.noaa.gov/) | Maximum temperature, minimum temperature |

**Details**
- **Open-Meteo** archive API: tmax/tmin, sunshine duration, precipitation hours, wind
- **Visual Crossing** timeline API: tmax/tmin, humidity, wind (requires API key)
- **Meteostat**: tmax/tmin (and other station fields)
- **NOAA NCEI**: daily summaries (tmax/tmin + many station variables)

**Stored artifacts (per city, in `Data/`)**
- `merged_df_<city>.pkl`: historical merged dataframe (2016→…)
- `prediction_merged_df_<city>.pkl`: historical + most recent ingested window (used for daily operation)
- `prediction_data_cleaned_<city>.pkl`: cleaned/feature-engineered frame used by prediction

### 2) Cleaning + feature engineering
Cleaning is performed inside `daily_prediction.py` during the daily run:

- **Unit normalization**:
  - Visual Crossing + NCEI are treated as already **°F**
  - Open-Meteo + Meteostat are treated as **°C** and converted to °F via \(F = C \cdot \frac{9}{5} + 32\)
- **Derived features**:
  - `day` = day-of-year (1..365)
  - `tmax_avg` = mean over available sources \(`tmax_vc`, `tmax_om`, `tmax_ms`, `tmax_ncei`\)
  - `tmin_avg` = mean over available sources \(`tmin_vc`, `tmin_om`, `tmin_ms`, `tmin_ncei`\)
- **Missing values**:
  - forward-fill/back-fill is used for prediction-time continuity (`ffill()`/`bfill()`)

### 3) Prediction (three modes)
`daily_prediction.py` supports:

- **`--prediction-mode lstm`**: uses the trained LSTM model only.
- **`--prediction-mode forecast`**: uses provider forecasts only.
  - Open-Meteo forecast endpoint (no key) + Visual Crossing forecast (key) + Tomorrow.io forecast (key via `TOMORROW`) + WeatherAPI.com forecast (key via `WEATHERAPI`), averaged over available sources.
  - Optional: OpenWeatherMap forecast (key via `OPENWEATHERMAP_API_KEY`), aggregated from 3-hourly forecasts into the local-day max.
  - Optional: Pirate Weather forecast (key via `PIRATE_WEATHER_API_KEY`), daily `temperatureMax` for the local day.
  - Optional: weather.gov (NWS) forecast (no key; requires `NWS_USER_AGENT`), selects the daytime period temperature for the trade date.
- **`--prediction-mode blend`**: weighted mix of forecast and LSTM:
  - `pred = w * forecast + (1 - w) * lstm` (default `w=0.8`, configurable via `--blend-forecast-weight`)

This is important operationally: the LSTM is an autoregressive model over recent observed days and can diverge from “online consensus”; `forecast`/`blend` are usually better aligned with what markets price.

### 4) Writing predictions
The trading entrypoint is `run_daily.py`, which writes:
- `Data/predictions_latest.csv` (canonical “what we traded on” for that run; overwritten each run)
- `Data/predictions_history.csv` (append-only history)

Current per-city schema includes:
- `date` (the **trade date** / event date)
- `city` (`ny`, `il`, `tx`, `fl`)
- Per-source forecasts (nullable): `tmax_open_meteo`, `tmax_visual_crossing`, `tmax_tomorrow`, `tmax_weatherapi`, `tmax_openweathermap`, `tmax_pirateweather`, `tmax_weather_gov`, `tmax_lstm`
- Voting/consensus outputs:
  - `tmax_predicted` (the value used for trading)
  - `sources_used` (comma-separated)
  - `weights_used` (comma-separated `source:weight`)
  - `spread_f`, `confidence_score`

Recommended: use a fresh output per run/mode, e.g. `--predictions-csv Data/predictions_latest.csv`.

### Provider rate limits (important)

> [!WARNING]
> Free-tier APIs have strict limits. Tomorrow.io: 3 req/sec, 25 req/hour, 500 req/day. Avoid re-running forecast mode in a tight loop.

- **Tomorrow.io (free tier)**: 3 req/sec, 25 req/hour, 500 req/day.
  - `daily_prediction.py` uses a **1-hour on-disk cache** (`Data/tomorrow_cache`) and a simple throttle to help stay within free-tier limits.
  - Operationally: avoid repeatedly re-running forecast mode in a tight loop.
- **OpenWeatherMap**:
  - `daily_prediction.py` uses a **1-hour on-disk cache** (`Data/openweathermap_cache`) to reduce repeat calls.
- **Pirate Weather**:
  - `daily_prediction.py` uses a **1-hour on-disk cache** (`Data/pirateweather_cache`) to reduce repeat calls.

### 5) Kalshi mapping + (dry-run) trading
Trading logic is in `kalshi_trader.py`.

### Budgeting + allocation (important)

> [!IMPORTANT]
> Each trade run enforces **two layers of safety**: a configured daily cap (`WT_DAILY_BUDGET`) and a balance-based cap (never more than half of available cash per run). Budget is then allocated across cities by confidence and historical MAE/bucket hit-rate.

Each trade run enforces **two layers of safety**:

- **Configured cap**: `WT_DAILY_BUDGET` (passed as `--max-dollars-total`) is the *absolute* max the bot is allowed to spend in a day.
- **Balance-based cap (hard rule)**: before sizing orders, the bot calls `GET /trade-api/v2/portfolio/balance` and applies:
  - **per-run spend cap = min(configured_cap, 0.5 × available_cash)** by default
  - this ensures we **never risk more than half the current balance** even if `WT_DAILY_BUDGET` is higher.
  - note: if balance fetch is unavailable (commonly in demo), it falls back to the configured cap

Then, the per-run budget is **allocated across cities** based on:
- **today’s confidence** (`confidence_score` from spread/guardrails), and
- **historical feedback** from `Data/daily_metrics.csv` (rolling window; default 14 days), using:
  - consensus MAE (`source_name=consensus`) and
  - bucket hit-rate (`metric_type=bucket_hit_rate`)

This results in a per-city budget cap that prefers the **most reliable cities** over time (with a small minimum allocation per city so cities aren’t accidentally starved).

If you want to keep it simple and split evenly across cities, run `kalshi_trader.py` with:
- `--allocation-mode equal`

### Order sizing (contracts)
By default, orders are **auto-sized** to spend up to the **city’s allocated budget** (and the overall per-run cap), bounded by:
- `--max-contracts-per-order` (default in Docker wrapper: `WT_MAX_CONTRACTS_PER_ORDER=500`)

Liquidity note: the trader **reads** best-ask depth (`ask_qty`) and prints warnings when your desired size exceeds displayed depth. It does **not** automatically cap `count` to `ask_qty`; if live trading is enabled it submits a single limit order and any unfilled remainder may rest on the book at the limit price.

If you want fixed sizing instead, pass `--count N` (where `N>0`) to `kalshi_trader.py`.

### Probability “width” (sigma) is now city-aware
For each city, the trader now sets:

- `sigma = max(current_spread, historical_MAE)`

Where:
- `current_spread` is the forecast disagreement (`spread_f`) for that city/day
- `historical_MAE` comes from `Data/city_metadata.json` (updated nightly by `update_city_metadata.py`)

This lets predictable cities trade with tighter distributions while volatile cities stay wider.

## Kalshi market mapping + resolution (confirmed)

This project targets “daily high temperature” markets whose settlement source is the **National Weather Service (NWS) climatological report** for a specific station.

### Series tickers (demo + production)
We currently default to the `KX*` series because they exist in the Kalshi demo environment. The non-`KX` variants exist as well, but demo availability can differ by ticker.

| City | Code | Default series ticker | NWS station used for settlement | NWS “Daily Climate Report (CLI)” link |
| --- | --- | --- | --- | --- |
| New York City | `ny` | `KXHIGHNY` | **NYC (Central Park)** | `https://forecast.weather.gov/product.php?site=OKX&product=CLI&issuedby=NYC` |
| Chicago | `il` | `KXHIGHCHI` | **MDW (Midway)** | `https://forecast.weather.gov/product.php?site=LOT&product=CLI&issuedby=MDW` |
| Austin | `tx` | `KXHIGHAUS` | **AUS (Austin Bergstrom)** | `https://forecast.weather.gov/product.php?site=EWX&product=CLI&issuedby=AUS` |
| Miami | `fl` | `KXHIGHMIA` | **MIA (Miami Intl)** | `https://forecast.weather.gov/product.php?site=MFL&product=CLI&issuedby=MIA` |

You can override series tickers via env vars:
- `KALSHI_SERIES_NY`, `KALSHI_SERIES_IL`, `KALSHI_SERIES_TX`, `KALSHI_SERIES_FL`

### Contract certification / terms (resolution details)
Kalshi publishes a product certification PDF (“contract_url”) and contract terms PDF (“contract_terms_url”) for each series. These documents define the **resolution procedure**, including:
- which NWS report/station is authoritative,
- when the report is considered final for settlement (revisions after expiration are not used),
- how edge cases are handled when “yesterday” vs “today” wording appears in the CLI report.

Examples (NYC):
- Product certification: `https://kalshi-public-docs.s3.us-east-1.amazonaws.com/regulatory/product-certifications/NHIGH.pdf`
- Contract terms: `https://kalshi-public-docs.s3.amazonaws.com/contract_terms/NHIGH.pdf`

**How contract selection works**
- Build an **event ticker** by combining a “series ticker” and the date suffix `YYMONDD` (uppercased), e.g. `KXHIGHNY-26JAN23`.
- Fetch the event via:
  - `GET /trade-api/v2/events/{event_ticker}?with_nested_markets=true`
- Each event has markets whose `subtitle` encodes the temperature bucket (e.g. `71° to 72°`, `40° or above`, `76° or below`).
- `kalshi_trader.py` parses those subtitles and chooses the bucket that contains the predicted temperature.

**Authentication**
Kalshi Trade API v2 requests are signed with:
- `KALSHI-ACCESS-KEY` (your key id)
- `KALSHI-ACCESS-TIMESTAMP` (ms)
- `KALSHI-ACCESS-SIGNATURE` = RSA-PSS signature of `timestamp + HTTP_METHOD + PATH_WITHOUT_QUERY`

**Dry-run first**

> [!CAUTION]
> By default the trader does **not** place orders; it prints what it *would* submit. It will only place orders if you pass `--send-orders`. Test with `--env demo` and dry-run before enabling live trading.

---

## Operational runbook

### Autonomous operation (Docker, recommended)

Goal: you do a **one-time setup**, then the container runs on its own:
- **Daily trade job**: runs `run_daily.py` for *tomorrow’s* markets with a **$50 total cap** (default), and logs decisions/trades/eval.
- **Nightly calibration job**: runs `calibrate_sources.py` for *yesterday* to update per-source error logs + learned weights when NWS CLI truth is available.
- **Nightly settlement/metrics job**: runs `settle_eval.py` + `daily_metrics.py` for *yesterday* to backfill realized outcomes and roll up MAE/RMSE + hit-rate + PnL.

#### One-time setup

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

#### Controlling budget and live trading

The scheduler wrapper scripts in `scripts/` support environment variables:
- **`WT_DAILY_BUDGET`**: total daily budget in dollars (default `50`)
- **`WT_ENV`**: `demo` or `prod` (default `demo`)
- **`WT_SEND_ORDERS`**: `true` to actually place orders (default: **dry-run**)

#### Idempotency (one live trade per city per date)

> [!IMPORTANT]
> When live trading is enabled, the bot enforces **one live trade per city per trade date**. It checks `Data/trades_history.csv` for an existing row with `send_orders=true` for that `(env, trade_date, city)`; if found, it skips with reason `already_traded` and does not submit another order.

To enable live trading, set in `docker-compose.yml`:
- `WT_ENV=prod`
- `WT_SEND_ORDERS=true`
- (optional) `WT_DAILY_BUDGET=50`

Then restart the container: `docker compose up -d`

#### Where to look for logs
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

### Autonomous operation (macOS, legacy / optional)
If you prefer `launchd` on macOS, see the old LaunchAgents under `ops/launchd/`. Docker is the recommended path on macOS due to more predictable scheduling and fewer permission surprises.

### Generate predictions (recommended: forecast or blend)

```bash
source .venv/bin/activate

# Forecast consensus (Open-Meteo + Visual Crossing)
python daily_prediction.py --trade-date 2026-01-23 --prediction-mode forecast --skip-fetch

# Blend forecast with LSTM (useful when you want some historical smoothing)
python daily_prediction.py --trade-date 2026-01-23 --prediction-mode blend --blend-forecast-weight 0.9 --skip-fetch
```

### Refresh local historical pickles (for LSTM input continuity)

```bash
python daily_prediction.py --trade-date 2026-01-23 --prediction-mode lstm
```

This fetches a recent observed window ending at `trade_date - 1` and regenerates `Data/prediction_data_cleaned_<city>.pkl`.

### Dry-run trade selection (no orders)

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

### Place orders (demo first)

> [!TIP]
> Run with `--env demo` first. Once satisfied, use `--env prod` and `--send-orders` for live trading.

```bash
python kalshi_trader.py --env demo --trade-date 2026-01-23 --send-orders
```

---

## LSTM model details (inspired by LSTM-Automated-Trading-System)

The LSTM setup is **inspired by and aligned with** the [LSTM-Automated-Trading-System](https://github.com/pranavgoyanka/LSTM-Automated-Trading-System) repo: LSTMs are well-suited to weather forecasting because they capture sequential patterns and seasonality (trend, seasonality, residual) in temperature over time.

**Training** is in `train_models.py` (and historically `data_lstm.ipynb` in the upstream repo):

- **One model per city**, saved as `Data/model_<city>.keras` (e.g. `model_fl.keras`, `model_il.keras`, `model_ny.keras`, `model_tx.keras`) and optionally versioned under `Data/models/<YYYYMMDD>/`
- **Input window**: `time_steps = 10` days (same as the original repo)
- **Features used**: `day_of_year`, `tmax`, `tmin`, and optionally `prec`, `humi` (from the merged multi-source data)
- **Optimizer**: Adam (same as upstream)
- **Epochs**: configurable (e.g. 30 in this repo; the original used 80)
- **Preprocessing**: `StandardScaler` fit on the full feature matrix; inference uses the same feature set. The model predicts *scaled* `tmax`; a dummy feature vector is inverse-transformed to recover °F.

**Cleaned data for LSTM** (same naming as upstream): after cleaning and feature engineering, per-city data is stored as `Data/prediction_data_cleaned_<city>.pkl` (e.g. `prediction_data_cleaned_fl.pkl`, `prediction_data_cleaned_il.pkl`, `prediction_data_cleaned_ny.pkl`, `prediction_data_cleaned_tx.pkl`).

---

## Notes / legacy report
This repo originated from a BU CS542 common-task project and is **inspired by and copied from** the [LSTM-Automated-Trading-System](https://github.com/pranavgoyanka/LSTM-Automated-Trading-System) repo (Kalshi Weather Prediction Common Task for BU's CS542 Spring 2024), including LSTM models and data pipelines. Historical trade logs and report screenshots remain in the repo (e.g. `Kalshi-Recent-Activity-Pranav.csv` and images under `CS542 Common Task Report .../`).

---

## Optimization roadmap (accuracy → confidence → sizing)

The goal is not just to “predict a temperature”, but to quantify **how much we trust** a forecast and translate that into **position sizing**.

### Forecaster learning improvements
See `ForecasterLearningImprovements.md` for a concrete roadmap to improve long-run forecaster trust, including better scoring, regime adaptation, uncertainty calibration, and safe rollout.

### 1) Build a daily evaluation dataset
For each city + trade date, store:
- **Predictions**: `tmax_lstm`, `tmax_forecast` (per-provider), `tmax_blend`
- **Market context** (optional but useful): market midprice / orderbook for the chosen bucket
- **Resolved outcome**: settlement temperature (from Kalshi settlement fields or directly from the NWS CLI report used for settlement)

### Recommended schedule
- **Daily (trading run)**: `python run_daily.py --trade-date YYYY-MM-DD --env demo|prod [--send-orders]`
  - runs predictions, voting consensus, guardrails, orderbook-aware trade selection, and (optionally) submits orders.
- **Nightly (best-effort calibration)**: `python calibrate_sources.py --trade-date YYYY-MM-DD --window-days 14`
  - writes `Data/source_performance.csv` and updates `Data/weights.json` when NWS CLI truth is available.
- **Nightly (settlement + metrics)**: `python settle_eval.py --trade-date YYYY-MM-DD` and `python daily_metrics.py --as-of-date YYYY-MM-DD`
  - backfills settlement + realized PnL into `Data/eval_history.csv` and appends rollups to `Data/daily_metrics.csv`.

### 2) Scoring metrics (per city, season, horizon)
- **Temperature error**: MAE / RMSE vs resolved temperature
- **Bucket accuracy**: whether the chosen Kalshi interval contained the resolved temperature
- **Probabilistic scoring (future)**: if we turn forecasts into bucket probabilities, use Brier score / log loss

### 3) Sizing / “when to trade”
- Convert predicted temperature distribution → **bucket probabilities**
  - simplest: assume normal distribution around forecast mean with sigma derived from provider disagreement + historical residuals
- Combine with market prices to compute **expected value (EV)** per contract
- Trade only when EV clears a threshold, and size by:
  - capped Kelly fraction, or
  - a simpler “confidence ladder” (bigger size when sigma is small and edge is large)

### 4) RL-style allocation (later)
Once we have enough logged episodes (features → action → PnL), we can consider:
- contextual bandits for “which model/mode to trust today”
- constrained RL for position sizing (risk limits, max loss, liquidity-aware execution)

---

## References

- [LSTM-Automated-Trading-System](https://github.com/pranavgoyanka/LSTM-Automated-Trading-System) — Kalshi Weather Prediction Common Task for BU's CS542 Spring 2024 (inspiration for LSTM models and data pipeline)
- [Keras Documentation](https://keras.io/guides/)
- [Predicting Temperature of Major Cities Using Machine Learning and Deep Learning](https://arxiv.org/abs/2309.13330)