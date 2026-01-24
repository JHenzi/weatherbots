## Weather-trader (operational)

This repo predicts **daily maximum temperature** for 4 locations and maps the prediction into Kalshi “high temperature” markets for automated (or dry-run) trading.

### Cities / coordinates
- **NYC (Central Park / Belvedere Castle)**: `40.79736,-73.97785` (`ny`)
- **Chicago (Midway Intl)**: `41.78701,-87.77166` (`il`)
- **Austin (Bergstrom Intl)**: `30.14440,-97.66876` (`tx`)
- **Miami**: `25.77380,-80.19360` (`fl`)

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
  - `KALSHI_API_KEY_ID`
  - `KALSHI_PRIVATE_KEY_PATH` (points to your RSA PEM file, e.g. `gooony.txt`)
  - `KALSHI_ENV` (`demo` or `prod`)

Security note: `.env` and private-key files are gitignored.

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

### 3) Run it (one command)
From the repo root:

```bash
docker compose up -d --build
```

Note: the container runs jobs on the schedule in `ops/docker/crontab`. If you just started it, it may not run a job until the next scheduled time.

To trigger a one-off run immediately (manual):

```bash
docker exec weather-trader /bin/bash /app/scripts/run_trade.sh
docker exec weather-trader /bin/bash /app/scripts/run_calibrate.sh
```

Or enable “run once on startup” by adding these to `docker-compose.yml` environment:
- `WT_RUN_TRADE_ON_START=true`
- `WT_RUN_CALIBRATE_ON_START=true`

### 4) Logs and outputs
- Container logs: `docker logs -f weather-trader`
- Cron logs (persisted): `Data/logs/trade.cron.log`, `Data/logs/calibrate.cron.log`
- Settlement/metrics logs (persisted): `Data/logs/settle.cron.log`
- Decisions/trades/eval CSVs: `Data/decisions_history.csv`, `Data/trades_history.csv`, `Data/eval_history.csv`

### 5) Schedule / timezone
Cron times are defined in `ops/docker/crontab` (defaults: 3:30pm trade, 2:15am calibrate).
If you want cron to run in your local timezone, set `TZ` in `docker-compose.yml` (e.g. `America/New_York`).

---

## Data flow (end-to-end)

### 1) Historical + daily weather ingestion
The ingestion logic lives in `daily_prediction.py` (and historically in `data_fetcher_new.ipynb`).

**Sources**
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
Predictions are appended to a CSV (default: `predictions_final.csv`), with a header when the file is first created.

Current schema written by `daily_prediction.py`:
- `date` (the **trade date** / event date)
- `city` (`ny`, `il`, `tx`, `fl`)
- `tmax_predicted` (the value used for trading)
- `tmax_lstm` (nullable)
- `tmax_forecast` (nullable)
- `forecast_sources` (comma-separated list, e.g. `open-meteo,visual-crossing`)
- `forecast_sources` (comma-separated list, e.g. `open-meteo,visual-crossing,tomorrow,weatherapi`)

Recommended: use a fresh output per run/mode, e.g. `--predictions-csv Data/predictions_latest.csv`.

### Provider rate limits (important)
- **Tomorrow.io (free tier)**: 3 req/sec, 25 req/hour, 500 req/day.
  - `daily_prediction.py` uses a **1-hour on-disk cache** (`Data/tomorrow_cache`) and a simple throttle to help stay within free-tier limits.
  - Operationally: avoid repeatedly re-running forecast mode in a tight loop.
- **OpenWeatherMap**:
  - `daily_prediction.py` uses a **1-hour on-disk cache** (`Data/openweathermap_cache`) to reduce repeat calls.
- **Pirate Weather**:
  - `daily_prediction.py` uses a **1-hour on-disk cache** (`Data/pirateweather_cache`) to reduce repeat calls.

### 5) Kalshi mapping + (dry-run) trading
Trading logic is in `kalshi_trader.py`.

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
By default the trader does **not** place orders; it prints what it *would* submit. It will only place orders if you pass `--send-orders`.

---

## Operational runbook

### Autonomous operation (macOS, recommended)

Goal: you do a **one-time setup**, then the system runs on its own:
- **Daily trade job**: runs `run_daily.py` for *tomorrow’s* markets with a **$50 total cap** (default), and logs what it did.
- **Nightly calibration job**: runs `calibrate_sources.py` for *yesterday* to update weights when NWS CLI truth is available.

#### One-time setup

1) Ensure `.env` has the required values:
- **Kalshi (required)**: `KALSHI_API_KEY_ID`, `KALSHI_PRIVATE_KEY_PATH`, `KALSHI_ENV`
- **weather.gov forecasts (recommended)**: `NWS_USER_AGENT=weather-trader (contact: you@example.com)`
- Optional: provider keys for forecasts (`TOMORROW`, `WEATHERAPI`, `PIRATE_WEATHER_API_KEY`, `VISUAL_CROSSING_API_KEY`)

2) Install dependencies once:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

3) Install the LaunchAgents (runs automatically thereafter):

```bash
mkdir -p ~/Library/LaunchAgents
cp ops/launchd/com.weathertrader.*.plist ~/Library/LaunchAgents/

# (Re)load
launchctl unload -w ~/Library/LaunchAgents/com.weathertrader.trade.plist 2>/dev/null || true
launchctl unload -w ~/Library/LaunchAgents/com.weathertrader.calibrate.plist 2>/dev/null || true
launchctl load -w ~/Library/LaunchAgents/com.weathertrader.trade.plist
launchctl load -w ~/Library/LaunchAgents/com.weathertrader.calibrate.plist
```

#### Controlling budget and live trading

The LaunchAgents call wrapper scripts in `scripts/` that support environment variables:
- **`WT_DAILY_BUDGET`**: total daily budget in dollars (default `50`)
- **`WT_ENV`**: `demo` or `prod` (default `demo`)
- **`WT_SEND_ORDERS`**: `true` to actually place orders (default: **dry-run**)

To enable live trading, edit `~/Library/LaunchAgents/com.weathertrader.trade.plist` and set:
- `WT_ENV=prod`
- `WT_SEND_ORDERS=true`
- (optional) `WT_DAILY_BUDGET=50`

Then reload the trade agent:

```bash
launchctl unload -w ~/Library/LaunchAgents/com.weathertrader.trade.plist
launchctl load -w ~/Library/LaunchAgents/com.weathertrader.trade.plist
```

#### Where to look for logs
- `Data/logs/trade.out.log` and `Data/logs/trade.err.log`
- `Data/logs/calibrate.out.log` and `Data/logs/calibrate.err.log`
- Decisions/trades/eval CSVs in `Data/`:
  - `Data/decisions_history.csv`
  - `Data/trades_history.csv`
  - `Data/eval_history.csv`

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
python kalshi_trader.py --env demo --trade-date 2026-01-23 --predictions-csv predictions_final.csv
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
- `--max-contracts-per-order` (default 25)
- `--sigma-floor` (default 2.0) and `--sigma-mult` (default 1.0)

### Place orders (demo first)

```bash
python kalshi_trader.py --env demo --trade-date 2026-01-23 --send-orders
```

---

## LSTM model details (historical)
Training is in `data_lstm.ipynb`:
- Separate model per city (saved as `Data/model_<city>.keras`)
- Input window: `time_steps = 10` days
- Features used:
  - `day_of_year`, `tmax`, `tmin`, `prec`, `humi`
- Preprocessing:
  - `StandardScaler` is fit on the full feature matrix, and inference uses the same feature set
  - Model predicts the *scaled* `tmax`; a dummy feature vector is inverse-transformed to recover °F

---

## Notes / legacy report
This repo originated from a BU CS542 common-task project. Historical trade logs and report screenshots remain in the repo (e.g. `Kalshi-Recent-Activity-Pranav.csv` and images under `CS542 Common Task Report .../`).

---

## Optimization roadmap (accuracy → confidence → sizing)

The goal is not just to “predict a temperature”, but to quantify **how much we trust** a forecast and translate that into **position sizing**.

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