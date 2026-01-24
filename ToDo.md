## Weather → prediction → Kalshi trading (Operational ToDo)

Goal: **one monolithic daily process** that generates *per-source* forecasts + LSTM, computes a **weighted voting consensus** (learned from recent truth), converts that into Kalshi bucket probabilities + edge, sizes trades, and (optionally) submits orders — then logs outcomes and updates source weights nightly.

Bootstrap reality: we **cannot assume truth/scoring exists on day 1**. The bot must trade with:
- **uniform weights** across forecast providers (and optionally exclude LSTM from voting at first),
- a **spread/confidence guardrail** (abort when sources disagree),
- and then **start learning weights** automatically once NWS CLI truth becomes available.

### The monolithic daily run (one command)

- [x] **Create `run_daily.py` (single entrypoint)**
  - [x] Command shape: `python run_daily.py --trade-date YYYY-MM-DD --env demo|prod [--send-orders]`
  - [x] Default safety: **demo + dry-run** unless `--send-orders`
  - [x] Core flags: `--prediction-mode lstm|forecast|blend`, `--blend-forecast-weight`, `--refresh-history`
  - [x] New: `--retrain-lstm` (train models daily) and `--retrain-days-window` (e.g. last 365/730 days)

When `run_daily.py` runs, it should execute these steps in order:

1) **Load config + sanity checks**
   - [x] `.env`/secrets are gitignored; private keys are gitignored
   - [x] Validate env vars (fail fast on critical Kalshi config; warn on missing optional providers):
     - [x] `KALSHI_API_KEY_ID` (or aliases)
     - [x] `KALSHI_PRIVATE_KEY_PATH` exists
     - [x] `KALSHI_ENV` / `--env` is `demo|prod`
     - [x] Warn if missing `VISUAL_CROSSING_API_KEY` / `TOMORROW` / `WEATHERAPI` (Open-Meteo still works)
   - [x] Validate `trade_date` and compute `fetch_date = trade_date - 1`

2) **Refresh observed history (optional)**
   - [x] `daily_prediction.py` supports a recent-window fetch and avoids hard-coded dates
   - [x] `--refresh-history` triggers `daily_prediction.py` without `--skip-fetch` to refresh:
     - [x] `Data/prediction_data_cleaned_<city>.pkl`

3) **Train LSTM daily (new requirement)**
   - [x] **Add `train_models.py` (non-notebook training script)**
     - [x] Inputs: `Data/prediction_data_cleaned_<city>.pkl`
     - [x] Trains 1 model per city with `time_steps = 10`
     - [x] Features match inference: `day_of_year`, `tmax`, `tmin`, `prec`, `humi`
     - [x] Saves **versioned** models:
       - [x] `Data/models/<YYYYMMDD>/model_<city>.keras`
     - [x] Updates “current model” pointer safely:
       - [x] write/replace `Data/model_<city>.keras` **only after training + validation succeeds**
     - [x] Validation/rollback:
       - [x] Hold out last N days (e.g. 30) as validation
       - [x] If new model is worse than previous by threshold, keep previous model
       - [x] Log metrics to `Data/model_metrics.csv`
   - [x] `--retrain-lstm` in `run_daily.py` runs the above after refresh-history
   - [ ] In production: consider retraining as a **separate scheduled job** so trade execution is not delayed

4) **Generate predictions**
   - [x] `daily_prediction.py` supports `--prediction-mode lstm|forecast|blend`
   - [x] `run_daily.py` chooses the mode used for trading
   - [x] Write outputs:
     - [x] `Data/predictions_latest.csv` (overwrite, canonical “what we traded on”)
     - [x] `Data/predictions_history.csv` (append-only, includes mode + sources + timestamp)
  - [x] **VotingModel upgrade (required)**: stop collapsing sources into a single `tmax_forecast`
    - [x] Write per-source columns to predictions output (for each city/date):
      - [x] `tmax_open_meteo`
      - [x] `tmax_visual_crossing`
      - [x] `tmax_tomorrow`
      - [x] `tmax_weatherapi`
      - [x] `tmax_lstm`
    - [x] Keep `tmax_predicted` as the final “consensus” value produced by the voting system (computed in `run_daily.py`)
    - [x] Record `sources_used`, `spread_f`, `confidence_score`, `weights_used`

5) **Fetch Kalshi market data for `trade_date`**
   - [x] `kalshi_trader.py` supports signed requests and dry-run by default
   - [x] Confirmed city ↔ series tickers and settlement sources (NWS CLI)
   - [x] Fix bucket selection parsing (avoid wrong “or below” fallbacks)
  - [x] Add orderbook fetch + mid/spread extraction for each candidate market bucket

6) **Compute probability, edge, and select trades**
  - [x] **Weighted Voting consensus (VotingModel.md)**
    - [x] Bootstrap: if `Data/weights.json` doesn’t exist yet, use **uniform weights** across available forecast providers
    - [x] Create `Data/source_performance.csv` logging (written when truth exists)
      - [x] `date,city,source_name,predicted_tmax,actual_tmax,absolute_error`
     - [x] Implement `calculate_weights()` (`calibrate_sources.py`):
       - [x] Look at last **N days** (configurable) of errors per city+source
       - [x] Weight formula: \(w_{source} = 1 / MAE_{source}^2\)
       - [x] Normalize weights to sum to 1 (per city)
       - [x] Persist to `Data/weights.json` (per city, per source, with timestamp + window)
     - [x] Compute **weighted consensus temperature** (in `run_daily.py` voting postprocess)
  - [x] **Confidence guardrail (VotingModel.md)**
    - [x] Compute spread (std dev) across sources (forecast providers by default)
    - [x] Abort trades when spread > 3.0°F; require `confidence_score >= 0.5`
  - [x] **Fair price / edge**
    - [x] Convert consensus + sigma into bucket probabilities (normal approximation)
    - [x] Fetch orderbook / market quotes and compute implied probability
    - [x] Edge = model_prob − market_prob; EV (cents) = \(100p - ask\)
  - [x] **Trade selection**
    - [x] baseline: best-EV single bucket per city (YES-only)
     - [ ] optional: bucket sweep portfolio

7) **Size orders + liquidity/risk filters**
  - [x] Liquidity filter (uses orderbook ask depth when available; conservative fallback when not)
  - [x] Sizing (conservative defaults + CLI flags):
    - [x] start fixed size + caps
    - [ ] then fractional Kelly (capped)
  - [x] Risk controls:
    - [x] per-city cap + daily total cap
    - [x] “no trade” when missing data / count=0 after caps

8) **Execute orders (or dry-run)**
   - [x] Dry-run prints intent; only submits when `--send-orders`
   - [x] Log decisions even on dry-run:
     - [x] `Data/trades_history.csv` (timestamp, city, event_ticker, market_ticker, side, price, qty, mode, edge, sigma)
  - [x] **VotingModel guardrail change**:
    - [x] If `confidence_score < 0.5` or spread > 3.0°F: do **not** place orders
    - [x] Log a separate `Data/decisions_history.csv` for “skips” (reason, spread, confidence), and only log to `trades_history.csv` when a trade would be/was placed

9) **Settlement truth + scoring (learn daily)**
  - [x] Create `truth_engine.py`:
    - [x] Fetch settlement truth from NWS CLI and parse “TEMPERATURE (F) → YESTERDAY → MAXIMUM”
  - [x] Nightly “calibration” job (VotingModel.md) scaffold:
    - [x] `calibrate_sources.py` writes rows to `Data/source_performance.csv` when CLI truth exists
    - [x] `calibrate_sources.py` recomputes `Data/weights.json` (rolling window) when enough history exists
    - [x] Bootstrap behavior: if CLI truth isn’t available yet, calibration exits cleanly without blocking the bot
  - [x] Create `Data/eval_history.csv` keyed by (city, trade_date):
    - [x] store per-source forecasts, consensus, sigma, chosen bucket, market prices, model_prob, market_prob, edge/EV, count
    - [x] add settlement temperature + realized PnL (nightly backfill via `settle_eval.py`)
  - [x] Daily metrics (nightly):
    - [x] MAE/RMSE per city per source (`daily_metrics.py`)
    - [x] bucket hit-rate (`daily_metrics.py`)
    - [x] weight drift history (`Data/weights_history.csv` appended by `calibrate_sources.py`)

10) **Optimization / RL (after logs exist)**
   - [ ] Implement `rl_manager.py` on top of the Voting Model:
     - [ ] State includes: per-source errors, current weights, spread/confidence, orderbook/liquidity, time-to-expiry
     - [ ] Actions include: adjust weights, adjust confidence threshold, sizing multiplier, and sweep selection
     - [ ] Reward = (profit − fees) − (penalty × drawdown)

---

### Cities (lat/long) and settlement stations (confirmed)
- **NYC (`ny`)**: `40.79736,-73.97785` → NWS CLI `issuedby=NYC` (OKX)
- **Chicago (`il`)**: `41.78701,-87.77166` → NWS CLI `issuedby=MDW` (LOT)
- **Austin (`tx`)**: `30.14440,-97.66876` → NWS CLI `issuedby=AUS` (EWX)
- **Miami (`fl`)**: `25.77380,-80.19360` → NWS CLI `issuedby=MIA` (MFL)

### Data sources currently used
- **Open-Meteo**: historical + forecast
- **Visual Crossing**: historical + forecast (requires key)
- **Tomorrow.io**: forecast (requires key; free-tier limits: 3 req/sec, 25 req/hour, 500 req/day)
- **WeatherAPI.com**: forecast (requires key)
- **Pirate Weather**: forecast (requires key)
- **weather.gov (NWS)**: forecast (no key; requires `NWS_USER_AGENT`)
- **OpenWeatherMap**: forecast (requires key; can be disabled via `DISABLE_OPENWEATHERMAP=true`)
- **Meteostat**: historical station data
- **NOAA NCEI**: historical station data

### Current status (high-signal)
- [x] `daily_prediction.py` supports `trade_date` and `prediction-mode` (lstm/forecast/blend)
- [x] `kalshi_trader.py` supports signed Kalshi API requests and dry-run by default
- [ ] Missing: optional bucket sweep portfolio, fractional Kelly sizing, RL
