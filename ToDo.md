## Weather → prediction → Kalshi trading (Operational ToDo)

Goal: **one monolithic daily process** that generates predictions, computes edge vs Kalshi markets, sizes trades, and (optionally) submits orders — then logs outcomes and learns from settlement.

### The monolithic daily run (one command)

- [ ] **Create `run_daily.py` (single entrypoint)**
  - [ ] Command shape: `python run_daily.py --trade-date YYYY-MM-DD --env demo|prod [--send-orders]`
  - [ ] Default safety: **demo + dry-run** unless `--send-orders`

When `run_daily.py` runs, it should execute these steps in order:

1) **Load config + sanity checks**
   - [x] `.env`/secrets are gitignored; private keys are gitignored
   - [ ] Validate required env vars:
     - [ ] `VISUAL_CROSSING_API_KEY`
     - [ ] `KALSHI_API_KEY_ID`
     - [ ] `KALSHI_PRIVATE_KEY_PATH`
     - [ ] `KALSHI_ENV` / `--env`
   - [ ] Validate `trade_date` (and decide the correct “observed window end date” = `trade_date - 1`)

2) **Fetch/refresh observed history (optional but required for LSTM mode)**
   - [x] `daily_prediction.py` supports a recent-window fetch and avoids hard-coded dates
   - [ ] In `run_daily.py`, add `--refresh-history` (default: off for speed)
     - [ ] If on: call `daily_prediction.py` without `--skip-fetch` to refresh `Data/prediction_data_cleaned_<city>.pkl`

3) **Generate predictions (choose a mode)**
   - [x] Implemented in `daily_prediction.py`: `--prediction-mode lstm|forecast|blend`
   - [ ] In `run_daily.py`, decide which mode is used for trading:
     - [ ] **forecast** (recommended baseline for trading)
     - [ ] **blend** (forecast + LSTM bias-corrector)
     - [ ] **lstm** (model-only; can diverge from forecast consensus)
   - [ ] Write outputs:
     - [ ] `Data/predictions_latest.csv` (overwrite, canonical “what we traded on”)
     - [ ] `Data/predictions_history.csv` (append-only, includes mode + sources + timestamp)

4) **Fetch Kalshi market data for `trade_date`**
   - [x] `kalshi_trader.py` supports signed requests and dry-run by default
   - [x] Confirmed city ↔ series tickers and settlement sources (NWS CLI)
   - [ ] Add orderbook fetch + mid/spread extraction for each candidate market bucket

5) **Compute probability, edge, and select trades**
   - [ ] **Consensus + uncertainty**:
     - [ ] Add at least one more forecast provider (e.g. PirateWeather) to estimate sigma via disagreement
     - [ ] Convert \(\mu,\sigma\) to bucket probabilities for each market subtitle
   - [ ] **Edge**:
     - [ ] Convert market price → implied probability
     - [ ] Edge = model_prob − market_prob
     - [ ] Filter: `edge >= min_edge` and `spread <= max_spread`
   - [ ] **Trade selection**:
     - [ ] Baseline: pick best-EV single bucket per city
     - [ ] Optional: “bucket sweep” portfolio (multiple buckets/thresholds with bounded downside)

6) **Size orders + liquidity/risk filters**
   - [ ] **Liquidity filter** (from `RLSYS.md`):
     - [ ] Estimate price impact from orderbook depth
     - [ ] If buying ~$50 moves price by > 2¢ → reduce size or skip
   - [ ] **Sizing**:
     - [ ] Start: fixed size + strict caps (max contracts per city/day)
     - [ ] Next: fractional Kelly (capped) using edge + variance
   - [ ] **Risk controls**:
     - [ ] Daily max loss cap
     - [ ] Per-city max exposure cap
     - [ ] “No trade” when forecast sources missing or sigma too large

7) **Execute orders (or dry-run)**
   - [x] Dry-run prints intent; only submits when `--send-orders`
   - [ ] Write a trade intent log even on dry-run:
     - [ ] `Data/trades_history.csv` (timestamp, city, event_ticker, market_ticker, side, price, qty, mode, edge, sigma)

8) **Post-trade monitoring (optional)**
   - [ ] Pull open orders / fills / positions and log
   - [ ] Track slippage vs expected entry

9) **Settlement truth + scoring (learn daily)**
   - [ ] Create `truth_engine.py`:
     - [ ] Fetch settlement truth from the same NWS CLI used by Kalshi
     - [ ] Parse “Temperature → Yesterday/Today → Observed Value → Maximum”
   - [ ] Create `Data/eval_history.csv` keyed by (city, trade_date):
     - [ ] store: predictions, sigma, chosen bucket(s), market prices at decision time, realized settlement temperature, realized PnL
   - [ ] Compute daily metrics:
     - [ ] MAE/RMSE per city
     - [ ] bucket hit-rate
     - [ ] provider “trust scores” (rolling windows)

10) **Optimization / RL (after logs exist)**
   - RLSYS framing:
     - **State**: LSTM prediction, consensus forecast, uncertainty, orderbook, time-to-expiry, recent errors
     - **Reward**: (profit − fees) − (penalty × drawdown)
   - [ ] Create `rl_manager.py`
     - [ ] Learn provider trust weights and blend weights over time
     - [ ] Add “previous day error” features for calibration offsets
   - [ ] Start with contextual bandit (simpler than full RL):
     - [ ] Action = choose mode/weights + sizing multiplier
     - [ ] Evaluate out-of-sample on logged history

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
- **Meteostat**: historical station data
- **NOAA NCEI**: historical station data

### Current status (high-signal)
- [x] `daily_prediction.py` supports `trade_date` and `prediction-mode` (lstm/forecast/blend)
- [x] `kalshi_trader.py` supports signed Kalshi API requests and dry-run by default
- [ ] Missing: single `run_daily.py` orchestration + logging + EV/sizing + truth scoring
## Weather → prediction → Kalshi trading (Operational ToDo)

Goal: **predict daily max temperature** for Kalshi HIGH{CITY} markets, then **place trades** based on predicted temp range (with guardrails + dry-run).

## Action plan (build an automated trading system)

This plan is ordered to get to a reliable end-to-end bot quickly, then layer on confidence scoring + sizing, then RL/bandit optimization.

### Phase A — Reliable daily pipeline (must-have)
- [ ] **Single daily entrypoint**: create `run_daily.py`
  - [ ] Run: prediction generation → write `predictions_latest.csv` → trading dry-run → optional live submit
  - [ ] Flags: `--trade-date`, `--env demo|prod`, `--prediction-mode lstm|forecast|blend`, `--send-orders`
- [ ] **Stable output artifacts**
  - [ ] Stop appending to `predictions_final.csv` by default; write a “latest” file per run (avoid mixing modes/dates)
  - [ ] Keep historical logs in a separate file (append-only) with run metadata (mode, sources, timestamp)
- [ ] **Production-safe defaults**
  - [ ] Default to `demo` and dry-run unless explicitly `--send-orders`
  - [ ] Add max-loss and max-contract caps per city/day

### Phase B — “Truth Engine” + scoring (confidence)
- [ ] **Settlement truth fetcher (NWS CLI)**
  - [ ] Implement `truth_engine.py` to fetch the resolved high temperature from the NWS CLI pages used by Kalshi settlement:
    - [ ] NYC: `site=OKX`, `issuedby=NYC`
    - [ ] Chicago: `site=LOT`, `issuedby=MDW`
    - [ ] Austin: `site=EWX`, `issuedby=AUS`
    - [ ] Miami: `site=MFL`, `issuedby=MIA`
  - [ ] Parse “Temperature → Yesterday/Today → Observed Value → Maximum” per Kalshi contract terms
  - [ ] Store resolved `tmax_settle_f` per (city, date)
- [ ] **Evaluation log**
  - [ ] Create `Data/eval_history.csv` (or sqlite) keyed by (city, trade_date)
  - [ ] Store: provider forecasts, `tmax_lstm`, `tmax_blend`, `tmax_predicted`, chosen market ticker, order prices, and `tmax_settle_f`
- [ ] **Accuracy scoring**
  - [ ] MAE/RMSE per city and by month
  - [ ] Bucket hit rate (% of times the selected interval contains settlement)
  - [ ] Provider “trust score” over rolling windows (e.g., last 30/90 days)

### Phase C — Probability + EV + sizing (what to buy, how much)
- [ ] **Consensus + uncertainty model**
  - [ ] Add at least one more forecast source (e.g. PirateWeather) for redundancy
  - [ ] Compute mean + std across providers (uncertainty proxy)
  - [ ] Convert to probabilities for Kalshi buckets (normal approx around mean)
- [ ] **Kalshi bridge (edge calculation)**
  - [ ] Fetch orderbook for candidate markets
  - [ ] Compute implied probability from price (yes_price/100)
  - [ ] Compute edge = model_prob − market_prob and only trade if edge > threshold
- [ ] **Liquidity filter (from RLSYS.md)**
  - [ ] Before submitting, estimate price impact using orderbook depth
  - [ ] If buying ~$50 moves price by > 2¢, reduce size / skip
- [ ] **Position sizing**
  - [ ] Implement fractional Kelly sizing: stake ∝ edge / variance, with strict caps
  - [ ] Add drawdown-aware penalty (reduce size after losses)

### Phase D — “Bucket sweep” strategy (portfolio of buckets)
- [ ] Implement “sweep” execution:
  - [ ] For a predicted temp \(T\), buy multiple thresholds/buckets on one side when cheap (risk-defined net)
  - [ ] Optimize which buckets to include based on EV and correlation

### Phase E — RL / bandit (learn what to trust)
RLSYS framing:
- **State**: LSTM prediction, consensus forecast, std/uncertainty, orderbook prices/spreads, time to expiry, recent model errors
- **Reward**: (profit − fees) − (penalty × drawdown), with preference for closing at 10–20% profit vs holding to 0

Implementation steps:
- [ ] Create `rl_manager.py`
  - [ ] Maintain “trust scores” per provider/model using `eval_history` residuals
  - [ ] Learn an offset using previous-day errors (calibration feature): if yesterday was +2°F, adjust today
- [ ] Start with **contextual bandit** (simpler than full RL):
  - [ ] Action = choose prediction mode/weights and sizing multiplier
  - [ ] Train on historical logged outcomes, evaluate out-of-sample
- [ ] Later: constrained RL for sizing with risk limits and liquidity-aware execution

### Locations (lat/long)
- **NYC (Central Park / Belvedere Castle)**: `40.79736,-73.97785` (`ny`)
- **Chicago (Midway Intl)**: `41.78701,-87.77166` (`il`)
- **Austin (Bergstrom Intl)**: `30.14440,-97.66876` (`tx`)
- **Miami**: `25.77380,-80.19360` (`fl`)

### Data sources currently used by pipeline
- **Open-Meteo** (archive): max/min temp, sunshine duration, precipitation hours, wind speed
- **Visual Crossing** (needs API key): max/min temp, humidity, wind speed
- **Meteostat**: max/min temp, precipitation, snow, etc
- **NCEI NOAA**: max/min temp + many station variables

---

## ASAP: Make it run end-to-end (highest priority)

### 0) Secrets/config
- [ ] Ensure `.env` exists locally (gitignored) with:
  - [ ] `VISUAL_CROSSING_API_KEY=...`
  - [ ] `KALSHI_API_KEY_ID=...` (Key ID)
  - [ ] `KALSHI_PRIVATE_KEY_PATH=...` (path to `gooony.txt` / PEM)
  - [ ] `KALSHI_ENV=demo` (or `prod`)
- [x] Add `.env.example` (safe template, no secrets)
- [x] Add `.gitignore` rules for private key files (`gooony.txt`, `*.pem`, etc.)

### 1) Dependencies / environment
- [x] Add `requirements.txt` (or `pyproject.toml`) for the scripts (notebooks optional)
- [ ] Confirm runtime imports work: `tensorflow`, `kalshi_python_sync`, `cryptography`, `openmeteo_requests`, `requests_cache`, `retry_requests`, `meteostat`, etc.

### 2) Fix daily pipeline correctness
- [x] Remove hard-coded 2024 dates from `daily_prediction.py`
  - [ ] Introduce canonical **`trade_date`** (default: today)
  - [ ] Fetch latest observed day (usually `trade_date - 1`) and append to pickles
  - [ ] Predict **for `trade_date`** and write to a stable CSV (with header)
- [x] Fix `time_steps` consistency (training uses `10`; inference must match model input)
- [ ] Make cleaning robust to missing columns (don’t crash if one source fails)
- [x] Stop printing entire Visual Crossing response bodies (too noisy, can leak info)
- [x] Add `--prediction-mode` (`lstm` / `forecast` / `blend`) so we can trade on forecast consensus

### 3) Fix Kalshi trading to use API key auth (required)
- [x] Replace email/password auth in `kalshi_trader.py`
  - [x] Use API Key ID + RSA private key signing (per Kalshi docs)
  - [x] Default to dry-run (only sends orders with `--send-orders`)
  - [x] Confirm series tickers + resolution sources (NWS CLI) for NYC/MDW/AUS/MIA
- [ ] Add checks:
  - [ ] Prediction exists for `trade_date` & city
  - [ ] Event ticker exists (e.g. `HIGHNY-YYMONDD`)
  - [ ] Market subtitle parsing succeeds (ranges, “or below/above”)
  - [ ] Predicted temperature is finite + plausible

### 4) Operational runbook (one command)
- [ ] Add a single entrypoint like `run_daily.py`:
  - [ ] `python run_daily.py --trade-date 2026-01-23 --env demo --dry-run`
  - [ ] Runs: fetch → clean → predict → (optional) trade

---

## Near-term model improvement (after it runs)
- [ ] Ensemble “forecast providers” for `trade_date` (Open-Meteo forecast + Visual Crossing) and use model as bias-corrector
- [ ] Add uncertainty (std/quantiles across providers) to size/skip trades
- [ ] Backtest vs settlement station (calibration per city/season)

---

## Optimization / confidence / sizing (add next)
- [ ] Create `Data/eval_history.csv` (or sqlite) logging per (city, trade_date):
  - [ ] tmax_lstm, tmax_forecast (per-provider), tmax_blend, chosen_bucket
  - [ ] resolved_tmax (from NWS CLI or Kalshi settlement), bucket_hit boolean
  - [ ] optional: market prices / spreads used at decision time
- [ ] Implement resolver fetcher:
  - [ ] Option A: parse NWS CLI HTML/text directly from the settlement_source URL
  - [ ] Option B: use Kalshi settled event data (if available in API response)
- [ ] Add scoring + calibration report:
  - [ ] MAE/RMSE per city and by month
  - [ ] bucket accuracy (% correct interval)
  - [ ] residual distribution to estimate sigma for probability modeling
- [ ] Add sizing logic:
  - [ ] Compute EV vs market prices (requires orderbook fetch)
  - [ ] Trade only when EV > threshold
  - [ ] Position sizing via capped Kelly or a confidence ladder
- [ ] Explore RL-style allocation (later):
  - [ ] contextual bandit for choosing mode/weight daily
  - [ ] constrained RL for sizing with risk limits and liquidity-aware execution
