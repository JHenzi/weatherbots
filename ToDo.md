## Weather → prediction → Kalshi trading (Operational ToDo)

Goal: **one monolithic daily process** that generates *per-source* forecasts + LSTM, computes a **weighted voting consensus** (learned from recent truth), converts that into Kalshi bucket probabilities + edge, sizes trades, and (optionally) submits orders — then logs outcomes and updates source weights nightly.

Bootstrap reality: we **cannot assume truth/scoring exists on day 1**. The bot must trade with:
- **uniform weights** across forecast providers (and optionally exclude LSTM from voting at first),
- a **spread/confidence guardrail** (abort when sources disagree),
- and then **start learning weights** automatically once NWS CLI truth becomes available.

---

## Recently completed (remove from backlog)

- [x] **Truth fetch robustness**: NWS CLI version scanning + force-refresh + better diagnostics (`truth_engine.py`)
- [x] **Cron retry window** for late-issued CLIs (extra `run_calibrate.sh`/`run_settle.sh` runs)
- [x] **Provider weights learned immediately**: calibration window is inclusive of the newly-scored day (`calibrate_sources.py`)
- [x] **Intraday forecast visibility**: `intraday_pulse.py` supports `--print` and `--no-write`
- [x] **Docs**: in-place weights/sigma math documented + new design docs for Kelly/execution/allocation + fallback plan

---

## Priority backlog (WSJF-style)

Scoring guide:
- **BV** (business value): improves capital growth / decision quality
- **TC** (time criticality): unblocks reliable daily ops quickly
- **RR/OE** (risk reduction / opportunity enablement): reduces blow-ups / enables later features
- **Effort**: rough days of work
- **WSJF**: \((BV+TC+RR/OE) / Effort\) — higher first

| Priority | Feature | BV | TC | RR/OE | Effort | WSJF | Notes |
| --- | --- | ---:| ---:| ---:| ---:| ---:| --- |
| P0 | Liquidity-aware sizing (cap to depth + effective ask) | 8 | 8 | 9 | 1 | 25.0 | Turn current warnings into actual caps; foundation for Kelly |
| P0 | Fractional Kelly sizing (flagged, dry-run first) | 10 | 7 | 8 | 1 | 25.0 | Use \(p\) vs price for stake; keep existing hard caps |
| P0 | “Best bet” fallback when 0 trades (flagged) | 7 | 8 | 6 | 0.5 | 42.0 | Ensure at least one small position when desired |
| P1 | Probability calibration (shrinkage/Platt) + Brier logging | 9 | 6 | 8 | 2 | 11.5 | Calibrate \(p\) before trusting Kelly more |
| P1 | City allocation upgrade using win-rate + PnL/volatility | 8 | 5 | 7 | 2 | 10.0 | Reduce exposure to “bad regime” cities (e.g., Miami) |
| P2 | Order slicing / repost policy (optional) | 6 | 3 | 6 | 2 | 7.5 | Improves fills without paying up |
| P2 | Bucket sweep portfolio (multi-bucket) | 6 | 2 | 4 | 2 | 6.0 | Hedge around mu; more complex accounting |
| P3 | Contextual bandit/RL for mode selection | 5 | 1 | 5 | 5 | 2.2 | Only after enough history |

---

## “One feature per day” plan (next 7 days)

Day 1 (Execution): implement liquidity caps in `kalshi_trader.py`
- cap size by best-level depth (and optionally allow 2nd level if impact < threshold)
- log `effective_ask`, `liquidity_capped_count`, `impact_pct` into `eval_history.csv`

Day 2 (Sizing): fractional Kelly (default off)
- add flags: `--sizing-mode budget|kelly`, `--kelly-fraction` (default 0.25), `--fallback-best-bet`
- convert calibrated \(p\) (initially `p_model`) + `yes_ask` into a target stake, then to `count`

Day 3 (Operational): best-bet fallback (default off)
- if no trades pass filters, pick “best bet” by highest EV (or most confident) and place a tiny capped order
- strict liquidity cap (best level only), strict size cap (e.g. \$1–\$5)

Day 4 (Learning): probability calibration + logging
- compute Brier score on settled trades
- implement shrinkage calibration (`p_cal = λ p_model + (1-λ) hit_rate_city`)
- log both `p_model` and `p_cal`

Day 5 (Allocation): incorporate win-rate + PnL volatility
- extend `daily_metrics.py` to emit per-city rolling PnL mean/std and Sharpe-like score
- use this in city budget allocation (still respecting min/max city fractions)

Day 6 (Execution): optional order slicing (dry-run)
- split large orders into small child orders when edge is high, but book is thin
- add a “no chase” default (place once, do not reprice)

Day 7 (Portfolio): bucket sweep (optional)
- distribute budget across adjacent buckets with EV>0
- add logging fields so we can evaluate sweep vs single-bucket

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
- [x] Lookback/learning loop runs nightly in Docker:
  - [x] per-source error log: `Data/source_performance.csv`
  - [x] learned weights: `Data/weights.json` + drift history `Data/weights_history.csv`
  - [x] settle + realized PnL backfill: `settle_eval.py` → `Data/eval_history.csv`
  - [x] daily rollups: `daily_metrics.py` → `Data/daily_metrics.csv`
- [ ] Next: liquidity-aware sizing, fractional Kelly sizing, best-bet fallback, probability calibration, improved city allocation
