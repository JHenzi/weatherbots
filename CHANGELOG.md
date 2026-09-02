# Changelog

A rough history of major changes, derived from git. For full details see the [documentation](documentation/) and git history.

---

## 2026

### 2026-09 — Label-leakage fix: provider grading moved to decision time

The adaptive source weighting was being trained on a leaked label, which inverted the
provider ranking and made the learned ensemble measurably worse than a plain average.

- **Fixed** `intraday_pulse.forecast_tmax_weather_gov()` returned the **overnight low** as the
  daily high whenever no `isDaytime` period matched the target date. NWS drops the daytime
  period once it has passed, so after ~19:00 local only "Tonight" remained and its
  temperature was used as the forecast. Measured signed bias −15.01°F, under-predicting on
  99% of 248 graded rows (Miami reported 81–85°F on days settling 91–100°F). Now returns
  `None`, so the ensemble drops the provider for that run.
- **Fixed** `calibrate_sources._load_predictions_for_date()` kept the **last** row per
  city/date. On 60 of 64 recent days that row was written at 23:00, after the day's high
  had already occurred, so providers were scored on convergence to a known outcome rather
  than on forecast skill. Now selects the snapshot nearest `WT_DECISION_HOUR` (default
  `9`), preferring the **same-day** row over the day-ahead forecast for the same date —
  both exist in `predictions_history.csv` and both have an 09:00 entry.
- **Added** `scripts/regrade_source_performance.py` — rebuilds `Data/source_performance.csv`
  from the 09:00 same-day snapshot in `intraday_forecasts.csv`, graded against the settled
  actuals already on file. Ground truth is reused, never recomputed. The seven-month
  history was re-graded in place (8,129 → 6,864 rows); the original is retained as
  `source_performance.csv.leaked.bak`.
- **Changed** Provider MAEs collapse from a distorted 0.86–14.85°F range into a realistic
  2.28–3.73°F band. `weather.gov` moves from ~0% ensemble weight to 25% in Chicago and 24%
  in Austin. Walk-forward on 432 out-of-sample city-days: consensus MAE **2.170°F → 1.654°F**
  (equal-weighting scores 2.041°F, i.e. the pre-fix adaptive scheme was worse than a plain
  average).
- **Changed** README accuracy figures and the "who to trust" leaderboard were rebuilt — the
  previously published per-city MAEs (0.67–0.87°F) and the "trust Visual Crossing above
  everything" guidance were artifacts of the leaked label. True decision-time consensus MAE
  is 2.27°F.
- **Added** `adaptive_ensemble.py`, `decision_policy.py`, `feedback_loop.py` and
  `tests/test_adaptive_engine.py` from the architecture audit. `feedback_loop.py` fits an
  isotonic calibration of `model_prob_yes` against realized outcomes (+4.7% Brier
  out-of-sample) and gates parameter auto-tuning behind a PnL significance test — realized
  edge is currently +$87 over 271 trades at t = 0.79, i.e. not distinguishable from zero.
- **Note** `adaptive_ensemble.py` is **not wired in**. Measured against the corrected
  baseline it is 4.4% *worse* (1.727°F vs 1.654°F), and a 72-config sweep could not beat
  the existing flat 7-day `1/MAE²` scheme. The gain was the label fix, not the machinery.

### 2026-03 — 10 AM morning strategy with observation-based exits

- **Changed** Morning Kelly trade moved from 07:15 to **10:00 ET** (crontab). Empirical intraday MAE at 10:00 (~2.4°F overall, FL ~3.6°F) is meaningfully better than the 07:00 baseline (~2.6–3.8°F), and order-book liquidity is still healthy before the afternoon settlement window.
- **Changed** Exit manager now runs every 30 minutes (10:30, 11:00, 11:30, 12:00) instead of hourly from 09:00, giving tighter exit cadence after the 10:00 entry.
- **Added** `exit_manager.py` observation-based exit trigger: reads `observations_latest.json` on each check pass. If `projected_high > bucket_hi + 1.5°F` (too hot) or `projected_high < bucket_lo - 1.5°F` (too cold), cancels the resting profit-take sell and places an aggressive sell at `bid − 1¢` to cut the loss. If `projected_high` is within the bucket and `yes_bid ≥ 85% of target_exit_price`, places an immediate sell at bid to capture the win early. Threshold configurable via `--danger-threshold-f`.
- **Added** `mu_pred` and `sigma_pred` columns to `morning_entries.csv` — records the exact bias-corrected forecast used at buy time, independent of later intraday-pulse updates that overwrite `predictions_latest.csv`.
- **Added** `historical_MAE_morning` to `city_metadata.json` — computed by `update_city_metadata.py --morning-entries-csv` from morning entries joined against settled actuals. Kept separate from `historical_MAE` (1 PM consensus) so the two windows don't skew each other's sigma baselines.
- **Fixed** `observations_history.csv` schema mismatch: the file was written with an 8-column header but rows contained 11 columns (`observed_high_today`, `projected_high`, `time_temp_will_max` were appended without updating the header). `web_dashboard_api.py` now calls `_migrate_observations_csv()` on each write cycle, atomically rewriting the file with the correct header on first detection.

### 2026-03 — Condition-stratified bias correction

- **Changed** `update_city_metadata.py` now joins `source_performance.csv` with `context_features_history.csv` to compute per-city, per-condition-bucket signed bias corrections (`clear / mixed / precip / snow`), stored as `bias_correction_by_condition` in `city_metadata.json`. The flat `bias_correction_f` is retained as a fallback.
- **Changed** `bandit/modes.py` `compute_candidate_mode_predictions` now accepts `bias_correction_by_condition` and `condition_token`, applying the condition-specific correction for the `blend` action instead of a uniform offset. This fixes `blend` consistently losing to `forecast` because a flat +0.8–1.0°F correction was overcorrecting in clear conditions.
- **Changed** `intraday_pulse.py` loads and forwards condition-stratified corrections to the bandit mode computation.
- **Fixed** `morning_trader.py` was buying wrong temperature buckets because the flat bias overcorrected `mu`, shifting the Gaussian into an adjacent range. Morning trader now reads today's condition token from `context_features_history.csv` and applies the condition-stratified correction before scoring buckets.

### 2026-03 — Trade log visibility

- **Added** Recent trade-log summaries to the web dashboard, showing the latest per-city decisions for the last three trade dates.
- **Added** `scripts/trade_log_summary.py`, dashboard API support, and tests for aggregating executed, planned, and skipped trades.

### 2026-02 — Security, analytics, and contextual trading

- **Added** SECURITY guidance, safer `.env.example` / `.gitignore` defaults, and Docker scheduling updates for a more production-ready local setup.
- **Added** Forecast API limit documentation plus richer dashboard and analytics endpoints for operational visibility.
- **Added** PostgreSQL-backed telemetry support: schema updates, CSV migration tooling, and new reporting scripts.
- **Added** Contextual-bandit infrastructure for `forecast` / `blend` / `lstm` mode selection, including weather-context feature extraction, policy state, and nightly reward updates.
- **Fixed** Calibration and settlement ingestion to skip duplicate source-performance rows, parse flagged NWS maxima correctly, and cover both paths with tests.
- **Changed** Intraday confidence scoring, atomic bandit-state writes, city-metadata updates, and trade scripts to better handle concurrent jobs and city-local trade windows.

### 2026-01 — Documentation overhaul

- **Added** comprehensive system documentation (PR #1): operational runbook, Docker setup, environment variables, dashboard, data flow, Kalshi markets, LSTM models, mathematical foundations, system architecture, data reference, audit report, improvement roadmap.
- Documentation is linked from the main README with dedicated pages per topic.

---

## 2024

### 2024-06 — Repo cleanup and README

- **Changed** README updated for clarity.
- **Changed** Final code and repo cleanup.

### 2024-04 — Paths and prediction storage

- **Fixed** Paths and prediction storage (data and code layout).
- **Changed** Better paths for data and code.

### 2024-03 — Trading and daily pipeline

- **Added** Working Kalshi API trades with daily predictions.
- **Added** Daily data fetching scripts.
- **Changed** Better model prediction code and latest-data handling.

### 2024-03 — Data pipeline and sources

- **Added** Data cleaning for all cities.
- **Added** Data cleaning and regression notebook.
- **Added** Support for NCEI (NOAA) data.
- **Added** Support for MeteoStat.
- **Added** Support to merge all dataframes and pickle them.
- **Changed** Read from pre-created JSON files for Visual Crossing data.
- **Added** Data from Open-Meteo and Visual Crossing.

### 2024-03 — Early models and data

- **Added** Initial model experimentation (WIP).
- **Added** First working model (prototype).
- **Added** Data fetching and cleaning code.
- **Added** `.gitignore`.
- **Added** Initial commit.

---

*This changelog is maintained manually from git history; not every commit is listed.*
