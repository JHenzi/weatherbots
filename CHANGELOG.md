# Changelog

A rough history of major changes, derived from git. For full details see the [documentation](documentation/) and git history.

---

## 2026

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
