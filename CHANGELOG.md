# Changelog

A rough history of major changes, derived from git. For full details see the [documentation](documentation/) and git history.

---

## 2026

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
