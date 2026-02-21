## The Ultimate Kalshi Weather Market Prediction Engine

Weather Trader is a weather-intelligence + execution system for Kalshi daily high-temperature markets.

It does more than predict a number. It tries to answer:
**which forecaster is right, for this city, in this regime, and why** - then sizes and executes market decisions with guardrails.

![Next trade view: prediction, confidence, source set, and weights.](Forecasts.png)

### What makes it different

- **Forecast selection, not just forecasting**: multiple providers are scored continuously against realized truth and reweighted nightly.
- **Reasoned confidence**: confidence is derived from disagreement among reliable sources and weighted source quality, not a single model probability.
- **Context-aware mode selection**: optional contextual bandit learns when to trust forecast consensus vs blend vs LSTM.
- **Execution-grade pipeline**: intraday refresh, local-time trade gates, EV-aware market selection, budgeting, idempotency, and settlement feedback.
- **Operational visibility**: live dashboard + analytics + notifications so you can see what the bot believes and why.

### Inspiration and origin

Inspired by the [LSTM-Automated-Trading-System](https://github.com/pranavgoyanka/LSTM-Automated-Trading-System) repo (Kalshi Weather Prediction Common Task, BU CS542 Spring 2024). This repo reuses core dataset/model foundations and extends them with provider blending, calibration, intraday updates, contextual-bandit experimentation, scheduling, and production-oriented observability.

---

## Quick start

Use Docker. From the repo root:

1. **Secrets:** Copy `.env.example` → `.env` and fill in values. Minimum for trading: `KALSHI_API_KEY_ID`, `KALSHI_PRIVATE_KEY_PATH`, `KALSHI_ENV`. Mount your Kalshi private key (see [Docker setup](documentation/docker_setup.md)).
2. **Run:**
   ```bash
   docker compose up -d --build
   ```
3. **Dashboard:** Open **http://localhost:8080** for the web UI (cron, trades, and dashboard all run in the container).

> [!IMPORTANT]
> **Next:** [Docker setup](documentation/docker_setup.md) (mount key, logs, schedule) · [Operational runbook](documentation/operational_runbook.md) (budget, live trading, commands) · [Environment variables](documentation/environment_variables.md)

Never commit `.env` or private keys. See [SECURITY.md](SECURITY.md) if keys were exposed.

**Without Docker:** Install deps with `pip install -r requirements.txt` (e.g. in a venv), configure `.env`, then run scripts manually (e.g. `python scripts/web_dashboard_api.py` for the dashboard). Cron and scheduling are up to you.

---

## Cities / coordinates

| City               | Code | Lat/Lon             |
| ------------------ | ---- | ------------------- |
| NYC (Central Park) | `ny` | 40.79736, -73.97785 |
| Chicago (Midway)   | `il` | 41.78701, -87.77166 |
| Austin (Bergstrom) | `tx` | 30.14440, -97.66876 |
| Miami              | `fl` | 25.77380, -80.19360 |

## Features

- **Forecast intelligence**
  - Multi-source weather forecasts: Open-Meteo, Visual Crossing, Tomorrow.io, WeatherAPI, OpenWeatherMap, Pirate Weather, NWS (+ optional LSTM signal).
  - Nightly calibration updates per-source MAE and writes learned weights to `Data/weights.json`.
  - Consensus prediction is MAE-weighted, with source-level explainability (`sources_used`, `weights_used`, spread, confidence).

- **Contextual learning (optional)**
  - Contextual-bandit mode (`off`/`shadow`/`canary`) can select among `forecast`/`blend`/`lstm`.
  - Context features include sky/condition votes, cloud cover, spread, provider count, and city/date features.
  - Full decision and reward telemetry is logged for post-settlement learning.

- **Trading and risk pipeline**
  - Intraday pulse refreshes forecasts and writes `predictions_latest.csv`.
  - Trade windows run at city-local 13:00 (ny/fl at 13:00 ET; il/tx at 14:00 ET).
  - EV-aware bucket selection using live orderbook snapshots, with confidence/spread guardrails and budget controls.
  - Idempotent execution and nightly settlement rollups close the loop.

- **Dashboard + analytics + notifications**
  - **Web dashboard (`/`)**: live observations, projected highs, next-trade table, positions, and risk/sell advisor.
  - **Analytics page (`/analytics`)**: source MAE, projection-vs-actual performance, lock-in timing analysis, contextual-bandit performance.
  - **Desktop notifications (browser permission-based)**: urgent risk and at-risk bracket opportunities.
  - **Terminal dashboard (TUI)**: operational snapshot directly in shell.

## Measurements (what we track)

| Artifact                                                                                                    | Purpose                                                |
| ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| `Data/source_performance.csv`                                                                               | Per-source prediction error vs NWS actual              |
| `Data/daily_metrics.csv`                                                                                    | Rollups for allocation and scoring                     |
| `Data/eval_history.csv`                                                                                     | Per-trade outcome and market state                     |
| `Data/city_metadata.json`                                                                                   | Per-city historical MAE (used for σ)                   |
| `Data/context_features_history.csv`, `Data/bandit_decisions_history.csv`, `Data/bandit_rewards_history.csv` | Contextual-bandit telemetry and settled reward updates |

> [!NOTE]
> **Schemas and key fields:** [Data reference](documentation/data_reference.md)

---

## Documentation

Visit these dedicated pages for full details — each link opens a dedicated doc with full content.

> [!NOTE]
> **Changelog** — **[CHANGELOG.md](CHANGELOG.md)** — Rough history of major changes (from git).

> [!IMPORTANT]
> **Operational runbook**
> **[→ Open Operational runbook](documentation/operational_runbook.md)** — Autonomous operation, one-time setup, budget and live trading, idempotency, logs, generate predictions, dry-run, place orders.

> [!IMPORTANT]
> **Docker setup**
> **[→ Open Docker setup](documentation/docker_setup.md)** — Prepare `.env`, mount Kalshi key, run container, logs, data persistence, schedule/timezone, 13:00 local gate.

> [!TIP]
> **Environment variables**
> **[→ Open Environment variables](documentation/environment_variables.md)** — Full table of required and optional env vars and secrets.

> [!TIP]
> **Dashboard**
> **[→ Open Dashboard](documentation/dashboard.md)** — Web UI and TUI: how to run, pages, observations.

> [!TIP]
> **Data flow**
> **[→ Open Data flow](documentation/data_flow.md)** — End-to-end: ingestion, cleaning, prediction modes, writing predictions, provider limits, weights/consensus, Kalshi budgeting and sigma.

> [!TIP]
> **Kalshi markets**
> **[→ Open Kalshi markets](documentation/kalshi_markets.md)** — Series tickers, NWS resolution, contract selection, authentication, dry-run.

> [!NOTE]
> **LSTM models**
> **[→ Open LSTM models](documentation/lstm_models.md)** — Training, input window, features, preprocessing.

> [!NOTE]
> **Mathematical foundations**
> **[→ Open Mathematical foundations](documentation/mathematical_foundations.md)** — Weights, sigma, probability, EV.

> [!NOTE]
> **System architecture**
> **[→ Open System architecture](documentation/system_architecture.md)**

> [!NOTE]
> **Data reference**
> **[→ Open Data reference](documentation/data_reference.md)** — Data files and schemas.

> [!NOTE]
> **Audit report**
> **[→ Open Audit report](documentation/audit_results.md)** — Risks and strengths.

> [!NOTE]
> **Improvement roadmap**
> **[→ Open Improvement roadmap](documentation/improvement_roadmap.md)**

> [!CAUTION]
> **Secrets and key rotation**
> **[→ Open SECURITY.md](SECURITY.md)** — Purging history, rotating keys, deleting cache files.

---

## Notes / legacy

This repo originated from a BU CS542 common-task project. Historical trade logs and report screenshots remain (e.g. `Kalshi-Recent-Activity-Pranav.csv`, `CS542 Common Task Report .../`).

> [!NOTE]
> **Optimization and accuracy:** [Improvement roadmap](documentation/improvement_roadmap.md) · `ForecasterLearningImprovements.md`

---

## References

- [LSTM-Automated-Trading-System](https://github.com/pranavgoyanka/LSTM-Automated-Trading-System) — Kalshi Weather Prediction Common Task, BU CS542 Spring 2024
- [Keras Documentation](https://keras.io/guides/)
- [Predicting Temperature of Major Cities Using Machine Learning and Deep Learning](https://arxiv.org/abs/2309.13330)
