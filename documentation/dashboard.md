# Dashboard (how-to)

## Web dashboard

Single process: background NWS observation fetch + FastAPI API + HTML dashboard. Serves live station data, next-trade predictions, Risk/Sell advisor, at-risk brackets, and positions.

**Use Docker (recommended):** The dashboard runs inside the container. After `docker compose up -d`, open **http://localhost:8080** (or your host’s IP if remote). No extra steps. To disable it, set `WT_RUN_DASHBOARD_ON_START=false` in the weather-trader service.

**Without Docker:** From repo root run `python scripts/web_dashboard_api.py`, then open http://localhost:8080.

### Pages

- **/** — Main dashboard: city cards (current temp, observed high today, projected high, trend °/hr, sun progress), next-trade predictions table (consensus, sources, spread, confidence), Risk/Sell advisor (hedge signals + desktop notifications), at-risk brackets (BUY NO suggestions), and open positions with P/L.
- **/markets** — Kalshi markets feed: upcoming high-temp markets with filters and links to Kalshi.
- **/analytics** — Forecasts vs actuals (MAE per source and per city/source), observation projected_high vs actual (MAE by city), and "when to lock in" (by hour and trend bucket). Data from `source_performance.csv` and `observations_history.csv`.

### Forecasts vs actuals

The analytics page shows Mean Absolute Error (°F) for each source and for the consensus. A bar chart compares MAE by source; a filterable table breaks it down by city and source, with color coding by accuracy (green = lower MAE, coral/red = higher).

### Observations

Observations are fetched from **api.weather.gov** every 2 minutes. The dashboard uses the **latest** observation per station. The [NWS Time Series Viewer](https://www.weather.gov/wrh/timeseries) (hourly mode) can show slightly different values because it uses only hourly observations. Both are from the same station.

Observations are written to `Data/observations_latest.json` and `Data/observations_history.csv`.

---

## TUI (terminal dashboard)

Ncurses-style status view: file row counts (predictions, trades, decisions, eval, intraday), latest predictions table, forecast comparison (all sources), recent trades, next cron runs.

```bash
./scripts/dashboard_live.sh [env] [interval_sec] [limit] [city]
```

- **env**: `prod` (default) or `demo` — filters trades by environment.
- **interval_sec**: Refresh interval in seconds (default `5`).
- **limit**: Number of recent rows to show (default `10`).
- **city**: Optional — focus forecast comparison on one city (`ny` / `il` / `tx` / `fl`). Omit to show all cities.

Example: `./scripts/dashboard_live.sh prod 5 10` — refresh every 5 s, show 10 rows, all cities. Press **q** to quit.
