# Data flow (end-to-end)

## 1) Historical + daily weather ingestion

The ingestion logic lives in `daily_prediction.py` (and historically in `data_fetcher_new.ipynb`), following the same data-sources approach as the [LSTM-Automated-Trading-System](https://github.com/pranavgoyanka/LSTM-Automated-Trading-System) repo.

**Data sources used:**

| Source | Features used |
|--------|----------------|
| [Open-Meteo](https://open-meteo.com/) | Maximum temperature, precipitation |
| [Visual Crossing](https://www.visualcrossing.com/) | Maximum temperature, humidity |
| [Meteostat](https://meteostat.net/en/) | Maximum temperature, minimum temperature |
| [NOAA NCEI](https://www.ncei.noaa.gov/) | Maximum temperature, minimum temperature |

**Details:** Open-Meteo archive API (tmax/tmin, sunshine, precipitation, wind); Visual Crossing timeline API (tmax/tmin, humidity, wind; requires API key); Meteostat (tmax/tmin); NOAA NCEI (daily summaries).

**Stored artifacts (per city, in `Data/`):** `merged_df_<city>.pkl`, `prediction_merged_df_<city>.pkl`, `prediction_data_cleaned_<city>.pkl`.

## 2) Cleaning + feature engineering

Performed inside `daily_prediction.py` during the daily run:

- **Unit normalization**: Visual Crossing + NCEI treated as °F; Open-Meteo + Meteostat converted from °C via \(F = C \cdot \frac{9}{5} + 32\).
- **Derived features**: `day` (day-of-year), `tmax_avg`, `tmin_avg` over available sources.
- **Missing values**: forward-fill/back-fill for prediction-time continuity.

## 3) Prediction (three modes)

`daily_prediction.py` supports:

- **`--prediction-mode lstm`**: trained LSTM only.
- **`--prediction-mode forecast`**: provider forecasts only (Open-Meteo, Visual Crossing, Tomorrow, WeatherAPI, optional OpenWeatherMap, Pirate Weather, weather.gov), averaged over available sources.
- **`--prediction-mode blend`**: \(pred = w \cdot forecast + (1 - w) \cdot lstm\) (default `w=0.8`).

Operationally, `forecast`/`blend` are usually better aligned with what markets price than LSTM-only.

## 4) Writing predictions

The trading entrypoint is `run_daily.py` (or intraday `intraday_pulse.py --write-predictions`), which writes:

- `Data/predictions_latest.csv` (overwritten each run)
- `Data/predictions_history.csv` (append-only)

Per-city schema includes: `date`, `city`, per-source forecasts, `tmax_predicted`, `sources_used`, `weights_used`, `spread_f`, `confidence_score`.

## 5) Provider rate limits

Free-tier APIs have strict limits. Tomorrow.io: 3 req/sec, 25 req/hour, 500 req/day. The app uses 1-hour on-disk caches (e.g. `Data/tomorrow_cache`, `Data/openweathermap_cache`, `Data/pirateweather_cache`) and throttling. Avoid re-running forecast mode in a tight loop.

## 6) Weights + consensus (what’s actually running)

### Learned provider weights (`Data/weights.json`)

Updated by the nightly calibration job (`calibrate_sources.py`) once NWS CLI “truth” is available. Inputs: predictions from `Data/predictions_history.csv`, actual max temp from NWS CLI (`truth_engine.py`), errors in `Data/source_performance.csv`. Per-city per-source MAE over a rolling window; \(w_i \propto 1/MAE_i^2\), normalized.

### Intraday consensus (`intraday_pulse.py`)

Cron runs at :00 and :30 (hours 0–16). For each city, fetches provider forecasts; mean forecast \(\mu = \sum_i w_i x_i\) (uses `Data/weights.json` or equal weights); snapshot spread \(\sigma_{\text{snapshot}} = \text{pstdev}(\{x_i\})\). Outputs: `Data/intraday_forecasts.csv`, and with `--write-predictions`: `Data/predictions_latest.csv`, `Data/predictions_history.csv`.

### Trading sigma (`kalshi_trader.py`)

\(\sigma = \max(spread\_f,\ historical\_MAE)\) where `historical_MAE` is from `Data/city_metadata.json`. Used for bucket probabilities and EV.

See **[Mathematical foundations](mathematical_foundations.md)** for full formulas.

## 7) Kalshi mapping + budgeting

Trading logic is in `kalshi_trader.py`. See **[Kalshi markets](kalshi_markets.md)** for series tickers, resolution, and contract selection.

### Budgeting + allocation

- **Configured cap**: `WT_DAILY_BUDGET` is the absolute max per day.
- **Balance-based cap**: bot calls `GET /trade-api/v2/portfolio/balance`; per-run cap = min(configured_cap, 0.5 × available_cash). If balance fetch fails (e.g. demo), falls back to configured cap.
- **Per-city allocation**: based on today’s confidence and historical feedback from `Data/daily_metrics.csv` (MAE, bucket hit-rate). Use `--allocation-mode equal` for even split.

### Order sizing

Orders are auto-sized up to the city’s allocated budget and per-run cap, bounded by `--max-contracts-per-order`. The trader reads best-ask depth and prints warnings when size exceeds depth; it does not auto-cap to displayed depth. For fixed size: `--count N`.

### Sigma (city-aware)

For each city: `sigma = max(current_spread, historical_MAE)` so predictable cities trade with tighter distributions.
