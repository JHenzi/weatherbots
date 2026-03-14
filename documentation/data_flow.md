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

### Bias correction (`city_metadata.json`)

The consensus forecast has a systematic cold bias (~0.5–1.0°F depending on city and conditions). `update_city_metadata.py` computes the correction as the rolling mean of `(actual_tmax − predicted_tmax)` over `--window-days` (default 30). Two corrections are stored per city:

- **`bias_correction_f`** — flat scalar, used as fallback.
- **`bias_correction_by_condition`** — per-condition-bucket corrections for `clear / mixed / precip / snow`. Computed by joining `source_performance.csv` with `context_features_history.csv` on `city + trade_date`. Requires ≥ 3 days per bucket.

Example (TX, 30-day window): clear=+0.04°F, mixed=+1.08°F, precip=+1.79°F. The flat scalar (+0.81°F) was overcorrecting by ~0.77°F on clear days and undercorrecting by ~1.0°F on precip days.

The `blend` bandit action is defined as `forecast + condition_correction`; `morning_trader.py` applies the same correction before computing bucket probabilities.

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

## 8) Contextual bandit (mode selection)

`intraday_pulse.py` runs a LinUCB contextual bandit that selects between `forecast` and `blend` per city per trade decision. LSTM was retired as an action (stale training data; 20–35°F errors).

**Actions:**
- `forecast` — raw weighted-ensemble mean; no correction applied.
- `blend` — `forecast + bias_correction_by_condition[bucket]`; falls back to flat `bias_correction_f` when a bucket has < 3 samples. Lookup key is the current `condition_token` mapped to `clear / mixed / precip / snow`.

**Feature vector (22 dims):** city one-hot (4), day-of-year sin/cos (2), spread_scaled, provider_count_scaled, mean_cloud_cover_scaled, vote_entropy, weather token weights (9: clear, partly_cloudy, cloudy_overcast, rain, snow, storm, fog, wind, other), sky labels (3: sunny, mixed, cloudy).

**Guardrails** — bandit selection is overridden back to `forecast` if:
- `spread_guardrail`: market spread > 3.0°F (`--bandit-max-spread`)
- `confidence_guardrail`: UCB confidence score < 0.35 (`--bandit-min-confidence`)
- `deviation_guardrail`: |blend_pred − forecast_pred| > 6°F (`--bandit-max-deviation-f`)

**Modes** (set via `WT_BANDIT_MODE` env var or `--bandit-mode`):

| Mode | Behaviour |
|------|-----------|
| `off` | Always uses `forecast`; bandit disabled |
| `shadow` | Bandit selects but never applies (pure logging) |
| `canary` | Applies bandit for one city (`--bandit-canary-city`, default `ny`) |
| `live` | Applies bandit for all 4 cities (default in `scripts/run_trade.sh`) |

**Data files:**
- `Data/context_features_history.csv` — condition token/label, sky label, cloud proxy, vote entropy, provider count, spread per run.
- `Data/bandit_decisions_history.csv` — selected vs applied action, guardrail reason, candidate predictions, feature vector.
- `Data/bandit_state.json` — LinUCB A/b matrices and theta vectors (atomically written).

Nightly update (`bandit_update.py`, called from `scripts/run_calibrate.sh`) joins settled actuals from `Data/source_performance.csv` to compute rewards and update the policy:

- `Data/bandit_rewards_history.csv` — per-city error and normalised reward for each action post-settlement.
- `Data/bandit_state_snapshots.csv` — daily policy snapshots for auditing drift.

**Learning signal:** reward = `1 − |error| / 5` (clipped 0–1), favouring the action with smaller absolute temperature error. The bandit learns which condition contexts favour `blend` over `forecast` — e.g. rainy/stormy days where the cold bias is large benefit most from correction, while clear days are already well-calibrated and overcorrection hurts.
