# Weather Trader — Dataset

This directory is a public, append-only dataset of daily **temperature forecasts vs.
actual observed highs** for four US cities, plus the trading decisions and market
prices derived from them. It exists so others can study forecast skill (MAE),
calibration, source comparison, and prediction-market behavior.

All files are plain CSVs tracked directly in git (no Git LFS). They are append-only:
each run adds rows, so history is preserved and diffs stay small.

## Cities

| code | city | Kalshi series | NWS station |
|------|------|---------------|-------------|
| `ny` | New York, NY | KXHIGHNY | (Central Park) |
| `il` | Chicago, IL | KXHIGHCHI | |
| `tx` | Austin, TX | KXHIGHAUS | |
| `fl` | Miami, FL | KXHIGHMIA | |

Temperatures are daily maximum (`tmax`) in °F.

## Start here: forecast accuracy / MAE

**`source_performance.csv`** — the most direct accuracy dataset. One row per
(date, city, forecast source) with the predicted high, the actual high, and the
absolute error. Group by `source_name` to rank providers; average `absolute_error`
for MAE.

```
date, city, source_name, predicted_tmax, actual_tmax, absolute_error
```

> **Grading convention (changed 2026-09-01 — read this before using the file).**
> `predicted_tmax` is each provider's forecast taken from the **09:00 local same-day
> snapshot**, which is when the bot commits capital. `actual_tmax` is the settled NWS
> CLI high.
>
> Prior to 2026-09-01 this file graded the *last* snapshot of each day — written at
> 23:00, after the day's high had already occurred. That leaked the outcome into the
> label: sources were effectively scored on how quickly they converged to a value that
> was already known, not on forecast skill. It distorted the ranking severely
> (weather.gov measured 15.26°F under the old convention and 2.29°F under the new one)
> and any provider comparison drawn from the old file is unreliable.
>
> The history was re-graded in place by `scripts/regrade_source_performance.py`.
> Two consequences for anyone analysing this data:
> - Rows before **2026-03-18** are absent for google-weather, openweathermap,
>   pirateweather and weather.gov — those providers were not yet logged to
>   `intraday_forecasts.csv`, so they cannot be re-graded. Coverage from
>   2026-03-18 onward is complete.
> - `lstm` rows were dropped entirely (retired model, no snapshot column to re-grade).
>
> Realistic MAE for every provider at the 09:00 snapshot is **2.3–3.7°F**. If your
> analysis produces a sub-1°F MAE for a next-day high forecast, you are almost
> certainly measuring leakage rather than skill.
>
> Note that `WT_DECISION_HOUR` defaults to `9`, but the bot's live orders are placed at
> **13:00 ET** (NY, Miami) and **14:00 ET** (Chicago, Austin). Forecast accuracy improves
> through the morning, so this file grades a slightly harder task than the one the bot
> actually trades: consensus MAE is 2.21°F at 09:00 but 1.78°F at 13:00 and 1.62°F at
> 14:00. Use `Data/intraday_forecasts.csv` directly if you want to score a specific hour.

**`eval_history.csv`** — per-trade evaluation joined to settlement. Contains the
model's predicted mean (`mu_tmax_f`), the settled actual (`settlement_tmax_f`),
whether the chosen market bucket hit (`bucket_hit`), and realized P&L. Use it to
compute model MAE (`mu_tmax_f` − `settlement_tmax_f`) and to study calibration of
`confidence_score` and `model_prob_yes` against outcomes.

**`daily_metrics.csv`** — long-format rollups (`metric_type`, `source_name`,
`value`) computed nightly, e.g. per-source MAE and bucket hit rate over rolling
windows.

## Predictions & observations

**`predictions_history.csv`** — the daily ensemble prediction and every individual
provider's forecast (`tmax_open_meteo`, `tmax_visual_crossing`, `tmax_tomorrow`,
`tmax_weatherapi`, `tmax_google_weather`, …), plus the ensemble `tmax_predicted`,
`spread_f` (provider disagreement), `confidence_score`, and `weights_used`.

**`observations_history.csv`** — intraday station observations (every ~10 min):
current `temp`, running `observed_high_today`, a `projected_high`, and short-horizon
temperature trends/acceleration. This is the ground-truth signal the projected high
is built from.

**`intraday_forecasts.csv` / `hourly_forecasts.csv`** — snapshots of the forecast as
it evolves through the day.

## Trading & markets

**`decisions_history.csv`** — one row per trade decision: `decision` (trade/skip)
and a `reason` string (guardrails, confidence, market disagreement, etc.).

**`market_price_history.csv`** — Kalshi order-book snapshots (`yes_ask`, `yes_bid`,
depth) per market bucket, for studying how the prediction market priced each day.

**`trades_history.csv`** — orders actually placed (when live).

## Contextual bandit (model-internal) — distributed separately

**`bandit_decisions_history.csv`** and **`context_features_history.csv`** — the
LinUCB contextual bandit's per-run action selection and its 22-dim context feature
vectors. High-frequency (multiple runs/day × 4 cities). Of interest only if you want
to study the online-learning layer that chooses between `forecast` (raw ensemble) and
`blend` (bias-corrected) predictions.

These two are **not tracked in git**: they exceed GitHub's 100MB per-file limit and
grow fastest, and using Git LFS for them reintroduces the storage bloat this repo was
cleaned up to avoid. They are available as a release download rather than in-tree.
Everything needed to study forecast skill / MAE is in the tracked files above.

## Learned state (small, versioned)

Not in this folder's CSV set but tracked alongside: `weights.json` (ensemble source
weights), `city_metadata.json` (per-city bias corrections), and `bandit_state.json`
(the trained LinUCB policy). These carry the system's current "learnings" so a fresh
clone runs calibrated without needing to replay all history.

## Notes

- `run_ts` / `timestamp` / `logged_at` are ISO-8601 with local offset.
- Files are append-only; to reproduce a point in time, filter by date rather than
  checking out an old commit.
- Not tracked here (regenerated at runtime, no learning value): `Data/logs/`,
  the Postgres data dir, and binary feature caches (`*.pkl`).
