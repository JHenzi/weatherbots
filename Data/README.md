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
