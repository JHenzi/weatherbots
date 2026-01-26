# Capital Allocation & “Win Rate” (City-Level Strategy)

This document describes how to evolve from “fixed per-city caps” to a more capital-optimal approach that:

- reduces exposure to cities with degraded calibration (e.g., “Miami has been off by 2°F+ lately”)
- increases exposure where we have stable edge
- stays robust under small sample sizes

It complements:
- `documentation/kelly_sizing.md` (per-bet sizing)
- `documentation/orderbook_execution.md` (execution under liquidity)

---

## 1) Current behavior (baseline)

`kalshi_trader.py` already does a form of allocation:

- compute a per-run total spend cap (min(configured, balance_fraction * available_cash))
- split across cities using historical metrics in `Data/daily_metrics.csv`
  - consensus MAE (`metric_type=mae_f, source_name=consensus`)
  - bucket hit rate (`metric_type=bucket_hit_rate, source_name=trade`)

This is good, but we can improve:
- incorporate *recent* performance more strongly
- incorporate realized PnL volatility / drawdown
- incorporate confidence calibration quality (is \(p_{\text{model}}\) well-calibrated?)

---

## 2) What “win rate” should mean here

For a daily temperature bucket bet:

- `bucket_hit` (0/1) from `eval_history.csv` is a natural “win” label
- but “win rate” alone is not sufficient:
  - you can have a high hit-rate but overpay (negative EV)

So we want both:

- **Calibration / accuracy**: MAE, hit-rate, Brier score
- **Profitability**: realized PnL after costs

---

## 3) Recommended metrics to compute (rolling window)

For each city, over last \(W\) settled days:

- **Hit-rate**: \(\hat{h} = \text{mean}(bucket\_hit)\)
- **Mean realized PnL**: \(\hat{\mu}_{pnl}\)
- **PnL volatility**: \(\hat{\sigma}_{pnl}\)
- **Sharpe-like score**: \(\hat{\mu}_{pnl} / (\hat{\sigma}_{pnl} + \epsilon)\)
- **Calibration**: Brier score of predicted probability for chosen bucket:
  - \(BS = \text{mean}((p_{\text{cal}} - bucket\_hit)^2)\)

You can store these in `Data/daily_metrics.csv` as additional metric types.

---

## 4) Robustness under small sample size

Because we often only have a few settled days, use shrinkage:

- **Beta-Binomial** prior for hit-rate:
  - posterior mean \(E[h] = (a + wins) / (a+b+n)\)
  - choose \(a=b=1\) (uniform) or \(a=b=2\) (slightly conservative)

- **PnL shrinkage**:
  - clamp negative outliers
  - EWMA weighting so the most recent week matters more than a month ago

---

## 5) Allocation policy options

### A) Kelly-at-city-level (advanced)

Treat each city as a stream of bets with an estimated edge distribution and allocate capital to maximize growth.

This requires:
- stable calibrated probabilities
- stable cost model
- enough data to estimate variance

### B) Practical heuristic (recommended next step)

Compute a city score:

\[
Score_{city} =
\underbrace{\frac{1}{1+MAE^2}}_{\text{accuracy}}
\times
\underbrace{(0.5 + \hat{h})}_{\text{win-rate}}
\times
\underbrace{\text{clamp}(Sharpe)}_{\text{profitability}}
\times
\underbrace{\text{confidence-calibration term}}_{\text{optional}}
\]

Normalize scores to allocate the per-run cap.

Add guardrails:
- minimum per-city fraction to avoid starving cities due to noise
- maximum per-city fraction to avoid concentration risk

---

## 6) Proposed code changes

- Extend `daily_metrics.py` to compute additional metrics:
  - `brier_score`
  - `mean_realized_pnl`
  - `pnl_stddev`
  - `sharpe_like`
- Extend `kalshi_trader.py` allocation:
  - use those metrics when present
  - fall back to current behavior if not present

---

## 7) Expected outcome

This should address your concern directly:

- if Miami starts “losing” or becomes systematically miscalibrated, its score drops → smaller budget
- cities with stable calibration and good realized PnL get more budget

---

## 8) Backup plan: avoid “0-bet days”

Allocation + execution can legitimately produce **no trades** on some days (low edge, thin books, or strict caps). If your operational goal is to always place at least one bet:

- Add a final step after allocation/execution:
  - if `placed_trades == 0`, select the single “best bet” from the evaluated candidate set (see `kelly_sizing.md`)
  - size it with a tiny fixed cap (e.g., \$1–\$5) and strict liquidity limits

This keeps the portfolio from being completely inactive while still respecting risk limits and making the “forced trade” explicitly auditable in logs.

