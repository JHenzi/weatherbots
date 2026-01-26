# Kelly Criterion Bet Sizing (Design Notes)

This document proposes how to extend the current bot to use **Kelly-style sizing** for Kalshi YES contracts, while accounting for:

- **Probability estimation uncertainty** (we do *not* know the true \(p\))
- **Transaction costs / slippage**
- **Liquidity constraints** (partial fills, thin books)
- **City-specific “win rate” / reliability** (recent performance)

It is written to match this repo’s current flow (intraday forecasts → `predictions_latest.csv` → `kalshi_trader.py`).

References:
- Kelly criterion overview: `https://en.wikipedia.org/wiki/Kelly_criterion`
- Practical considerations / rebalancing: `https://www.frontiersin.org/articles/10.3389/fams.2020.577050/full`
- Fractional Kelly explanation: `https://quantmatter.com/kelly-criterion-formula/`

---

## 1) Current behavior (baseline)

Today, the bot:

- builds a point estimate \(\mu\) and a spread proxy \(\sigma\) from forecast sources
- estimates a bucket probability \(p_{\text{model}}\) (Normal CDF over bucket bounds)
- uses EV-style heuristics and caps (budget + max contracts) to pick a `count`

What it does **not** do is map \(p\) and market price into a bankroll-optimal fraction.

---

## 2) Contract economics for a YES buy

Let:

- `yes_ask` = ask price in cents (1..99)
- \(c = yes\_ask / 100\) = dollars paid per contract
- payout is \$1 if YES wins, \$0 otherwise

If you buy 1 contract:

- **win profit** = \((1 - c)\) dollars
- **loss** = \(c\) dollars

If you stake \$s at price \(c\), you buy \(n = s/c\) contracts.

- if win: bankroll multiplier is \(1 + f \cdot (1/c - 1)\)
- if lose: bankroll multiplier is \(1 - f\)

where \(f = s / B\) is fraction of bankroll \(B\) staked on this bet.

---

## 3) “Full Kelly” fraction for a binary contract

Define:

- \(p\) = probability the YES outcome resolves true
- \(R = 1/c = 100/yes\_ask\) = gross return multiple on the stake *when winning*

Full Kelly fraction for this two-outcome bet:

\[
f^\* = \frac{pR - 1}{R - 1}
\]

Equivalent forms:

- \(R - 1 = (1-c)/c = (100-yes\_ask)/yes\_ask\)
- If \(f^\* \le 0\), you should not bet (or you should bet the opposite side; our current bot is YES-only).

### Practical note
Full Kelly is extremely sensitive to \(p\) estimation error. In practice, we almost always use **fractional Kelly**.

---

## 4) Fractional Kelly (recommended)

Use a conservative multiplier \(k \in (0,1)\):

\[
f = k \cdot \max(0, f^\*)
\]

Suggested defaults:

- **Quarter-Kelly**: \(k = 0.25\)
- **Half-Kelly**: \(k = 0.5\)

Why: probability calibration error, regime shifts, and liquidity/slippage can all cause real \(p\) to be lower than \(p_{\text{model}}\).

---

## 5) Where does \(p\) come from?

### A) Current \(p_{\text{model}}\)
`kalshi_trader.py` computes:

- \(\mu\) = consensus mean
- \(\sigma = \max(spread\_f,\ historical\_MAE)\)
- \(p_{\text{model}} = P(L \le T \le H)\) using Normal CDF

This is a good starting point, but needs calibration.

### B) Calibrated probability \(p_{\text{cal}}\) (proposal)
Use historical outcomes to map raw probabilities to calibrated probabilities.

Two simple approaches:

1) **Platt scaling / logistic calibration** on historical decisions:
   - features: `p_model`, `sigma`, `city`, `month`, `spread_f`
   - label: `bucket_hit` from `eval_history.csv`
   - output: \(p_{\text{cal}}\)

2) **Reliability shrinkage** using city recent win-rate:
   - compute rolling hit-rate per city (e.g. last 30 days)
   - shrink: \(p_{\text{cal}} = \lambda p_{\text{model}} + (1-\lambda)\, \bar{p}_{city}\)
   - where \(\bar{p}_{city}\) could be the empirical hit-rate for that city, and \(\lambda\) increases with sample size

### C) Add cost/slippage to effective \(p\) or price
The real edge is reduced by:

- crossing spreads (YES ask vs feasible fill)
- partial fills (time risk)
- fees (if any)

We should compute an **effective fill price** \(c_{\text{eff}}\) (see `orderbook_execution.md`) and use it in Kelly.

---

## 6) Converting Kelly fraction → contract count

Given:

- bankroll \(B\) (use `available_cash` from Kalshi balance)
- fraction \(f\) from fractional Kelly
- effective price \(c_{\text{eff}}\)

Target stake: \(s = f \cdot B\)

Contracts:

\[
count_{\text{kelly}} = \left\lfloor \frac{s}{c_{\text{eff}}} \right\rfloor
\]

Then apply existing hard caps:

- `max_balance_fraction` (already exists)
- `max_dollars_total`, `max_dollars_per_city`
- `max_contracts_per_order`
- optional: liquidity caps (recommended)

---

## 7) City “win rate” and capital allocation

You already have a city allocation system driven by:

- consensus MAE and bucket hit-rate from `Data/daily_metrics.csv`

To make this closer to “maximize capital”, we can:

- weight cities by **risk-adjusted return**, not just accuracy
- incorporate:
  - realized PnL per city (from `eval_history.csv` or `daily_metrics.csv`)
  - win rate (bucket hit-rate)
  - drawdown / variance proxy (e.g., stddev of daily pnl)

Design target:

- allocate more bankroll to cities with consistent edge and stable calibration
- allocate less to cities with unstable forecast→truth mapping (e.g., Miami if it’s systematically off or under-dispersed)

---

## 8) Proposed code changes (high-level)

### New modules
- `sizing/kelly.py`
  - `kelly_fraction(p, yes_ask_cents) -> f_star`
  - `fractional_kelly(f_star, k) -> f`
  - `count_from_fraction(f, bankroll, price) -> count`

- `calibration/probability_calibration.py`
  - fit + apply calibration mapping (logistic or isotonic)
  - store coefficients in `Data/prob_calibration.json`

### Integrate into `kalshi_trader.py`
- After computing `p_yes` and pricing:
  - compute `p_cal` (optional; fallback to `p_yes`)
  - compute `f = fractional_kelly(kelly(p_cal, yes_ask), k)`
  - derive `count` from `f`, then apply caps and liquidity rules

### New config knobs (env/CLI)
- `--kelly-fraction` (default 0.25)
- `--min-edge-prob` or keep `--min-ev-cents`
- `--liquidity-cap-mode` (see execution doc)

---

## 9) Backup plan: “best bet” fallback when zero trades

Problem: with strict guardrails (min EV, liquidity caps, city budgets, etc.) the system can end up placing **0 trades** on a day. If your operational goal is “always express at least one view”, implement an optional fallback.

### Principle

- The fallback should be **opt-in** and **small** by default.
- It should pick the **single best candidate** after evaluating the whole opportunity set.
- It should **not** override absolute safety rails (e.g., never exceed balance fraction caps).

### Candidate set

After the main pass produces `eligible_trades` (possibly empty), define `fallback_candidates` as trades that at least have:

- valid market pricing (`yes_ask` known)
- valid model probability (`p_model` or `p_cal`)
- non-trivial liquidity (`ask_qty` > 0)
- within basic sanity constraints (e.g., `yes_ask` in 1..99, `sigma` finite)

### Scoring options (choose one)

1) **Best expected value** (most aligned with current code):
   - maximize \(EV_{\text{cents}} = 100 p_{\text{cal}} - yes\_ask\)

2) **Most confident** (your request: “most confident buy”):
   - maximize \(p_{\text{cal}}\) subject to a price sanity check (avoid paying 95¢ for a 0.96 probability unless you truly want that)

3) **Best Kelly growth** (most principled if using Kelly):
   - maximize expected log growth \(E[\log(W)]\) using the same \(p_{\text{cal}}\) and `yes_ask` model, with a very small fractional Kelly \(k\)

### Size policy for fallback (recommended)

Keep fallback exposure intentionally small and capped, e.g.:

- `fallback_max_dollars = min($5, 0.01 * available_cash)` (tunable)
- `fallback_kelly_fraction = min(k, 0.10)` (quarter/half Kelly is too aggressive here)
- `fallback_count = floor(fallback_max_dollars / c_eff)` (and cap by `max_contracts_per_order`)

### Guardrails that still apply

Even in fallback, enforce:

- balance fraction cap (existing `max_balance_fraction`)
- `max_yes_spread_cents` / `max_spread` sanity (avoid pathological markets)
- liquidity cap at best level if the book is thin (see `orderbook_execution.md`)

### Logging

If fallback triggers, log:

- `decision=fallback_trade`
- `fallback_reason=no_primary_trades`
- the chosen scoring mode (best_ev / most_confident / best_log_growth)

This makes it easy to audit how often the bot is “forcing” a trade.

