# Liquidity-Aware Execution (Order Book Strategy)

This document proposes improvements to execution against Kalshi’s order book so the bot can:

- size more intelligently under **thin liquidity**
- avoid self-inflicted price impact
- decide when to be aggressive vs patient based on **edge** and **confidence**

References (Kalshi docs):
- Limit orders (help): `https://help.kalshi.com/trading/order-types/limit-orders`
- Orderbook response format: `https://docs.kalshi.com/getting_started/orderbook_responses`

---

## 1) Current behavior (baseline)

Today, `kalshi_trader.py`:

- fetches orderbook depth
- infers YES ask from reciprocal NO bids
- computes a `count` from budget caps
- prints a liquidity warning if `count > ask_qty`
- submits **one** limit order at the inferred ask (when `--send-orders`)

Implication:
- you can easily place an order larger than displayed depth
- unfilled remainder rests on the book (no active management)

---

## 2) Kalshi orderbook mechanics (important)

Kalshi’s orderbook shows bids on YES and NO as `[price, quantity]` levels. There may be no explicit asks visible; asks can be inferred via reciprocity:

- YES ask ≈ \(100 -\) best NO bid

Your code already uses this reciprocity (see `_best_yes_prices_from_orderbook` and `get_yes_pricing`).

---

## 3) What we want to optimize

Objective: maximize long-run bankroll growth (or risk-adjusted PnL), which requires balancing:

- **edge**: \(p_{\text{cal}} - p_{\text{mkt}}\) (or EV in cents)
- **execution cost**: worse fill price if we “take” too much depth
- **fill risk**: placing patient orders may not fill
- **inventory/risk**: exposure limits per city/day

This doc focuses on execution; sizing logic lives in `kelly_sizing.md`.

---

## 4) Execution primitives we can implement

### A) Liquidity cap at best ask (simple, safe)

If we want to avoid any immediate impact:

- `count_exec = min(count_target, ask_qty_best * max_take_fraction)`

Where:
- `max_take_fraction` defaults 1.0 (take all displayed best)
- for conservative mode, 0.25–0.5

Pros: very safe
Cons: leaves edge on the table if there’s more depth at similar prices

### B) Two-level model (best + next level)

Your code already computes `next_yes_ask` and warns on >10% worse.

Make it actionable:

- If `next_yes_ask` is much worse (e.g. +10%):
  - cap size to best level
- Else if next level is close:
  - allow consuming more depth up to an “impact budget”

### C) “Effective price” for sizing

For Kelly sizing you need an effective price \(c_{\text{eff}}\).

Approximate it using a simple depth-weighted average across levels you expect to fill immediately:

- fill first `q1` at `ask1`
- fill next `q2` at `ask2`
- …

Compute:
\[
yes\_ask_{\text{eff}} = \frac{\sum_k q_k \cdot ask_k}{\sum_k q_k}
\]

Then use `yes_ask_eff` in EV and Kelly.

### D) Order slicing / child orders (time-based)

Instead of one big order:

- place smaller child orders (e.g. 5–20 contracts) at best ask
- if not filled within `T` seconds, optionally:
  - leave resting (patient)
  - cancel/repost (more aggressive)

This reduces impact and gives the market time to refill.

### E) Passive vs aggressive mode (edge-driven)

Define an “aggressiveness score” from:

- `edge_prob` or `ev_cents`
- `sigma` / confidence
- recent city win-rate (optional)

Policy example:

- **High edge & high confidence** → allow higher `max_take_fraction`, allow next level, use shorter patience
- **Low edge / uncertain** → only take best ask depth (or skip)

---

## 5) Practical “good” defaults for this repo

Given current daily markets + small book depth, a practical conservative policy is:

1) Compute `count_target` (budget or Kelly)
2) Compute `ask1`, `qty1`, optionally `ask2`, `qty2`
3) If `ask2` implies >10% worse price:
   - `count_exec = min(count_target, qty1)`
4) Else:
   - `count_exec = min(count_target, qty1 + qty2)` (or a fraction)
5) Submit limit order for `count_exec`
6) If remaining size exists:
   - either skip (no chase), or place a second resting order at `ask1` for some remainder

This preserves “don’t pay up” while still scaling in when the book is healthy.

---

## 6) Proposed code changes (high-level)

### New helpers
- `execution/orderbook_model.py`
  - parse multiple levels from orderbook
  - compute effective ask for a target size
  - compute suggested `count_exec` given edge and impact thresholds

### Integrate into `kalshi_trader.py`
- replace current “warning only” with an actual sizing cap:
  - `count = min(count_target, recommended_count_from_liquidity(...))`
- log execution decision fields to `eval_history.csv`:
  - `effective_ask`, `levels_used`, `impact_pct`, `liquidity_cap_reason`

---

## 7) How to test safely

1) Run in dry-run and compare:
   - old `count` vs new `count_exec`
   - effective price used
2) Ensure decisions still logged and idempotency still works
3) If enabling live:
   - start with small caps + fractional Kelly

---

## 8) Backup plan interaction (when “no trades”)

If you implement the “best bet” fallback from `kelly_sizing.md`, execution logic should treat fallback trades as **extra conservative**:

- **Cap to best-level liquidity**: never cross into the next level on fallback.
  - `count_exec = min(count_target, ask_qty_best)`
- **No chasing / no repricing loop**: submit a single small limit order and stop.
- **Stricter max spread**: optionally require `yes_spread <= X` (e.g. 6¢) even if the normal strategy might allow more in high-edge cases.

Rationale: fallback exists to avoid “0 exposure days”, not to take execution risk.

