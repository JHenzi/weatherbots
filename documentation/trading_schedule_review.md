# Trading Schedule Review

This note documents the system's current trading behavior as of March 20, 2026 so it can be reviewed against the intended design.

## Summary

The codebase currently has **two separate trading paths**:

1. The original **main trade path**, which still waits for **1 PM local time** before executing city trades.
2. A newer **morning Kelly path**, which runs at **10:00 AM ET** but is configured separately and is currently in shadow mode by default.

This means "we changed it to trade in the morning" is only partly true in the current implementation. Morning trading was **added**, but the old afternoon live-trade flow was **not replaced**.

## Current Scheduler Behavior

The Docker cron schedule lives in `ops/docker/crontab`.

- `07:00 ET`: run `scripts/run_trade.sh`
- `13:00 ET`: run `scripts/run_trade.sh`
- `14:00 ET`: run `scripts/run_trade.sh`
- `10:00 ET`: run `scripts/run_morning_trade.sh`
- `10:30 ET` to `12:30 ET`: run `scripts/run_exit_manager.sh` for morning positions

Important detail:

- `run_trade.sh` is still the main path for the original Kalshi trade engine.
- `run_morning_trade.sh` is a separate strategy.

## Main Trade Path

The main trade flow is:

1. `scripts/run_trade.sh`
2. `intraday_pulse.py`
3. `kalshi_trader.py`

`run_trade.sh` always computes `TRADE_DATE` as "today" in the configured timezone and refreshes predictions first.

After that, `kalshi_trader.py` applies a **hard 13:00 local-time gate per city**:

- `ny` and `fl` only execute when it is 13:00 in `America/New_York`
- `il` and `tx` only execute when it is 13:00 in `America/Chicago`

As a result:

- the `07:00 ET` run refreshes data but skips actual city trades
- the `13:00 ET` run is the live decision window for `ny` and `fl`
- the `14:00 ET` run is the live decision window for `il` and `tx`

This is the current source of truth for the original trade engine.

## Morning Kelly Path

The morning flow is:

1. `scripts/run_morning_trade.sh`
2. `intraday_pulse.py`
3. `log_market_prices.py`
4. `morning_trader.py`

This path is intentionally separate from the main trade engine.

Key differences:

- It uses `WT_MORNING_SEND_ORDERS`, not `WT_SEND_ORDERS`, to decide whether to place live orders.
- It runs at `10:00 ET`.
- It scans for low-priced positive-Kelly opportunities rather than using the original afternoon "confident mu" trade path.
- It may validly place **zero trades** if no bucket passes its filters.

The module docstring in `morning_trader.py` explicitly says this strategy **"supplements, not replaces"** the 1 PM trade flow.

## Environment Flags

The current Docker config sets:

- `WT_ENV=prod`
- `WT_SEND_ORDERS="true"`
- `WT_MORNING_SEND_ORDERS="false"`
- `TZ=America/New_York`

Operationally, that means:

- the main `run_trade.sh` path is configured for live trading once the 1 PM local gate is reached
- the morning path runs, but remains **shadow-only**

So even if the morning job fires correctly, it will not submit live orders unless `WT_MORNING_SEND_ORDERS` is enabled.

## Why March 20, 2026 Looked Like "It Did Not Run"

On Friday, March 20, 2026, the logs show:

- `07:00 ET`: `run_trade.sh` executed and refreshed predictions
- that same run skipped all cities because each city was "waiting for 13:00"
- `10:00 ET`: `run_morning_trade.sh` executed
- the morning strategy evaluated markets in shadow mode and logged `0 position(s)`
- `10:30`, `11:00`, and `11:30 ET`: `exit_manager.py` ran and found no morning entries to manage

So the system **did run** on March 20, 2026. What happened is:

- the original trade engine had **not reached its live trade window yet**
- the morning strategy **ran but did not place any morning entries**

## Current Design Reality

The current implementation is best described as:

- **Afternoon live trade engine**: still active
- **Morning Kelly strategy**: additional path, currently shadow-only

That is different from a true "morning-only trading" design.

## Review Takeaway

If the intended behavior is:

- "trade in the morning instead of 1 PM"

then the repository is currently **out of alignment** with that intent.

Specifically, the following are still true:

- the main scheduled trade path still targets the 1 PM local execution window
- the 13:00 local gate is still enforced in `kalshi_trader.py`
- the morning path is treated as an extra strategy, not the replacement live path
- the morning path is still configured as shadow-only in `docker-compose.yml`

## Questions For Review

These are the decisions that need to be confirmed:

1. Should the morning strategy replace the original 1 PM path, or continue to coexist with it?
2. If morning should be the primary live strategy, should `run_trade.sh` be removed from the 13:00 and 14:00 schedule?
3. Should the 13:00 local gate in `kalshi_trader.py` be removed, disabled, or kept only for a legacy/manual mode?
4. Should `WT_MORNING_SEND_ORDERS` be turned on in production once the strategy is approved?

Until those decisions are made and the code is updated, the system should be understood as running **both** a morning review/entry path and an afternoon live-trade path.
