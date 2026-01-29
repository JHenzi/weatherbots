---
name: Postgres Final Transition Plan
overview: Finalize the transition to Postgres by implementing a "Postgres-First, CSV-Fallback" architecture across all core trading and forecasting scripts.
todos:
  - id: db_read_layer
    content: Implement ENABLE_PG_READ and Query Helpers in db.py
    status: pending
  - id: update_trader
    content: Update kalshi_trader.py (Idempotency, Allocation, Intraday Gate)
    status: pending
  - id: update_pulse
    content: Update intraday_pulse.py (Volatility/Dynamic Weights)
    status: pending
  - id: update_metrics
    content: Update daily_metrics.py (Aggregation Logic)
    status: pending
  - id: update_calibration
    content: Update calibrate_sources.py (Performance Window Loading)
    status: pending
  - id: verify_full_system
    content: End-to-end verification with ENABLE_PG_READ=true
    status: pending
isProject: false
---

# Postgres Final Transition Plan

This plan details the final steps to move the system's "Source of Truth" from local CSV files to the Postgres database. We will use a **Hybrid Mode** approach to ensure zero downtime and safe fallbacks.

## 1. Data Access Layer (`db.py`)

We will enhance `db.py` to support high-performance reads with automatic host detection.

- **Config**: Add `_pg_read_enabled()` checking for `ENABLE_PG_READ=true`.
- **Query Helpers**: Implement methods that return data in the specific formats (mostly lists of dicts) currently expected by the CSV parsers.
  - `get_already_traded(env, trade_date, city_code)` -> bool
  - `get_allocation_scores(trade_date, window_days)` -> `dict[city, score]`
  - `get_recent_intraday_snapshots(city_code, trade_date, limit)` -> `list[dict]`
  - `get_source_performance_window(city_code, source_name, start, end)` -> `list[float]`
  - `get_predictions_for_date(trade_date)` -> `dict[city, row]`

## 2. Script Updates (Hybrid Logic)

Each script will be updated to follow this pattern:

```python
def load_data(...):
    if db and db._pg_read_enabled():
        try:
            return db.get_data(...)
        except Exception as e:
            log(f"Postgres read failed ({e}), falling back to CSV")
    
    return load_from_csv(...)
```

### Affected Files:

- `**kalshi_trader.py**`:
  - `_already_traded`: Check `trades` table.
  - `_load_allocation_scores`: Query `eval_metrics`.
  - `get_intraday_gate`: Query `intraday_snapshots`.
- `**intraday_pulse.py**`:
  - `_load_recent_intraday_history`: Query `intraday_snapshots`.
- `**calibrate_sources.py**`:
  - `_load_performance_window`: Query `source_performance`.
  - `_load_predictions_for_date`: Query `predictions`.
- `**daily_metrics.py**`:
  - Refactor main loop to query `eval_events` where `settlement_tmax_f` is present.

## 3. Configuration & Rollout

1. **Stage 1**: Deploy updated code with `ENABLE_PG_READ=false` (default). System continues to use CSVs.
2. **Stage 2**: Enable `ENABLE_PG_READ=true` in development/local. Verify that logs show Postgres being used.
3. **Stage 3**: Deploy to production container.
4. **Stage 4 (Optional)**: After confidence is high, CSV writing can be disabled or moved to a background "archive" process.

## 4. Performance Optimization

- Ensure all queries use the indexes defined in `db/schema.sql`.
- Use `JOIN cities` in every query to resolve city codes efficiently.
- Use `Json` helper for `provider_values` and `weights` columns.

