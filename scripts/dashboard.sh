#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_FILTER="${1:-prod}"
LIMIT="${2:-10}"
CITY_FILTER="${3:-}"

python - "$ENV_FILTER" "$LIMIT" "$CITY_FILTER" <<'PY'
import csv
import datetime as dt
import os
import sys
from typing import Any

env = (os.environ.get("WT_ENV_FILTER") or (sys.argv[1] if len(sys.argv) > 1 else "prod")).strip()
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
city_filter = (sys.argv[3] if len(sys.argv) > 3 else "").strip().lower()

def _truthy(x: Any) -> bool:
    s = str(x or "").strip().lower()
    return s in ("1", "true", "yes", "y")

def _read_csv(path: str) -> list[dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        return [row for row in r if row]

def _parse_iso(ts: str) -> dt.datetime | None:
    s = (ts or "").strip()
    if not s:
        return None
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None

def _fmt_money(x: float) -> str:
    return f"${x:,.2f}"

def _fmt(x: Any) -> str:
    if x is None:
        return ""
    return str(x)

def _print_table(headers: list[str], rows: list[list[Any]]) -> None:
    cols = len(headers)
    widths = [len(h) for h in headers]
    for row in rows:
        for i in range(cols):
            widths[i] = max(widths[i], len(_fmt(row[i])))
    def line(row):
        return "  ".join(_fmt(row[i]).ljust(widths[i]) for i in range(cols))
    print(line(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(line(row))

pred_path = "Data/predictions_latest.csv"
trades_path = "Data/trades_history.csv"
decisions_path = "Data/decisions_history.csv"
eval_path = "Data/eval_history.csv"
metrics_path = "Data/daily_metrics.csv"
hourly_path = "Data/hourly_forecasts.csv"

preds = _read_csv(pred_path)
trades = _read_csv(trades_path)
decisions = _read_csv(decisions_path)
evals = _read_csv(eval_path)

env_trades = [t for t in trades if (t.get("env") or "").strip() == env and _truthy(t.get("send_orders"))]
env_trades.sort(key=lambda r: (r.get("run_ts") or ""))

env_decisions = [d for d in decisions if (d.get("env") or "").strip() == env]
env_decisions.sort(key=lambda r: (r.get("run_ts") or ""))

env_evals = [e for e in evals if (e.get("env") or "").strip() == env]
env_evals.sort(key=lambda r: (r.get("run_ts") or ""))

now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
print(f"weather-trader status  |  env={env}  |  now={now}")
print()

print("Data files (how many rows):")
for p in [pred_path, trades_path, decisions_path, eval_path, metrics_path, hourly_path]:
    status = "missing"
    if os.path.exists(p):
        try:
            n = sum(1 for _ in open(p, "r", encoding="utf-8")) - 1
            status = f"{n} rows"
        except Exception:
            status = "present"
    print(f"- {p}: {status}")
print()

print("Latest predictions (what the bot believes):")
if not preds:
    print("  (none)")
else:
    rows = []
    for r in preds:
        rows.append([
            r.get("date",""),
            r.get("city",""),
            r.get("tmax_predicted",""),
            r.get("spread_f",""),
            r.get("confidence_score",""),
            (r.get("sources_used") or r.get("forecast_sources") or "")[:60],
        ])
    _print_table(["date","city","predicted_high(F)","disagreement(F)","confidence","sources"], rows)
print()

print("Forecast comparison (all cities, side-by-side):")
if not preds:
    print("  (none)")
else:
    cols = [
        ("city", "city"),
        ("cons", "tmax_predicted"),
        ("om", "tmax_open_meteo"),
        ("vc", "tmax_visual_crossing"),
        ("tom", "tmax_tomorrow"),
        ("wapi", "tmax_weatherapi"),
        ("owm", "tmax_openweathermap"),
        ("pw", "tmax_pirateweather"),
        ("nws", "tmax_weather_gov"),
        ("lstm", "tmax_lstm"),
        ("spr", "spread_f"),
        ("conf", "confidence_score"),
    ]
    def fmt(v):
        s = (v or "").strip()
        if not s:
            return ""
        try:
            return f"{float(s):.2f}"
        except Exception:
            return s

    # Optional filter by city, if provided.
    rows_in = preds
    if city_filter:
        rows_in = [r for r in preds if (r.get("city") or "").strip().lower() == city_filter]

    rows = []
    for r in rows_in:
        rows.append([fmt(r.get(k)) for _, k in cols])
    _print_table([h for h, _ in cols], rows)
print()

print("Hourly forecast trailing average (last 6 samples):")
if not os.path.exists(hourly_path):
    print("  (missing Data/hourly_forecasts.csv)")
else:
    hourly = _read_csv(hourly_path)
    # Prefer the trade_date from predictions_latest; fallback to most recent trade_date in hourly file.
    trade_date = ""
    if preds:
        trade_date = (preds[0].get("date") or "").strip()
    if not trade_date and hourly:
        trade_date = max((r.get("trade_date") or "").strip() for r in hourly)

    def _mean(xs):
        return sum(xs) / len(xs) if xs else None

    def _std(xs):
        if len(xs) <= 1:
            return 0.0
        m = sum(xs) / len(xs)
        return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5

    rows = []
    for city in ["ny", "il", "tx", "fl"]:
        pts = []
        for r in hourly:
            if (r.get("city") or "").strip().lower() != city:
                continue
            if trade_date and (r.get("trade_date") or "").strip() != trade_date:
                continue
            ts = _parse_iso(r.get("timestamp", ""))
            if ts is None:
                continue
            mf = (r.get("mean_forecast") or "").strip()
            if not mf:
                continue
            try:
                v = float(mf)
            except Exception:
                continue
            pts.append((ts, v))
        pts.sort(key=lambda x: x[0])
        pts = pts[-6:]
        xs = [v for _, v in pts]
        if not xs:
            rows.append([trade_date, city, 0, "", "", "", ""])
            continue
        avg = _mean(xs)
        jitter = _std(xs) if len(xs) >= 2 else ""
        drift = (xs[-1] - xs[0]) if len(xs) >= 2 else ""
        rows.append([
            trade_date,
            city,
            len(xs),
            f"{xs[-1]:.2f}",
            f"{avg:.2f}" if avg is not None else "",
            (f"{jitter:.2f}" if jitter != "" else ""),
            (f"{drift:+.2f}" if drift != "" else ""),
        ])
    _print_table(["trade_date","city","n","last(F)","trail_avg(F)","jitter(F)","drift(F)"], rows)
print()

print(f"Last {limit} placed bets (real orders):")
if not env_trades:
    print("  (none)")
else:
    rows = []
    for t in env_trades[-limit:]:
        try:
            count = int(float(t.get("count") or 0))
        except Exception:
            count = 0
        try:
            yes_price = int(float(t.get("yes_price") or 0))
        except Exception:
            yes_price = 0
        cost = count * (yes_price / 100.0)
        max_profit = count * ((100 - yes_price) / 100.0)
        max_loss = cost
        rows.append([
            t.get("run_ts",""),
            t.get("trade_date",""),
            t.get("city",""),
            t.get("market_ticker",""),
            count,
            yes_price,
            _fmt_money(cost),
            _fmt_money(max_profit),
            _fmt_money(max_loss),
        ])
    _print_table(["run_ts","date","city","market","qty","price(c)","paid","could_win","could_lose"], rows)
print()

print("Open bets (not settled yet):")
open_rows = []
if env_trades:
    # Build index of settled (env, date, city) from eval_history where settlement_tmax_f is present
    settled = set()
    for e in env_evals:
        if not _truthy(e.get("send_orders")):
            continue
        if (e.get("settlement_tmax_f") or "").strip():
            settled.add((e.get("trade_date","").strip(), e.get("city","").strip()))
    for t in env_trades:
        key = (t.get("trade_date","").strip(), t.get("city","").strip())
        if key in settled:
            continue
        try:
            count = int(float(t.get("count") or 0))
            yes_price = int(float(t.get("yes_price") or 0))
        except Exception:
            continue
        cost = count * (yes_price / 100.0)
        max_profit = count * ((100 - yes_price) / 100.0)
        open_rows.append([t.get("trade_date",""), t.get("city",""), t.get("market_ticker",""), count, yes_price, _fmt_money(cost), _fmt_money(max_profit)])
if not open_rows:
    print("  (none)")
else:
    _print_table(["date","city","market","qty","price(c)","paid","could_win"], open_rows)
print()

print(f"Settled results (final outcomes + PnL):")
settled_rows = []
for e in env_evals:
    if not _truthy(e.get("send_orders")):
        continue
    pnl = (e.get("realized_pnl_dollars") or "").strip()
    if not pnl:
        continue
    settled_rows.append(e)
if not settled_rows:
    print("  (none)")
else:
    rows = []
    for e in settled_rows[-limit:]:
        rows.append([
            e.get("trade_date",""),
            e.get("city",""),
            e.get("chosen_market_ticker",""),
            e.get("yes_ask",""),
            e.get("count",""),
            e.get("settlement_tmax_f",""),
            e.get("bucket_hit",""),
            e.get("realized_pnl_dollars",""),
        ])
    _print_table(["date","city","market","price(c)","qty","final_high(F)","won?","pnl($)"], rows)
print()

print(f"Recent bot actions (trade vs skip + why):")
if not env_decisions:
    print("  (none)")
else:
    rows = []
    for d in env_decisions[-limit:]:
        rows.append([
            d.get("run_ts",""),
            d.get("trade_date",""),
            d.get("city",""),
            d.get("decision",""),
            (d.get("reason","") or "")[:80],
        ])
    _print_table(["run_ts","date","city","action","why"], rows)
print()

print("Notes:")
print("- 'could_win' assumes the bet resolves YES (payout 100¢ per contract).")
print("- 'could_lose' is the money paid (premium) if it resolves NO.")
print("- Open bets exclude anything already settled in eval_history.csv.")
PY

