import argparse
import csv
import datetime as dt
import math
import os
from collections import defaultdict

try:
    import db  # type: ignore  # local Postgres helpers
except Exception:  # pragma: no cover - defensive fallback when db.py missing
    db = None  # type: ignore[assignment]


SOURCES = [
    ("consensus", "mu_tmax_f"),
    ("open-meteo", "tmax_open_meteo"),
    ("visual-crossing", "tmax_visual_crossing"),
    ("tomorrow", "tmax_tomorrow"),
    ("weatherapi", "tmax_weatherapi"),
    ("openweathermap", "tmax_openweathermap"),
    ("pirateweather", "tmax_pirateweather"),
    ("weather.gov", "tmax_weather_gov"),
    ("lstm", "tmax_lstm"),
]


def _parse_args():
    p = argparse.ArgumentParser(description="Compute daily rollups from eval_history.csv (requires settlements).")
    p.add_argument("--as-of-date", type=str, default=None, help="YYYY-MM-DD (default: yesterday UTC)")
    p.add_argument("--eval-csv", type=str, default="Data/eval_history.csv")
    p.add_argument("--out-csv", type=str, default="Data/daily_metrics.csv")
    return p.parse_args()


def _safe_float(x: str | None) -> float | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _load_eval_rows_for_date(as_of: dt.date, eval_csv: str):
    """Load eval rows for as_of from Postgres (if enabled) or CSV. Yields dicts with trade_date, city, settlement_tmax_f, etc."""
    if db is not None and getattr(db, "_pg_read_enabled", lambda: False)():
        try:
            rows = db.get_eval_events_for_date(as_of.strftime("%Y-%m-%d"))
            for row in rows:
                yield row
            return
        except Exception as e:
            print(f"Postgres read failed ({e}), falling back to CSV for daily metrics")
    if not os.path.exists(eval_csv):
        raise RuntimeError(f"Missing eval CSV: {eval_csv}")
    with open(eval_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            yield row


if __name__ == "__main__":
    args = _parse_args()
    if args.as_of_date:
        as_of = dt.datetime.strptime(args.as_of_date, "%Y-%m-%d").date()
    else:
        as_of = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)).date()

    # Metrics keyed by (date, city, source)
    abs_errs: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    sq_errs: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    bucket_hits: dict[tuple[str, str], list[int]] = defaultdict(list)
    pnls: dict[tuple[str, str], list[float]] = defaultdict(list)  # dollars

    for row in _load_eval_rows_for_date(as_of, args.eval_csv):
        d = (row.get("trade_date") or "").strip()
        if isinstance(d, dt.date):
            d = d.strftime("%Y-%m-%d")
        if d != as_of.strftime("%Y-%m-%d"):
            continue
        city = (row.get("city") or "").strip()
        if not city:
            continue
        actual = _safe_float(row.get("settlement_tmax_f"))
        if actual is None:
            continue

        # bucket hit and pnl for the chosen trade
        hit = row.get("bucket_hit")
        if hit in ("0", "1"):
            bucket_hits[(d, city)].append(int(hit))
        elif hit is True or hit == 1:
            bucket_hits[(d, city)].append(1)
        elif hit is False or hit == 0:
            bucket_hits[(d, city)].append(0)
        pnl = _safe_float(row.get("realized_pnl_dollars"))
        if pnl is not None:
            pnls[(d, city)].append(pnl)

        for src, col in SOURCES:
            pred = _safe_float(row.get(col))
            if pred is None:
                continue
            err = abs(pred - actual)
            abs_errs[(d, city, src)].append(err)
            sq_errs[(d, city, src)].append((pred - actual) ** 2)

    # Write one row per (date, city, source) plus city-level trade metrics.
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    write_header = not os.path.exists(args.out_csv)
    with open(args.out_csv, "a", newline="") as f:
        fieldnames = [
            "run_ts",
            "trade_date",
            "city",
            "metric_type",
            "source_name",
            "value",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()

        # Log in local timezone (TZ env, default ET) for human readability.
        try:
            from zoneinfo import ZoneInfo

            tz = ZoneInfo((os.getenv("TZ") or "").strip() or "America/New_York")
        except Exception:
            tz = dt.datetime.now().astimezone().tzinfo or dt.timezone.utc
        run_ts = dt.datetime.now(tz=tz).isoformat()

        for (d, city, src), errs in abs_errs.items():
            mae = sum(errs) / len(errs)
            rmse = math.sqrt(sum(sq_errs[(d, city, src)]) / len(sq_errs[(d, city, src)]))
            row_mae = {
                "run_ts": run_ts,
                "trade_date": d,
                "city": city,
                "metric_type": "mae_f",
                "source_name": src,
                "value": f"{mae:.4f}",
            }
            row_rmse = {
                "run_ts": run_ts,
                "trade_date": d,
                "city": city,
                "metric_type": "rmse_f",
                "source_name": src,
                "value": f"{rmse:.4f}",
            }
            w.writerow(row_mae)
            w.writerow(row_rmse)
            if db is not None:
                db.insert_eval_metric_row(row_mae)  # type: ignore[attr-defined]
                db.insert_eval_metric_row(row_rmse)  # type: ignore[attr-defined]

        for (d, city), hits in bucket_hits.items():
            if hits:
                row_hit = {
                    "run_ts": run_ts,
                    "trade_date": d,
                    "city": city,
                    "metric_type": "bucket_hit_rate",
                    "source_name": "trade",
                    "value": f"{sum(hits)/len(hits):.4f}",
                }
                w.writerow(row_hit)
                if db is not None:
                    db.insert_eval_metric_row(row_hit)  # type: ignore[attr-defined]

        for (d, city), vals in pnls.items():
            if vals:
                row_pnl = {
                    "run_ts": run_ts,
                    "trade_date": d,
                    "city": city,
                    "metric_type": "realized_pnl_dollars",
                    "source_name": "trade",
                    "value": f"{sum(vals):.2f}",
                }
                w.writerow(row_pnl)
                if db is not None:
                    db.insert_eval_metric_row(row_pnl)  # type: ignore[attr-defined]

    print(f"Wrote daily metrics for {as_of} to {args.out_csv}")

