import argparse
import csv
import datetime as dt
import math
import os
from collections import defaultdict


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


if __name__ == "__main__":
    args = _parse_args()
    if args.as_of_date:
        as_of = dt.datetime.strptime(args.as_of_date, "%Y-%m-%d").date()
    else:
        as_of = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)).date()

    if not os.path.exists(args.eval_csv):
        raise RuntimeError(f"Missing eval CSV: {args.eval_csv}")

    # Metrics keyed by (date, city, source)
    abs_errs: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    sq_errs: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    bucket_hits: dict[tuple[str, str], list[int]] = defaultdict(list)
    pnls: dict[tuple[str, str], list[float]] = defaultdict(list)  # dollars

    with open(args.eval_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            d = (row.get("trade_date") or "").strip()
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

        run_ts = dt.datetime.now(dt.timezone.utc).isoformat()
        for (d, city, src), errs in abs_errs.items():
            mae = sum(errs) / len(errs)
            rmse = math.sqrt(sum(sq_errs[(d, city, src)]) / len(sq_errs[(d, city, src)]))
            w.writerow({"run_ts": run_ts, "trade_date": d, "city": city, "metric_type": "mae_f", "source_name": src, "value": f"{mae:.4f}"})
            w.writerow({"run_ts": run_ts, "trade_date": d, "city": city, "metric_type": "rmse_f", "source_name": src, "value": f"{rmse:.4f}"})

        for (d, city), hits in bucket_hits.items():
            if hits:
                w.writerow(
                    {
                        "run_ts": run_ts,
                        "trade_date": d,
                        "city": city,
                        "metric_type": "bucket_hit_rate",
                        "source_name": "trade",
                        "value": f"{sum(hits)/len(hits):.4f}",
                    }
                )

        for (d, city), vals in pnls.items():
            if vals:
                w.writerow(
                    {
                        "run_ts": run_ts,
                        "trade_date": d,
                        "city": city,
                        "metric_type": "realized_pnl_dollars",
                        "source_name": "trade",
                        "value": f"{sum(vals):.2f}",
                    }
                )

    print(f"Wrote daily metrics for {as_of} to {args.out_csv}")

