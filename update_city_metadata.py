import argparse
import csv
import datetime as dt
import json
import os


def _parse_args():
    p = argparse.ArgumentParser(description="Compute per-city historical MAE and write Data/city_metadata.json.")
    p.add_argument("--metrics-csv", type=str, default="Data/daily_metrics.csv")
    p.add_argument("--out-json", type=str, default="Data/city_metadata.json")
    p.add_argument("--window-days", type=int, default=30, help="Lookback window (days) for historical MAE")
    p.add_argument("--as-of-date", type=str, default=None, help="YYYY-MM-DD (default: yesterday UTC)")
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

    if not os.path.exists(args.metrics_csv):
        # Bootstrap mode: no metrics yet. Write an empty city_metadata.json so trading can fall back safely.
        payload = {
            "as_of": as_of.strftime("%Y-%m-%d"),
            "window_days": int(args.window_days),
            "cities": {},
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source": f"{args.metrics_csv} (missing)",
        }
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"Wrote {args.out_json} (no metrics yet)")
        raise SystemExit(0)

    start = as_of - dt.timedelta(days=int(args.window_days))
    end = as_of

    sums: dict[str, float] = {}
    ns: dict[str, int] = {}

    with open(args.metrics_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            d = (row.get("trade_date") or "").strip()
            city = (row.get("city") or "").strip()
            if not d or not city:
                continue
            if (row.get("metric_type") or "").strip() != "mae_f":
                continue
            if (row.get("source_name") or "").strip() != "consensus":
                continue
            try:
                dd = dt.datetime.strptime(d, "%Y-%m-%d").date()
            except Exception:
                continue
            if dd < start or dd > end:
                continue
            v = _safe_float(row.get("value"))
            if v is None:
                continue
            sums[city] = sums.get(city, 0.0) + float(v)
            ns[city] = ns.get(city, 0) + 1

    cities: dict[str, dict] = {}
    for city in sorted(set(list(sums.keys()) + list(ns.keys()))):
        if ns.get(city, 0) <= 0:
            continue
        cities[city] = {
            "historical_MAE": sums[city] / ns[city],
            "n_days": ns[city],
        }

    payload = {
        "as_of": as_of.strftime("%Y-%m-%d"),
        "window_days": int(args.window_days),
        "cities": cities,
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": "Data/daily_metrics.csv (metric_type=mae_f, source_name=consensus)",
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(f"Wrote {args.out_json} with {len(cities)} cities")

