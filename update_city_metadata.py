import argparse
import csv
import datetime as dt
import json
import os


def _parse_args():
    p = argparse.ArgumentParser(description="Compute per-city historical MAE and write Data/city_metadata.json.")
    # Prefer source_performance.csv (doesn't require trades to settle), but keep compatibility
    # with the older daily_metrics.csv shape.
    p.add_argument("--metrics-csv", type=str, default="Data/source_performance.csv")
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

    try:
        from zoneinfo import ZoneInfo

        tz = ZoneInfo((os.getenv("TZ") or "").strip() or "America/New_York")
    except Exception:
        tz = dt.datetime.now().astimezone().tzinfo or dt.timezone.utc

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
        # Detect schema:
        # - daily_metrics.csv: trade_date, metric_type, source_name, value
        # - source_performance.csv: date, city, source_name, absolute_error
        schema = "daily_metrics"
        fns = set((r.fieldnames or []))
        if "absolute_error" in fns and "date" in fns:
            schema = "source_performance"

        for row in r:
            city = (row.get("city") or "").strip()
            if not city:
                continue

            if schema == "source_performance":
                d = (row.get("date") or "").strip()
                if (row.get("source_name") or "").strip() != "consensus":
                    continue
                v = _safe_float(row.get("absolute_error"))
            else:
                d = (row.get("trade_date") or "").strip()
                if (row.get("metric_type") or "").strip() != "mae_f":
                    continue
                if (row.get("source_name") or "").strip() != "consensus":
                    continue
                v = _safe_float(row.get("value"))

            if not d:
                continue
            try:
                dd = dt.datetime.strptime(d, "%Y-%m-%d").date()
            except Exception:
                continue
            if dd < start or dd > end:
                continue
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

    # If we found no consensus MAE rows, don't overwrite an existing metadata file with empties.
    # This commonly happens early on (no settled evals yet) or when settle truth isn't available yet.
    if not cities and os.path.exists(args.out_json):
        try:
            with open(args.out_json, "r") as f:
                existing = json.load(f) or {}
        except Exception:
            existing = None
        if isinstance(existing, dict) and existing.get("cities"):
            print(
                f"No consensus MAE rows found in {args.metrics_csv} for window_days={int(args.window_days)} "
                f"(as_of={as_of}). Leaving existing {args.out_json} unchanged."
            )
            raise SystemExit(0)

    payload = {
        "as_of": as_of.strftime("%Y-%m-%d"),
        "window_days": int(args.window_days),
        "cities": cities,
        # Human-friendly: record updated_at in local timezone (TZ env, default ET).
        "updated_at": dt.datetime.now(tz=tz).isoformat(),
        "source": (
            "Data/source_performance.csv (source_name=consensus, absolute_error)"
            if os.path.basename(args.metrics_csv) == "source_performance.csv"
            else "Data/daily_metrics.csv (metric_type=mae_f, source_name=consensus)"
        ),
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(f"Wrote {args.out_json} with {len(cities)} cities")

