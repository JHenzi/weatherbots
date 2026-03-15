import argparse
import csv
import datetime as dt
import json
import os

# Map raw condition tokens → 4 coarse buckets used for stratified bias correction.
_CONDITION_BUCKET: dict[str, str] = {
    "clear": "clear",
    "partly_cloudy": "mixed",
    "cloudy_overcast": "mixed",
    "wind": "mixed",
    "other": "mixed",
    "rain": "precip",
    "storm": "precip",
    "fog": "precip",
    "snow": "snow",
}
CONDITION_BUCKETS = ("clear", "mixed", "precip", "snow")


def _parse_args():
    p = argparse.ArgumentParser(description="Compute per-city historical MAE and write Data/city_metadata.json.")
    # Prefer source_performance.csv (doesn't require trades to settle), but keep compatibility
    # with the older daily_metrics.csv shape.
    p.add_argument("--metrics-csv", type=str, default="Data/source_performance.csv")
    p.add_argument("--context-csv", type=str, default="Data/context_features_history.csv",
                   help="context_features_history.csv for condition-stratified bias correction")
    p.add_argument("--out-json", type=str, default="Data/city_metadata.json")
    p.add_argument("--window-days", type=int, default=30, help="Lookback window (days) for historical MAE")
    p.add_argument("--as-of-date", type=str, default=None, help="YYYY-MM-DD (default: yesterday UTC)")
    p.add_argument(
        "--morning-entries-csv",
        type=str,
        default="Data/morning_entries.csv",
        help="morning_entries.csv — used to compute historical_MAE_morning separately from "
             "the consensus (1PM) MAE so the two trading windows don't skew each other.",
    )
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

    # Build (city, trade_date) → condition_bucket lookup from context_features_history.
    # Prefer decision_role=trade rows; fall back to first available row for that city+date.
    condition_lookup: dict[tuple[str, str], str] = {}
    if args.context_csv and os.path.exists(args.context_csv):
        _trade_keys: set[tuple[str, str]] = set()
        with open(args.context_csv, "r", newline="") as _cf:
            for _row in csv.DictReader(_cf):
                _city = (_row.get("city") or "").strip().lower()
                _date = (_row.get("trade_date") or "").strip()
                _token = (_row.get("condition_token") or "other").strip().lower()
                _bucket = _CONDITION_BUCKET.get(_token, "mixed")
                _role = (_row.get("decision_role") or "").strip().lower()
                _key = (_city, _date)
                if _role == "trade":
                    condition_lookup[_key] = _bucket
                    _trade_keys.add(_key)
                elif _key not in _trade_keys:
                    condition_lookup[_key] = _bucket

    sums: dict[str, float] = {}
    ns: dict[str, int] = {}
    # Signed bias: sum of (actual - predicted) for consensus source.
    # Positive = we run cold (under-predict), negative = we run hot.
    bias_sums: dict[str, float] = {}
    bias_ns: dict[str, int] = {}
    # Per-condition-bucket signed bias: keyed by (city, bucket).
    cond_bias_sums: dict[tuple[str, str], float] = {}
    cond_bias_ns: dict[tuple[str, str], int] = {}

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
                # Signed bias: actual_tmax - predicted_tmax
                actual = _safe_float(row.get("actual_tmax"))
                predicted = _safe_float(row.get("predicted_tmax"))
                signed = None if (actual is None or predicted is None) else actual - predicted
            else:
                d = (row.get("trade_date") or "").strip()
                if (row.get("metric_type") or "").strip() != "mae_f":
                    continue
                if (row.get("source_name") or "").strip() != "consensus":
                    continue
                v = _safe_float(row.get("value"))
                signed = None  # daily_metrics schema has no signed info

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

            if signed is not None:
                bias_sums[city] = bias_sums.get(city, 0.0) + float(signed)
                bias_ns[city] = bias_ns.get(city, 0) + 1
                # Condition-stratified accumulation.
                bucket = condition_lookup.get((city, d))
                if bucket:
                    ck = (city, bucket)
                    cond_bias_sums[ck] = cond_bias_sums.get(ck, 0.0) + float(signed)
                    cond_bias_ns[ck] = cond_bias_ns.get(ck, 0) + 1

    # --- 10 AM MAE: read morning_entries.csv, join mu_pred against settled actuals. ---
    # morning_entries rows must have mu_pred (added when buy at 10:00 was implemented).
    # Rows without mu_pred are skipped so old shadow entries don't pollute the metric.
    morning_mae_sums: dict[str, float] = {}
    morning_mae_ns: dict[str, int] = {}
    # Collect settled actuals once (already in the actuals_by_date dict we build inline here).
    settled_actuals: dict[tuple[str, str], float] = {}
    with open(args.metrics_csv, "r", newline="") as _f:
        for _row in csv.DictReader(_f):
            _city = (_row.get("city") or "").strip()
            _date = (_row.get("date") or "").strip()
            _actual = _safe_float(_row.get("actual_tmax"))
            if _city and _date and _actual is not None:
                settled_actuals[(_city, _date)] = _actual

    if os.path.exists(args.morning_entries_csv):
        with open(args.morning_entries_csv, "r", newline="") as _mf:
            for _row in csv.DictReader(_mf):
                _city = (_row.get("city") or "").strip().lower()
                _date = (_row.get("trade_date") or "").strip()
                _mu_str = (_row.get("mu_pred") or "").strip()
                _status = (_row.get("status") or "").strip()
                # Only count rows that reached a real exit (not shadow/error/open).
                if _status in ("shadow", "error", "open", ""):
                    continue
                if not _mu_str:
                    continue
                _mu = _safe_float(_mu_str)
                if _mu is None:
                    continue
                try:
                    _dd = dt.datetime.strptime(_date, "%Y-%m-%d").date()
                except Exception:
                    continue
                if _dd < start or _dd > end:
                    continue
                _actual = settled_actuals.get((_city, _date))
                if _actual is None:
                    continue
                _err = abs(_mu - _actual)
                morning_mae_sums[_city] = morning_mae_sums.get(_city, 0.0) + _err
                morning_mae_ns[_city] = morning_mae_ns.get(_city, 0) + 1

    cities: dict[str, dict] = {}
    for city in sorted(set(list(sums.keys()) + list(ns.keys()))):
        if ns.get(city, 0) <= 0:
            continue
        entry: dict = {
            "historical_MAE": sums[city] / ns[city],
            "n_days": ns[city],
        }
        if morning_mae_ns.get(city, 0) > 0:
            entry["historical_MAE_morning"] = round(
                morning_mae_sums[city] / morning_mae_ns[city], 4
            )
            entry["morning_mae_n"] = morning_mae_ns[city]
        if bias_ns.get(city, 0) > 0:
            entry["bias_correction_f"] = round(bias_sums[city] / bias_ns[city], 4)
            entry["bias_n_days"] = bias_ns[city]
        # Per-condition-bucket corrections (fall back to bias_correction_f if bucket missing).
        by_condition: dict[str, float] = {}
        for bucket in CONDITION_BUCKETS:
            ck = (city, bucket)
            if cond_bias_ns.get(ck, 0) >= 3:  # require at least 3 days to trust the estimate
                by_condition[bucket] = round(cond_bias_sums[ck] / cond_bias_ns[ck], 4)
        if by_condition:
            entry["bias_correction_by_condition"] = by_condition
            entry["bias_condition_ns"] = {b: cond_bias_ns.get((city, b), 0) for b in CONDITION_BUCKETS}
        cities[city] = entry

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

