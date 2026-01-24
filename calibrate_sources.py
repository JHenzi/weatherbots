import argparse
import csv
import datetime as dt
import json
import os
from collections import defaultdict

from truth_engine import get_actual_tmax_from_nws_cli


SOURCES = [
    ("consensus", "tmax_predicted"),
    ("open-meteo", "tmax_open_meteo"),
    ("visual-crossing", "tmax_visual_crossing"),
    ("tomorrow", "tmax_tomorrow"),
    ("weatherapi", "tmax_weatherapi"),
    ("google-weather", "tmax_google_weather"),
    ("openweathermap", "tmax_openweathermap"),
    ("pirateweather", "tmax_pirateweather"),
    ("weather.gov", "tmax_weather_gov"),
    ("lstm", "tmax_lstm"),
]


def _parse_args():
    p = argparse.ArgumentParser(description="Update source_performance.csv and weights.json from NWS CLI truth.")
    p.add_argument("--trade-date", type=str, required=True, help="YYYY-MM-DD (event date to grade)")
    p.add_argument("--predictions-history", type=str, default="Data/predictions_history.csv")
    p.add_argument("--performance-csv", type=str, default="Data/source_performance.csv")
    p.add_argument("--weights-json", type=str, default="Data/weights.json")
    p.add_argument("--window-days", type=int, default=14, help="Rolling window for MAE weights")
    return p.parse_args()


def _load_predictions_for_date(path: str, trade_date: str) -> dict[str, dict]:
    by_city: dict[str, dict] = {}
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if (row.get("date") or "").strip() != trade_date:
                continue
            city = (row.get("city") or "").strip()
            if not city:
                continue
            # keep last row (latest run) for that date/city
            by_city[city] = row
    return by_city


def _append_performance_rows(perf_path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(perf_path) or ".", exist_ok=True)
    write_header = not os.path.exists(perf_path)
    with open(perf_path, "a", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "date",
                "city",
                "source_name",
                "predicted_tmax",
                "actual_tmax",
                "absolute_error",
            ],
        )
        if write_header:
            w.writeheader()
        for row in rows:
            w.writerow(row)


def _load_performance_window(perf_path: str, *, city: str, source: str, start: dt.date, end: dt.date) -> list[float]:
    if not os.path.exists(perf_path):
        return []
    errs: list[float] = []
    with open(perf_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if (row.get("city") or "").strip() != city:
                continue
            if (row.get("source_name") or "").strip() != source:
                continue
            d = (row.get("date") or "").strip()
            try:
                dd = dt.datetime.strptime(d, "%Y-%m-%d").date()
            except Exception:
                continue
            if dd < start or dd > end:
                continue
            try:
                errs.append(float(row.get("absolute_error") or ""))
            except Exception:
                continue
    return errs


def _compute_weights(perf_path: str, *, trade_date: dt.date, window_days: int) -> dict:
    # Weight per city per source: w = 1/MAE^2 over last N days.
    weights: dict[str, dict] = {}
    start = trade_date - dt.timedelta(days=window_days)
    end = trade_date - dt.timedelta(days=1)
    for city in ("ny", "il", "tx", "fl"):
        ws: dict[str, float] = {}
        for source, _ in SOURCES:
            errs = _load_performance_window(perf_path, city=city, source=source, start=start, end=end)
            if not errs:
                continue
            mae = sum(errs) / len(errs)
            if mae <= 0:
                continue
            ws[source] = 1.0 / (mae * mae)
        # normalize
        s = sum(ws.values())
        if s > 0:
            ws = {k: v / s for k, v in ws.items()}
        weights[city] = {
            "window_days": window_days,
            "as_of": trade_date.strftime("%Y-%m-%d"),
            "weights": ws,
        }
    return weights


def _append_weights_history(path: str, weights: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["run_ts", "as_of", "city", "window_days", "weights_json"],
        )
        if write_header:
            w.writeheader()
        # Log in local timezone (TZ env, default ET) for human readability.
        try:
            from zoneinfo import ZoneInfo

            tz = ZoneInfo((os.getenv("TZ") or "").strip() or "America/New_York")
        except Exception:
            tz = dt.datetime.now().astimezone().tzinfo or dt.timezone.utc
        run_ts = dt.datetime.now(tz=tz).isoformat()
        for city, payload in (weights or {}).items():
            w.writerow(
                {
                    "run_ts": run_ts,
                    "as_of": payload.get("as_of", ""),
                    "city": city,
                    "window_days": payload.get("window_days", ""),
                    "weights_json": json.dumps(payload.get("weights", {}) or {}, sort_keys=True),
                }
            )


if __name__ == "__main__":
    args = _parse_args()
    trade_dt = dt.datetime.strptime(args.trade_date, "%Y-%m-%d").date()

    preds = _load_predictions_for_date(args.predictions_history, args.trade_date)
    if not preds:
        raise RuntimeError(
            f"No predictions found for date={args.trade_date} in {args.predictions_history}. "
            f"Calibration can only score dates that were actually predicted/logged."
        )

    perf_rows: list[dict] = []
    for city, row in preds.items():
        try:
            truth = get_actual_tmax_from_nws_cli(city, trade_dt)
        except Exception as e:
            # CLI is usually published the next day; skip if not available yet.
            print(f"SKIP truth for city={city} date={args.trade_date}: {e}")
            continue
        actual = float(truth.observed_max_f)

        for source, col in SOURCES:
            v = row.get(col)
            if v is None or v == "":
                continue
            try:
                pred = float(v)
            except Exception:
                continue
            perf_rows.append(
                {
                    "date": args.trade_date,
                    "city": city,
                    "source_name": source,
                    "predicted_tmax": f"{pred:.4f}",
                    "actual_tmax": f"{actual:.1f}",
                    "absolute_error": f"{abs(pred - actual):.4f}",
                }
            )

    _append_performance_rows(args.performance_csv, perf_rows)

    if not perf_rows:
        # Bootstrap mode: don't block the system. We just didn't have CLI truth yet.
        # Keep existing weights.json (if any) and exit cleanly.
        print(
            f"No truth rows written for date={args.trade_date}. "
            f"Most likely the NWS CLI report is not published yet. Skipping weight update."
        )
        raise SystemExit(0)

    weights = _compute_weights(args.performance_csv, trade_date=trade_dt, window_days=args.window_days)
    os.makedirs(os.path.dirname(args.weights_json) or ".", exist_ok=True)
    with open(args.weights_json, "w") as f:
        json.dump(weights, f, indent=2, sort_keys=True)
    _append_weights_history("Data/weights_history.csv", weights)

    print(f"Wrote {len(perf_rows)} rows to {args.performance_csv}")
    print(f"Wrote weights to {args.weights_json}")

