import argparse
import csv
import datetime as dt
import json
import os
from collections import defaultdict

try:
    import db  # type: ignore  # local Postgres helpers
except Exception:  # pragma: no cover - defensive fallback when db.py missing
    db = None  # type: ignore[assignment]

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


# Hour of day (local) at which the bot actually commits capital. Provider snapshots are
# graded against this moment, not against the last snapshot of the day.
DECISION_HOUR = int(os.getenv("WT_DECISION_HOUR", "9"))


def _run_stamp(row: dict) -> tuple[str, int] | None:
    """(local date, local hour) of a prediction row, from run_ts (preferred) or timestamp."""
    for key in ("run_ts", "timestamp"):
        raw = (row.get(key) or "").strip()
        if len(raw) >= 13 and raw[10] in ("T", " "):
            try:
                return (raw[:10], int(raw[11:13]))
            except ValueError:
                continue
    return None


def _load_predictions_for_date(path: str, trade_date: str) -> dict[str, dict]:
    if db is not None and getattr(db, "_pg_read_enabled", lambda: False)():
        try:
            return db.get_predictions_for_date(trade_date)
        except Exception as e:
            print(f"Postgres read failed ({e}), falling back to CSV for predictions")
    # Pick the snapshot nearest DECISION_HOUR rather than the last row of the day.
    #
    # Keeping the last row leaks the outcome into the label: by 23:00 the day's max has
    # already happened, so providers are scored on how fast they converge to a known
    # value rather than on how well they forecast at decision time. Measured over 30 days
    # this inverts the provider ranking outright -- weather.gov grades at 2.29F MAE on the
    # 09:00 snapshot and 15.26F on the 23:00 one, and visual-crossing's apparent 0.86F is
    # an artifact of late-day convergence, not skill. Since these MAEs drive the
    # 1/MAE^2 ensemble weights, the bot has been down-weighting its best decision-time
    # providers to near zero.
    best: dict[str, tuple[tuple[int, int], str, dict]] = {}
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if (row.get("date") or "").strip() != trade_date:
                continue
            city = (row.get("city") or "").strip()
            if not city:
                continue
            parsed = _run_stamp(row)
            stamp = (row.get("run_ts") or row.get("timestamp") or "").strip()
            if parsed is None:
                # Rows without a parseable run time sort last but remain a fallback.
                rank = (2, 99)
            else:
                run_date, hour = parsed
                # Prefer a SAME-DAY snapshot over the day-ahead forecast for the same date.
                # Both exist in predictions_history (each run writes today and tomorrow), and
                # both have a 09:00 row, so without this the grader would score the harder
                # day-ahead task rather than the same-day forecast the bot actually trades.
                rank = (0 if run_date == trade_date else 1, abs(hour - DECISION_HOUR))
            prev = best.get(city)
            if prev is None or (rank, stamp) < (prev[0], prev[1]):
                best[city] = (rank, stamp, row)
    return {city: row for city, (_d, _s, row) in best.items()}


def _load_existing_performance_keys(perf_path: str) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    if not os.path.exists(perf_path):
        return keys
    with open(perf_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            d = (row.get("date") or "").strip()
            city = (row.get("city") or "").strip()
            source = (row.get("source_name") or "").strip()
            if d and city and source:
                keys.add((d, city, source))
    return keys


def _append_performance_rows(perf_path: str, rows: list[dict]) -> tuple[int, int]:
    os.makedirs(os.path.dirname(perf_path) or ".", exist_ok=True)
    write_header = not os.path.exists(perf_path)
    existing_keys = _load_existing_performance_keys(perf_path)
    written = 0
    skipped_existing = 0
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
            key = (
                (row.get("date") or "").strip(),
                (row.get("city") or "").strip(),
                (row.get("source_name") or "").strip(),
            )
            if key in existing_keys:
                skipped_existing += 1
                continue
            w.writerow(row)
            existing_keys.add(key)
            written += 1
            if db is not None:
                db.insert_source_performance_row(row)  # type: ignore[attr-defined]
    return written, skipped_existing


def _load_performance_window(perf_path: str, *, city: str, source: str, start: dt.date, end: dt.date) -> list[float]:
    if db is not None and getattr(db, "_pg_read_enabled", lambda: False)():
        try:
            return db.get_source_performance_window(city, source, start, end)
        except Exception as e:
            print(f"Postgres read failed ({e}), falling back to CSV for performance window")
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
    # IMPORTANT:
    # We compute weights from the most recent *available truth*. This script is run for
    # "yesterday" (the event date being graded), so we SHOULD include that date in the window.
    #
    # Prior behavior excluded `trade_date` (end=trade_date-1), which meant the freshly
    # calibrated day (including per-provider forecasts) had zero influence until *tomorrow*.
    #
    # Window is inclusive: [trade_date - (window_days-1), trade_date]
    window_days = max(1, int(window_days))
    start = trade_date - dt.timedelta(days=window_days - 1)
    end = trade_date
    for city in ("ny", "il", "tx", "fl"):
        ws: dict[str, float] = {}
        for source, _ in SOURCES:
            errs = _load_performance_window(perf_path, city=city, source=source, start=start, end=end)
            if not errs:
                continue
            mae = sum(errs) / len(errs)
            # Handle perfect predictions (MAE=0): use a very small floor to avoid division by zero
            # This gives perfect sources extremely high weight (effectively infinite)
            if mae <= 0:
                mae = 0.01  # 0.01°F floor for perfect predictions
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
            row = {
                "run_ts": run_ts,
                "as_of": payload.get("as_of", ""),
                "city": city,
                "window_days": payload.get("window_days", ""),
                "weights_json": json.dumps(payload.get("weights", {}) or {}, sort_keys=True),
            }
            w.writerow(row)
            if db is not None:
                db.insert_weights_history_row(row)  # type: ignore[attr-defined]


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
            # Skip invalid predictions: 0.0 indicates missing data (not a real forecast)
            # Also skip values outside reasonable temperature range (-50°F to 150°F)
            if pred == 0.0 or pred < -50.0 or pred > 150.0:
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

    rows_written, rows_skipped_existing = _append_performance_rows(args.performance_csv, perf_rows)

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

    print(f"Wrote {rows_written} rows to {args.performance_csv}")
    if rows_skipped_existing > 0:
        print(f"Skipped {rows_skipped_existing} duplicate rows already present in {args.performance_csv}")
    print(f"Wrote weights to {args.weights_json}")
