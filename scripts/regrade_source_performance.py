"""
Re-grade Data/source_performance.csv from decision-hour snapshots.

The historical file was built by grading the LAST predictions_history row of each day,
which on most days was written at 23:00 -- after the day's maximum had already occurred.
That leaks the outcome into the label and inverts the provider ranking (weather.gov grades
at 2.29F on the 09:00 snapshot and 15.26F on the 23:00 one). Since these errors drive the
1/MAE^2 ensemble weights in prediction_mae.py, the whole learned weighting was built on a
corrupted signal.

This rebuilds the file from Data/intraday_forecasts.csv, taking for each (city, trade_date)
the SAME-DAY snapshot nearest --decision-hour, and grading it against the settled actual
already recorded in the existing performance log (NWS CLI truth via truth_engine.py).

Ground truth is reused, never recomputed: only the *predicted* side of each row changes.

Usage:
    python scripts/regrade_source_performance.py --dry-run     # report only
    python scripts/regrade_source_performance.py               # rewrite, keeping a backup
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import statistics
from collections import defaultdict

FIELDNAMES = ["date", "city", "source_name", "predicted_tmax", "actual_tmax", "absolute_error"]

# intraday_forecasts.csv column -> source_name used in source_performance.csv
PROVIDER_COLS = {
    "tmax_open_meteo": "open-meteo",
    "tmax_visual_crossing": "visual-crossing",
    "tmax_tomorrow": "tomorrow",
    "tmax_weatherapi": "weatherapi",
    "tmax_google_weather": "google-weather",
    "tmax_openweathermap": "openweathermap",
    "tmax_pirateweather": "pirateweather",
    "tmax_weather_gov": "weather.gov",
}
# The ensemble's own output is graded too, under the name calibrate_sources expects.
CONSENSUS_COL = "mean_forecast"


def _valid(pred: float) -> bool:
    """Same sanity filter calibrate_sources applies: 0.0 means missing, not a forecast."""
    return pred != 0.0 and -50.0 <= pred <= 150.0


def load_actuals(path: str) -> dict[tuple[str, str], float]:
    """(city, date) -> settled actual max F, from the existing graded log."""
    actuals: dict[tuple[str, str], float] = {}
    with open(path, "r", newline="") as f:
        for row in csv.DictReader(f):
            city = (row.get("city") or "").strip().lower()
            date = (row.get("date") or "").strip()
            raw = (row.get("actual_tmax") or "").strip()
            if not city or not date or not raw:
                continue
            try:
                actuals[(city, date)] = float(raw)
            except ValueError:
                continue
    return actuals


def pick_snapshots(path: str, *, decision_hour: int, max_hour_distance: int) -> dict[tuple[str, str], dict]:
    """
    (city, trade_date) -> the same-day snapshot row nearest decision_hour.

    Only same-day rows qualify: a row whose timestamp date differs from trade_date is a
    day-ahead forecast, a harder task than the one the bot actually trades.
    """
    best: dict[tuple[str, str], tuple[int, str, dict]] = {}
    with open(path, "r", newline="") as f:
        for row in csv.DictReader(f):
            city = (row.get("city") or "").strip().lower()
            trade_date = (row.get("trade_date") or "").strip()
            ts = (row.get("timestamp") or "").strip()
            if not city or not trade_date or len(ts) < 13:
                continue
            if ts[:10] != trade_date:
                continue
            try:
                hour = int(ts[11:13])
            except ValueError:
                continue
            dist = abs(hour - decision_hour)
            if dist > max_hour_distance:
                continue
            key = (city, trade_date)
            prev = best.get(key)
            if prev is None or (dist, ts) < (prev[0], prev[1]):
                best[key] = (dist, ts, row)
    return {k: v[2] for k, v in best.items()}


def build_rows(snapshots: dict[tuple[str, str], dict],
               actuals: dict[tuple[str, str], float]) -> tuple[list[dict], dict]:
    out: list[dict] = []
    stats = {"graded_keys": 0, "no_actual": 0, "skipped_values": 0}
    for (city, date), row in sorted(snapshots.items()):
        actual = actuals.get((city, date))
        if actual is None:
            stats["no_actual"] += 1
            continue
        stats["graded_keys"] += 1
        for col, source in list(PROVIDER_COLS.items()) + [(CONSENSUS_COL, "consensus")]:
            raw = (row.get(col) or "").strip()
            if not raw:
                continue
            try:
                pred = float(raw)
            except ValueError:
                continue
            if not _valid(pred):
                stats["skipped_values"] += 1
                continue
            out.append({
                "date": date,
                "city": city,
                "source_name": source,
                "predicted_tmax": f"{pred:.4f}",
                "actual_tmax": f"{actual:.1f}",
                "absolute_error": f"{abs(pred - actual):.4f}",
            })
    out.sort(key=lambda r: (r["date"], r["city"], r["source_name"]))
    return out, stats


def summarize(rows: list[dict]) -> dict[str, tuple[int, float]]:
    by: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        try:
            by[r["source_name"]].append(float(r["absolute_error"]))
        except ValueError:
            continue
    return {s: (len(v), statistics.mean(v)) for s, v in by.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--performance-csv", default="Data/source_performance.csv")
    p.add_argument("--intraday-csv", default="Data/intraday_forecasts.csv")
    p.add_argument("--decision-hour", type=int, default=int(os.getenv("WT_DECISION_HOUR", "9")))
    p.add_argument("--max-hour-distance", type=int, default=4,
                   help="Reject a date whose nearest same-day snapshot is further than this.")
    p.add_argument("--backup-suffix", default=".leaked.bak")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    old_rows = list(csv.DictReader(open(args.performance_csv, newline="")))
    actuals = load_actuals(args.performance_csv)
    snapshots = pick_snapshots(args.intraday_csv,
                               decision_hour=args.decision_hour,
                               max_hour_distance=args.max_hour_distance)
    new_rows, stats = build_rows(snapshots, actuals)

    old_sum = summarize(old_rows)
    new_sum = summarize(new_rows)

    print(f"decision hour        : {args.decision_hour:02d}:00 (same-day snapshots only)")
    print(f"rows  {len(old_rows)} -> {len(new_rows)}")
    print(f"city-days graded     : {stats['graded_keys']}"
          f"  (dropped {stats['no_actual']} with no settled actual)")
    print()
    print(f"{'source':18s} {'old n':>6} {'old MAE':>8} {'new n':>6} {'new MAE':>8} {'change':>8}")
    for source in sorted(set(old_sum) | set(new_sum)):
        on, om = old_sum.get(source, (0, float('nan')))
        nn, nm = new_sum.get(source, (0, float('nan')))
        if nn == 0:
            print(f"{source:18s} {on:6d} {om:8.2f} {'-':>6} {'DROPPED':>8}")
            continue
        if on == 0:
            print(f"{source:18s} {'-':>6} {'-':>8} {nn:6d} {nm:8.2f} {'NEW':>8}")
            continue
        print(f"{source:18s} {on:6d} {om:8.2f} {nn:6d} {nm:8.2f} {nm - om:+8.2f}")

    if args.dry_run:
        print("\n[dry-run] nothing written")
        return

    backup = args.performance_csv + args.backup_suffix
    shutil.copy2(args.performance_csv, backup)
    tmp = args.performance_csv + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(new_rows)
    os.replace(tmp, args.performance_csv)
    print(f"\nbackup  -> {backup}")
    print(f"rewrote -> {args.performance_csv}")


if __name__ == "__main__":
    main()
