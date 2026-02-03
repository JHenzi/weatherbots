"""
Shared rolling MAE per (city, source) for MAE-weighted consensus and smart-spread confidence.
Used by run_daily.py and intraday_pulse.py.
"""
import csv
import datetime as dt
import os
from collections import defaultdict
from typing import Optional

try:
    import db
except ImportError:
    db = None  # type: ignore[assignment]


def get_rolling_mae_per_city_source(
    perf_path: str,
    window_days: int = 7,
    end_date: Optional[dt.date] = None,
) -> dict[str, dict[str, float]]:
    """
    Mean MAE per (city, source) over the last window_days ending at end_date (inclusive).
    Returns {city: {source_name: mean_mae}}. Only (city, source) with at least one row in window.
    Default end_date = yesterday so we don't use today before it's graded.
    """
    if end_date is None:
        end_date = dt.date.today() - dt.timedelta(days=1)
    window_days = max(1, int(window_days))
    start = end_date - dt.timedelta(days=window_days - 1)

    out: dict[str, dict[str, float]] = defaultdict(dict)

    # Try Postgres when ENABLE_PG_READ
    if db is not None and getattr(db, "_pg_read_enabled", lambda: False)():
        try:
            cities = ("ny", "il", "tx", "fl")
            sources = [
                "open-meteo", "visual-crossing", "tomorrow", "weatherapi",
                "google-weather", "openweathermap", "pirateweather", "weather.gov", "lstm",
            ]
            for city in cities:
                for source in sources:
                    errs = db.get_source_performance_window(city, source, start, end_date)
                    if errs:
                        out[city][source] = sum(errs) / len(errs)
            return dict(out)
        except Exception:
            pass

    # CSV fallback
    if not perf_path or not os.path.exists(perf_path):
        return {}

    by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    with open(perf_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            c = (row.get("city") or "").strip().lower()
            s = (row.get("source_name") or "").strip()
            if not c or not s:
                continue
            d_str = (row.get("date") or "").strip()[:10]
            try:
                d = dt.datetime.strptime(d_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            if d < start or d > end_date:
                continue
            try:
                err = float(row.get("absolute_error") or 0)
            except (TypeError, ValueError):
                continue
            by_key[(c, s)].append(err)

    for (city, source), errs in by_key.items():
        if errs:
            out[city][source] = sum(errs) / len(errs)
    return dict(out)
