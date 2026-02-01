import argparse
import csv
import datetime as dt
import json
import math
import os
import statistics
import time
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

load_dotenv()

try:
    import db  # type: ignore  # local Postgres helpers
except Exception:  # pragma: no cover - defensive fallback when db.py missing
    db = None  # type: ignore[assignment]

try:
    import requests_cache  # type: ignore
except ModuleNotFoundError:
    requests_cache = None


# Providers included in intraday snapshots (09/15/21 + final 22:00).
# This is the "full" forecast set used for trading decisions.
SOURCES_ORDER = [
    "google-weather",
    "open-meteo",
    "openweathermap",
    "pirateweather",
    "tomorrow",
    "visual-crossing",
    "weather.gov",
    "weatherapi",
]

CITIES = ["ny", "il", "tx", "fl"]
LATLON = {
    "ny": (40.79736, -73.97785),
    "il": (41.78701, -87.77166),
    "tx": (30.14440, -97.66876),
    "fl": (25.77380, -80.19360),
}

PROVIDER_COLS: dict[str, str] = {
    "open-meteo": "tmax_open_meteo",
    "visual-crossing": "tmax_visual_crossing",
    "tomorrow": "tmax_tomorrow",
    "weatherapi": "tmax_weatherapi",
    "google-weather": "tmax_google_weather",
    "openweathermap": "tmax_openweathermap",
    "pirateweather": "tmax_pirateweather",
    "weather.gov": "tmax_weather_gov",
}


def _local_tz() -> dt.tzinfo:
    tzname = (os.getenv("TZ") or "").strip() or "America/New_York"
    try:
        return ZoneInfo(tzname)
    except Exception:
        return dt.datetime.now().astimezone().tzinfo or dt.timezone.utc


def _now_iso_local() -> str:
    return dt.datetime.now(tz=_local_tz()).isoformat()


def _safe_float(x) -> float | None:
    if x is None:
        return None
    try:
        s = str(x).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _parse_iso_dt(s: str) -> dt.datetime | None:
    ss = (s or "").strip()
    if not ss:
        return None
    try:
        return dt.datetime.fromisoformat(ss.replace("Z", "+00:00"))
    except Exception:
        return None


def _try_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def _confidence_from_spread(spread_f: float) -> float:
    # Match run_daily.py behavior.
    if spread_f <= 1.5:
        return 1.0
    if spread_f >= 3.0:
        return 0.0
    return float((3.0 - float(spread_f)) / (3.0 - 1.5))


def _skill_from_weights(weights_used: dict[str, float]) -> float:
    """
    Derive a per-city skill factor from the learned weights.

    Mirrors run_daily._skill_from_weights:
    - Interpret weights as a probability distribution over providers.
    - Use normalized Shannon entropy to capture how many competent sources
      contribute meaningfully to the ensemble:
        * High entropy (diversified, multiple good sources) → skill_conf ~ 1.0
        * Low entropy (one dominant source) → skill_conf ~ 0.0
    """
    if not weights_used:
        return 0.5

    ws = [max(0.0, float(v)) for v in weights_used.values()]
    s = sum(ws)
    if s <= 0:
        return 0.5

    probs = [w / s for w in ws if w > 0.0]
    if len(probs) <= 1:
        return 0.5

    H = -sum(p * math.log(p) for p in probs)
    H_max = math.log(len(probs))
    if H_max <= 0:
        return 0.5

    entropy_norm = max(0.0, min(1.0, H / H_max))
    return float(entropy_norm)


def _load_weights(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _weights_for_city(weights_all: dict, city: str) -> dict[str, float]:
    """
    Supported shapes:
      - {"ny": {"weights": {"open-meteo": 0.2, ...}, ...}, ...}
      - {"ny": {"open-meteo": 0.2, ...}, ...}
    """
    node = weights_all.get(city) if isinstance(weights_all, dict) else None
    if isinstance(node, dict) and isinstance(node.get("weights"), dict):
        node = node.get("weights")
    if not isinstance(node, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in node.items():
        # LSTM is a model output, not a forecast provider.
        if str(k) == "lstm":
            continue
        if str(k) not in SOURCES_ORDER:
            continue
        fv = _safe_float(v)
        if fv is None:
            continue
        out[str(k)] = float(fv)
    s = sum(max(0.0, v) for v in out.values())
    if s <= 0:
        return {}
    return {k: float(v) / s for k, v in out.items()}


def _append_intraday_row(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path)
    fieldnames = [
        "timestamp",
        "city",
        "trade_date",
        "mean_forecast",
        "current_sigma",
        "tmax_open_meteo",
        "tmax_visual_crossing",
        "tmax_tomorrow",
        "tmax_weatherapi",
        "tmax_google_weather",
        "tmax_openweathermap",
        "tmax_pirateweather",
        "tmax_weather_gov",
        "sources_used",
        "weights_used",
    ]
    if not write_header:
        try:
            with open(path, "r", newline="") as f:
                r = csv.reader(f)
                existing = next(r, [])
            existing = [str(x).strip() for x in existing if str(x).strip() != ""]
            if existing != fieldnames:
                _migrate_intraday_forecasts_schema(path, fieldnames)
        except Exception:
            # Best-effort: if migration fails, still append (worst case: row preserved but columns shifted).
            pass
    payload = {k: row.get(k, "") for k in fieldnames}
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(payload)
    if db is not None:
        db.insert_intraday_snapshot_row(payload)  # type: ignore[attr-defined]


def _load_recent_intraday_history(
    path: str,
    *,
    city: str,
    trade_date: str,
    max_rows: int = 4,
) -> list[dict]:
    """
    Load up to max_rows most recent intraday snapshots for (city, trade_date).
    Used for lead-time tracking / per-provider volatility.
    """
    if db is not None and getattr(db, "_pg_read_enabled", lambda: False)():
        try:
            return db.get_recent_intraday_snapshots(
                city_code=city, trade_date=trade_date, limit=max_rows
            )
        except Exception as e:
            print(f"Postgres read failed ({e}), falling back to CSV for intraday history")
    if not path or not os.path.exists(path):
        return []
    rows: list[tuple[dt.datetime, dict]] = []
    try:
        with open(path, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                if not row:
                    continue
                if (row.get("city") or "").strip() != (city or "").strip():
                    continue
                if (row.get("trade_date") or "").strip() != (trade_date or "").strip():
                    continue
                ts = _parse_iso_dt(row.get("timestamp") or "")
                if ts is None:
                    continue
                rows.append((ts, row))
    except Exception:
        return []
    if not rows:
        return []
    rows.sort(key=lambda t: t[0])
    rows = rows[-max_rows:]
    return [row for _, row in rows]


def _compute_volatility_info(
    history_rows: list[dict],
    current_vals: dict[str, float],
) -> dict[str, dict[str, float]]:
    """
    For each provider, compute:
      - volatility: mean |delta| over the last up-to-3 deltas
      - last_delta: most recent delta
      - mean_level: mean forecast level across history+current

    Deltas are consecutive differences in the provider's forecast for the same
    (city, trade_date) across pulses.
    """
    info: dict[str, dict[str, float]] = {}
    for src, col in PROVIDER_COLS.items():
        cur = current_vals.get(src)
        if cur is None:
            continue
        series: list[float] = []
        for row in history_rows:
            v = _safe_float(row.get(col))
            if v is None:
                continue
            series.append(float(v))
        series.append(float(cur))
        if len(series) < 2:
            volatility = 0.0
            last_delta = 0.0
        else:
            deltas = [series[i + 1] - series[i] for i in range(len(series) - 1)]
            recent = deltas[-3:]
            volatility = sum(abs(d) for d in recent) / float(len(recent))
            last_delta = deltas[-1]
        mean_level = sum(series) / float(len(series)) if series else 0.0
        info[src] = {
            "volatility": float(volatility),
            "last_delta": float(last_delta),
            "mean_level": float(mean_level),
        }
    return info


def _apply_volatility_weighting(
    available: dict[str, float],
    base_weights: dict[str, float],
    history_rows: list[dict],
) -> tuple[dict[str, float], float]:
    """
    Consensus 2.0:
      - Start from base_weights (learned weights if available, else uniform).
      - For each provider, compute volatility over the last few pulses.
      - If volatility > 2°F OR >10% of its mean level, apply a 50% penalty.
      - Agreement bonus: if two volatile providers share a similar last_delta
        (same sign, similar magnitude), do NOT penalize them (treat as trend).

    Returns:
      - new_weights: renormalized dynamic weights for available providers.
      - stability_score: 0..1 summarizing how many high-weight providers are
        stable or in coherent trend (used for conviction_score).
    """
    if not available:
        return ({}, 0.5)

    vol_info = _compute_volatility_info(history_rows, available)

    # If we don't have learned weights, start from uniform over available.
    bw: dict[str, float]
    if base_weights:
        bw = {k: float(v) for k, v in base_weights.items() if k in available}
        total = sum(max(0.0, v) for v in bw.values())
        if total <= 0:
            n = len(available)
            bw = {k: 1.0 / float(n) for k in available.keys()}
        else:
            bw = {k: float(v) / float(total) for k, v in bw.items()}
    else:
        n = len(available)
        bw = {k: 1.0 / float(n) for k in available.keys()}

    # First pass: identify which providers are "volatile".
    THRESH_DEG = 2.0
    is_volatile: dict[str, bool] = {}
    for src in available.keys():
        meta = vol_info.get(src) or {}
        vol = float(meta.get("volatility", 0.0))
        mean_level = abs(float(meta.get("mean_level", 0.0)))
        high_deg = vol > THRESH_DEG
        high_pct = False
        if mean_level > 0:
            high_pct = vol > (0.10 * mean_level)
        is_volatile[src] = bool(high_deg or high_pct)

    # Second pass: agreement bonus for coherent high skew and penalties for outliers.
    factors: dict[str, float] = {k: 1.0 for k in available.keys()}
    for src in available.keys():
        if not is_volatile.get(src, False):
            continue
        meta_i = vol_info.get(src) or {}
        delta_i = float(meta_i.get("last_delta", 0.0))
        if delta_i == 0.0:
            # Flat but flagged volatile via threshold; still treat as noise.
            factors[src] = 0.5
            continue
        sign_i = 1.0 if delta_i > 0 else -1.0
        found_partner = False
        for other in available.keys():
            if other == src or not is_volatile.get(other, False):
                continue
            meta_j = vol_info.get(other) or {}
            delta_j = float(meta_j.get("last_delta", 0.0))
            if delta_j == 0.0:
                continue
            sign_j = 1.0 if delta_j > 0 else -1.0
            if sign_j != sign_i:
                continue
            if abs(delta_j - delta_i) <= 1.0:
                found_partner = True
                break
        if not found_partner:
            # Volatile and not supported by another similarly-moving provider.
            factors[src] = 0.5

    # Third pass: staleness penalty. If most providers are updating in a common
    # direction but one stays flat, treat that as "stale" and downweight it.
    deltas = [
        float((vol_info.get(src) or {}).get("last_delta", 0.0)) for src in available.keys()
    ]
    # Ignore tiny noise when inferring the pack's movement.
    significant = [d for d in deltas if abs(d) >= 0.1]
    if len(significant) >= 2:
        try:
            import statistics  # local import to avoid top-level dependency issues
        except Exception:
            statistics = None  # type: ignore[assignment]

        if statistics is not None:
            med = statistics.median(significant)
            trend_mag = abs(med)
            if trend_mag >= 0.5:
                # Pack is moving meaningfully; penalize sources that are near-flat.
                for src in available.keys():
                    meta = vol_info.get(src) or {}
                    d = float(meta.get("last_delta", 0.0))
                    if abs(d) < 0.1:
                        # Everyone else is updating in roughly the same direction,
                        # this one is effectively unchanged → likely stale.
                        factors[src] = min(float(factors.get(src, 1.0)), 0.5)

    # Combine base weights with volatility factors and renormalize.
    pre: dict[str, float] = {}
    for src in available.keys():
        w0 = float(bw.get(src, 0.0))
        if w0 <= 0.0:
            continue
        f = float(factors.get(src, 1.0))
        pre[src] = w0 * f
    s = sum(pre.values())
    if s <= 0:
        new_weights = dict(bw)
    else:
        new_weights = {k: float(v) / float(s) for k, v in pre.items()}

    # Stability score: share of weight on non-penalized providers (or coherent trends).
    if not new_weights:
        stability = 0.5
    else:
        stable_mass = sum(
            new_weights[src]
            for src in new_weights.keys()
            if float(factors.get(src, 1.0)) >= 1.0
        )
        stability = max(0.0, min(1.0, float(stable_mass)))
    return (new_weights, float(stability))


def _migrate_intraday_forecasts_schema(path: str, new_fieldnames: list[str]) -> None:
    """
    Data/intraday_forecasts.csv may have an older, smaller header. When we add new provider
    columns, appends can create shifted rows. This migration rewrites the file using the
    canonical header.
    """
    old_fieldnames = [
        "timestamp",
        "city",
        "trade_date",
        "mean_forecast",
        "current_sigma",
        "tmax_open_meteo",
        "tmax_visual_crossing",
        "tmax_tomorrow",
        "tmax_weatherapi",
        "sources_used",
        "weights_used",
    ]

    with open(path, "r", newline="") as f:
        r = csv.reader(f)
        _ = next(r, None)  # existing header (may be stale)
        rows = list(r)

    out_rows: list[dict[str, str]] = []
    for row in rows:
        if not row:
            continue
        if len(row) == len(old_fieldnames):
            d = dict(zip(old_fieldnames, row))
        elif len(row) == len(new_fieldnames):
            d = dict(zip(new_fieldnames, row))
        else:
            d = dict(zip(old_fieldnames, row[: len(old_fieldnames)]))
        out_rows.append(d)

    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=new_fieldnames)
        w.writeheader()
        for d in out_rows:
            w.writerow({k: d.get(k, "") for k in new_fieldnames})
    os.replace(tmp, path)


def _write_predictions_latest(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "date",
        "city",
        "tmax_predicted",
        "tmax_lstm",
        "tmax_forecast",
        "forecast_sources",
        "tmax_open_meteo",
        "tmax_visual_crossing",
        "tmax_tomorrow",
        "tmax_weatherapi",
        "tmax_google_weather",
        "tmax_openweathermap",
        "tmax_pirateweather",
        "tmax_weather_gov",
        "spread_f",
        "confidence_score",
        "sources_used",
        "weights_used",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            # Ensure conviction_score doesn't leak into the CSV yet if it's not in the header.
            # (We will add it to the history schema properly in a later step if needed).
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _append_predictions_history(path: str, latest_rows: list[dict], *, extra_fields: dict[str, str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path)
    
    # Use the same canonical header as run_daily.py to avoid misalignment.
    fieldnames = [
        "date",
        "city",
        "tmax_predicted",
        "tmax_lstm",
        "tmax_forecast",
        "forecast_sources",
        "tmax_open_meteo",
        "tmax_visual_crossing",
        "tmax_tomorrow",
        "tmax_weatherapi",
        "tmax_google_weather",
        "tmax_openweathermap",
        "tmax_pirateweather",
        "tmax_weather_gov",
        "spread_f",
        "confidence_score",
        "sources_used",
        "weights_used",
        "run_ts",
        "env",
        "prediction_mode",
        "blend_forecast_weight",
        "refresh_history",
        "retrain_lstm",
    ]
    
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for r in latest_rows:
            rr = {k: r.get(k, "") for k in fieldnames if k in r}
            rr.update(extra_fields)
            # Fill missing keys with empty strings to avoid DictWriter errors
            row_to_write = {k: rr.get(k, "") for k in fieldnames}
            w.writerow(row_to_write)
            if db is not None:
                db.insert_prediction_row(row_to_write)  # type: ignore[attr-defined]


def forecast_tmax_open_meteo(*, city: str, trade_dt: dt.date) -> float | None:
    lat, lon = LATLON[city]
    session = (
        requests_cache.CachedSession("Data/open_meteo_forecast_cache", expire_after=3600)
        if requests_cache is not None
        else requests.Session()
    )
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max",
        "temperature_unit": "fahrenheit",
        "timezone": "UTC",
        "start_date": trade_dt.strftime("%Y-%m-%d"),
        "end_date": trade_dt.strftime("%Y-%m-%d"),
    }
    r = session.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return None
    js = r.json()
    daily = js.get("daily") or {}
    temps = daily.get("temperature_2m_max") or []
    if not temps:
        return None
    return _safe_float(temps[0])


def forecast_tmax_visual_crossing(*, city: str, trade_dt: dt.date) -> float | None:
    api_key = os.getenv("VISUAL_CROSSING_API_KEY")
    if not api_key:
        return None
    lat, lon = LATLON[city]
    start = trade_dt.strftime("%Y-%m-%d")
    end = trade_dt.strftime("%Y-%m-%d")
    url = (
        "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
        + f"{lat}%2C{lon}/{start}/{end}"
        + "?unitGroup=us&include=days&key="
        + api_key
        + "&contentType=json"
    )
    session = (
        requests_cache.CachedSession("Data/visualcrossing_cache", expire_after=3600)
        if requests_cache is not None
        else requests.Session()
    )
    r = session.get(url, timeout=30)
    if r.status_code != 200:
        return None
    js = r.json()
    days = js.get("days") or []
    if not days:
        return None
    return _safe_float((days[0] or {}).get("tempmax"))


def forecast_tmax_tomorrow(*, city: str, trade_dt: dt.date, _state: dict) -> float | None:
    api_key = os.getenv("TOMORROW")
    if not api_key:
        return None
    lat, lon = LATLON[city]
    url = "https://api.tomorrow.io/v4/weather/forecast"
    params = {
        "location": f"{lat},{lon}",
        "timesteps": "1d",
        "units": "imperial",
        "apikey": api_key,
    }
    session = (
        requests_cache.CachedSession("Data/tomorrow_cache", expire_after=3600)
        if requests_cache is not None
        else requests.Session()
    )

    last_ts = _state.get("last_req_ts")
    if last_ts is not None:
        elapsed = time.time() - float(last_ts)
        if elapsed < 0.40:
            time.sleep(0.40 - elapsed)

    r = session.get(url, params=params, timeout=30)
    if not getattr(r, "from_cache", False):
        _state["last_req_ts"] = time.time()
    if r.status_code != 200:
        return None
    js = r.json()
    tl = (js.get("timelines") or {}).get("daily") or []
    if not tl:
        return None
    target = trade_dt.strftime("%Y-%m-%d")
    for item in tl:
        t = str(item.get("time") or "")
        if t.startswith(target):
            vals = item.get("values") or {}
            return _safe_float(vals.get("temperatureMax"))
    t0 = str((tl[0] or {}).get("time") or "")
    if t0.startswith(target):
        vals = (tl[0] or {}).get("values") or {}
        return _safe_float(vals.get("temperatureMax"))
    return None


def forecast_tmax_weatherapi(*, city: str, trade_dt: dt.date) -> float | None:
    api_key = os.getenv("WEATHERAPI")
    if not api_key:
        return None
    lat, lon = LATLON[city]
    url = "https://api.weatherapi.com/v1/forecast.json"
    params = {
        "key": api_key,
        "q": f"{lat},{lon}",
        "days": 1,
        "dt": trade_dt.strftime("%Y-%m-%d"),
        "alerts": "no",
        "aqi": "no",
    }
    session = (
        requests_cache.CachedSession("Data/weatherapi_cache", expire_after=3600)
        if requests_cache is not None
        else requests.Session()
    )
    r = session.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return None
    js = r.json()
    fc = (js.get("forecast") or {}).get("forecastday") or []
    if not fc:
        return None
    day = (fc[0] or {}).get("day") or {}
    return _safe_float(day.get("maxtemp_f"))


def forecast_tmax_google_weather(*, city: str, trade_dt: dt.date) -> float | None:
    """
    Google Weather hourly forecast -> take max hourly temp on trade_dt (local to the location).

    Env var:
      - GOOGLE (preferred)
      - GOOGLE_WEATHER_API_KEY (fallback)
    """
    api_key = os.getenv("GOOGLE") or os.getenv("GOOGLE_WEATHER_API_KEY")
    if not api_key:
        return None

    lat, lon = LATLON[city]
    url = "https://weather.googleapis.com/v1/forecast/hours:lookup"
    params = {
        "key": api_key,
        "location.latitude": lat,
        "location.longitude": lon,
        "hours": 240,
    }
    session = (
        requests_cache.CachedSession("Data/google_weather_cache", expire_after=3600)
        if requests_cache is not None
        else requests.Session()
    )
    r = session.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return None
    js = r.json() or {}

    tz_id = ((js.get("timeZone") or {}) if isinstance(js.get("timeZone"), dict) else {}).get("id") or "UTC"
    try:
        tz = ZoneInfo(str(tz_id))
    except Exception:
        tz = ZoneInfo("UTC")

    hours = js.get("forecastHours") or []
    best = None
    for h in hours:
        interval = (h or {}).get("interval") or {}
        st = interval.get("startTime") or interval.get("endTime")
        if not st:
            continue
        try:
            t = dt.datetime.fromisoformat(str(st).replace("Z", "+00:00"))
        except Exception:
            continue
        try:
            local_day = t.astimezone(tz).date()
        except Exception:
            local_day = t.date()
        if local_day != trade_dt:
            continue

        temp = (h or {}).get("temperature") or {}
        deg = temp.get("degrees")
        unit = str(temp.get("unit") or "").upper()
        fv = _safe_float(deg)
        if fv is None:
            continue
        if unit == "CELSIUS":
            fv = (fv * 9.0 / 5.0) + 32.0
        best = fv if best is None else max(best, fv)
    return best


def forecast_tmax_openweathermap(*, city: str, trade_dt: dt.date) -> float | None:
    if str(os.getenv("DISABLE_OPENWEATHERMAP", "")).strip().lower() in ("1", "true", "yes", "y"):
        return None
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    if not api_key:
        return None
    lat, lon = LATLON[city]
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "imperial"}
    session = (
        requests_cache.CachedSession("Data/openweathermap_cache", expire_after=3600)
        if requests_cache is not None
        else requests.Session()
    )
    r = session.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return None
    js = r.json()
    tz_offset = int(((js.get("city") or {}).get("timezone")) or 0)
    items = js.get("list") or []
    if not items:
        return None
    max_t = None
    for it in items:
        ts = it.get("dt")
        if ts is None:
            continue
        try:
            local_day = (dt.datetime.utcfromtimestamp(int(ts)) + dt.timedelta(seconds=tz_offset)).date()
        except Exception:
            continue
        if local_day != trade_dt:
            continue
        main = it.get("main") or {}
        v = _safe_float(main.get("temp_max", main.get("temp")))
        if v is None:
            continue
        max_t = v if max_t is None else max(max_t, v)
    return max_t


def forecast_tmax_pirateweather(*, city: str, trade_dt: dt.date) -> float | None:
    api_key = os.getenv("PIRATE_WEATHER_API_KEY") or os.getenv("PIRATE_WEATER_API_KEY")
    if not api_key:
        return None
    lat, lon = LATLON[city]
    url = f"https://api.pirateweather.net/forecast/{api_key}/{lat},{lon}"
    params = {"units": "us", "exclude": "currently,minutely,hourly,alerts"}
    session = (
        requests_cache.CachedSession("Data/pirateweather_cache", expire_after=3600)
        if requests_cache is not None
        else requests.Session()
    )
    r = session.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return None
    js = r.json()
    tz_name = js.get("timezone") or "UTC"
    try:
        tz = ZoneInfo(str(tz_name))
    except Exception:
        tz = ZoneInfo("UTC")
    daily = (js.get("daily") or {}).get("data") or []
    for d in daily:
        ts = d.get("time")
        if ts is None:
            continue
        try:
            dd = dt.datetime.fromtimestamp(int(ts), tz=tz).date()
        except Exception:
            continue
        if dd != trade_dt:
            continue
        return _safe_float(d.get("temperatureMax", d.get("temperatureHigh")))
    return None


def forecast_tmax_weather_gov(*, city: str, trade_dt: dt.date) -> float | None:
    user_agent = os.getenv("NWS_USER_AGENT")
    if not user_agent:
        return None
    lat, lon = LATLON[city]
    points_url = f"https://api.weather.gov/points/{lat},{lon}"
    session = (
        requests_cache.CachedSession("Data/weather_gov_cache", expire_after=3600)
        if requests_cache is not None
        else requests.Session()
    )
    headers = {"User-Agent": user_agent, "Accept": "application/geo+json"}
    r = session.get(points_url, headers=headers, timeout=30)
    if r.status_code != 200:
        return None
    js = r.json()
    props = js.get("properties") or {}
    forecast_url = props.get("forecast")
    if not forecast_url:
        return None
    r2 = session.get(str(forecast_url), headers=headers, timeout=30)
    if r2.status_code != 200:
        return None
    js2 = r2.json()
    periods = (js2.get("properties") or {}).get("periods") or []
    best = None
    for p in periods:
        st = p.get("startTime")
        if not st:
            continue
        try:
            dt_start = dt.datetime.fromisoformat(str(st))
        except Exception:
            continue
        if dt_start.date() != trade_dt:
            continue
        if p.get("isDaytime") is True:
            return _safe_float(p.get("temperature"))
        v = _safe_float(p.get("temperature"))
        if v is None:
            continue
        best = v if best is None else max(best, v)
    return best


def _parse_args():
    p = argparse.ArgumentParser(description="Fetch forecasts at specific intraday times.")
    p.add_argument("--trade-date", type=str, required=True, help="YYYY-MM-DD (event date the forecasts target)")
    p.add_argument("--out-csv", type=str, default="Data/intraday_forecasts.csv")
    p.add_argument("--weights-json", type=str, default="Data/weights.json")
    p.add_argument("--env", type=str, default=os.getenv("KALSHI_ENV", "demo"))
    p.add_argument(
        "--write-predictions",
        action="store_true",
        help="Also write Data/predictions_latest.csv and append to predictions_history (for the 22:00 trade run).",
    )
    p.add_argument("--predictions-latest", type=str, default="Data/predictions_latest.csv")
    p.add_argument("--predictions-history", type=str, default="Data/predictions_history.csv")
    p.add_argument(
        "--print",
        action="store_true",
        help="Print fetched forecasts to stdout (useful for docker exec / debugging).",
    )
    p.add_argument(
        "--print-format",
        type=str,
        default="table",
        choices=["table", "json"],
        help="Output format when --print is set (default: table).",
    )
    p.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write Data/*.csv files (still performs API calls; combine with --print).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    trade_dt = dt.datetime.strptime(args.trade_date, "%Y-%m-%d").date()
    weights_all = _load_weights(args.weights_json)
    tomorrow_state: dict[str, float | None] = {"last_req_ts": None}

    started = _now_iso_local()
    wrote = 0
    pred_rows: list[dict] = []
    intraday_rows: list[dict] = []

    # Lead 0/1: fetch and store forecasts for both today (lead 0) and tomorrow (lead 1).
    lead_dates = [
        trade_dt,
        trade_dt + dt.timedelta(days=1),
    ]

    for target_dt in lead_dates:
        trade_date_str = target_dt.strftime("%Y-%m-%d")
        for city in CITIES:
            tmax_open_meteo = _try_call(forecast_tmax_open_meteo, city=city, trade_dt=target_dt)
            tmax_visual_crossing = _try_call(
                forecast_tmax_visual_crossing, city=city, trade_dt=target_dt
            )
            tmax_tomorrow = _try_call(
                forecast_tmax_tomorrow, city=city, trade_dt=target_dt, _state=tomorrow_state
            )
            tmax_weatherapi = _try_call(forecast_tmax_weatherapi, city=city, trade_dt=target_dt)
            tmax_google_weather = _try_call(
                forecast_tmax_google_weather, city=city, trade_dt=target_dt
            )
            tmax_openweathermap = _try_call(
                forecast_tmax_openweathermap, city=city, trade_dt=target_dt
            )
            tmax_pirateweather = _try_call(
                forecast_tmax_pirateweather, city=city, trade_dt=target_dt
            )
            tmax_weather_gov = _try_call(forecast_tmax_weather_gov, city=city, trade_dt=target_dt)

            vals = {
                "google-weather": tmax_google_weather,
                "open-meteo": tmax_open_meteo,
                "openweathermap": tmax_openweathermap,
                "pirateweather": tmax_pirateweather,
                "visual-crossing": tmax_visual_crossing,
                "tomorrow": tmax_tomorrow,
                "weatherapi": tmax_weatherapi,
                "weather.gov": tmax_weather_gov,
            }
            available = {k: float(v) for k, v in vals.items() if v is not None}

            # Base weights: learned weights (if present) or uniform over available.
            w_city = _weights_for_city(weights_all, city)
            weights_used: dict[str, float] = {
                k: float(w_city[k]) for k in available.keys() if k in w_city
            }
            if not weights_used:
                if available:
                    u = 1.0 / len(available)
                    weights_used = {k: u for k in available.keys()}
            else:
                s = sum(weights_used.values())
                weights_used = (
                    {k: v / s for k, v in weights_used.items()}
                    if s > 0
                    else ({k: 1.0 / len(available) for k in available.keys()} if available else {})
                )

            # Lead-time tracking: look back at recent pulses for this (city, trade_date)
            # and apply volatility-based dynamic weighting (Consensus 2.0).
            history_rows = _load_recent_intraday_history(
                args.out_csv,
                city=city,
                trade_date=trade_date_str,
                max_rows=4,
            )
            weights_used, stability_score = _apply_volatility_weighting(
                available, weights_used, history_rows
            )

            mean_forecast = (
                sum(weights_used[k] * available[k] for k in weights_used.keys()) if weights_used else None
            )
            sigma = (
                float(statistics.pstdev(list(available.values())))
                if len(available) > 1
                else (0.0 if available else None)
            )

            # 1) Spread-based component (agreement between providers) from current sigma.
            spread_conf_raw = _confidence_from_spread(float(sigma)) if sigma is not None else 0.0

            # 2) Skill-based component from the learned weights.
            skill_conf = _skill_from_weights(weights_used)

            # 3) Combine spread and skill into the final confidence score.
            spread_conf = max(0.0, min(0.9, float(spread_conf_raw)))
            conf_final = spread_conf * (0.5 + 0.5 * skill_conf) if sigma is not None else None

            # 4) Conviction score: blend confidence with stability of recent provider skews.
            conviction_score: float | None
            if conf_final is None:
                conviction_score = None
            else:
                # conf_final is in [0, ~0.9]; stability_score is in [0,1].
                # Rescale so that (high confidence & high stability) ≈ 1.0.
                raw = float(conf_final) * float(stability_score)
                conviction_score = max(0.0, min(1.0, raw / 0.9)) if raw > 0 else 0.0

            ts_now = _now_iso_local()
            row = {
                "timestamp": ts_now,
                "city": city,
                "trade_date": trade_date_str,
                "mean_forecast": "" if mean_forecast is None else f"{mean_forecast:.4f}",
                "current_sigma": "" if sigma is None else f"{sigma:.4f}",
                "tmax_open_meteo": "" if tmax_open_meteo is None else f"{tmax_open_meteo:.4f}",
                "tmax_visual_crossing": "" if tmax_visual_crossing is None else f"{tmax_visual_crossing:.4f}",
                "tmax_tomorrow": "" if tmax_tomorrow is None else f"{tmax_tomorrow:.4f}",
                "tmax_weatherapi": "" if tmax_weatherapi is None else f"{tmax_weatherapi:.4f}",
                "tmax_google_weather": "" if tmax_google_weather is None else f"{tmax_google_weather:.4f}",
                "tmax_openweathermap": "" if tmax_openweathermap is None else f"{tmax_openweathermap:.4f}",
                "tmax_pirateweather": "" if tmax_pirateweather is None else f"{tmax_pirateweather:.4f}",
                "tmax_weather_gov": "" if tmax_weather_gov is None else f"{tmax_weather_gov:.4f}",
                "sources_used": ",".join([s for s in SOURCES_ORDER if s in available]),
                "weights_used": ",".join(
                    [f"{k}:{weights_used[k]:.4f}" for k in sorted(weights_used.keys())]
                )
                if weights_used
                else "",
            }
            intraday_rows.append(row)
            if not args.no_write:
                _append_intraday_row(args.out_csv, row)
                wrote += 1

            # Write predictions for both today and tomorrow so the dashboard always has
            # the next trade date (e.g. after 7 PM ET we show tomorrow; file must have it).
            if args.write_predictions:
                spread_f = sigma if sigma is not None else ""
                conf = "" if conf_final is None else f"{float(conf_final):.4f}"
                conviction_str = (
                    "" if conviction_score is None else f"{float(conviction_score):.4f}"
                )
                pred_rows.append(
                    {
                        "date": trade_date_str,
                        "city": city,
                        "tmax_predicted": "" if mean_forecast is None else f"{mean_forecast:.4f}",
                        "tmax_lstm": "",
                        "tmax_forecast": "" if mean_forecast is None else f"{mean_forecast:.4f}",
                        "spread_f": "" if sigma is None else f"{float(sigma):.4f}",
                        "confidence_score": conf,
                        "conviction_score": conviction_str,
                        "forecast_sources": ",".join([s for s in SOURCES_ORDER if s in available]),
                        "tmax_open_meteo": "" if tmax_open_meteo is None else f"{tmax_open_meteo:.4f}",
                        "tmax_visual_crossing": ""
                        if tmax_visual_crossing is None
                        else f"{tmax_visual_crossing:.4f}",
                        "tmax_tomorrow": "" if tmax_tomorrow is None else f"{tmax_tomorrow:.4f}",
                        "tmax_weatherapi": ""
                        if tmax_weatherapi is None
                        else f"{tmax_weatherapi:.4f}",
                        "tmax_google_weather": ""
                        if tmax_google_weather is None
                        else f"{tmax_google_weather:.4f}",
                        "tmax_openweathermap": ""
                        if tmax_openweathermap is None
                        else f"{tmax_openweathermap:.4f}",
                        "tmax_pirateweather": ""
                        if tmax_pirateweather is None
                        else f"{tmax_pirateweather:.4f}",
                        "tmax_weather_gov": ""
                        if tmax_weather_gov is None
                        else f"{tmax_weather_gov:.4f}",
                        "sources_used": ",".join([s for s in SOURCES_ORDER if s in available]),
                        "weights_used": ",".join(
                            [f"{k}:{weights_used[k]:.4f}" for k in sorted(weights_used.keys())]
                        )
                        if weights_used
                        else "",
                    }
                )

    if args.print:
        if args.print_format == "json":
            payload = {
                "started": started,
                "trade_date": trade_dt.isoformat(),
                "env": args.env,
                "intraday_rows": intraday_rows,
                "prediction_rows": pred_rows,
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            # Human-readable summary.
            print(f"[intraday_pulse.py] started={started} trade_date={trade_dt.isoformat()} env={args.env}")
            for r in intraday_rows:
                city = r.get("city")
                mean_f = r.get("mean_forecast")
                sig = r.get("current_sigma")
                print(f"\n--- {city} ---")
                print(f"mean_forecast={mean_f} sigma={sig}")
                print(f"sources_used={r.get('sources_used')}")
                print(f"weights_used={r.get('weights_used')}")
                # Provider values (compact).
                for k in (
                    "tmax_open_meteo",
                    "tmax_visual_crossing",
                    "tmax_tomorrow",
                    "tmax_weatherapi",
                    "tmax_google_weather",
                    "tmax_openweathermap",
                    "tmax_pirateweather",
                    "tmax_weather_gov",
                ):
                    v = r.get(k, "")
                    if str(v).strip() != "":
                        print(f"{k}={v}")

    if args.write_predictions:
        if not args.no_write:
            _write_predictions_latest(args.predictions_latest, pred_rows)
            _append_predictions_history(
                args.predictions_history,
                pred_rows,
                extra_fields={
                    "run_ts": _now_iso_local(),
                    "env": args.env,
                    "prediction_mode": "forecast",
                    "blend_forecast_weight": "1.0",
                    "refresh_history": "False",
                    "retrain_lstm": "False",
                },
            )

    print(
        f"[intraday_pulse.py] done start={started} trade_date={trade_dt.isoformat()} "
        f"cities_written={wrote if not args.no_write else 0} out={args.out_csv}"
        + (f" predictions_latest={args.predictions_latest}" if (args.write_predictions and not args.no_write) else "")
        + (" (no-write)" if args.no_write else "")
    )

