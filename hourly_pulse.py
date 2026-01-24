import argparse
import csv
import datetime as dt
import json
import os
import statistics
import time
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

load_dotenv()

try:
    import requests_cache  # type: ignore
except ModuleNotFoundError:
    requests_cache = None


CITIES = ["ny", "il", "tx", "fl"]
LATLON = {
    "ny": (40.79736, -73.97785),
    "il": (41.78701, -87.77166),
    "tx": (30.14440, -97.66876),
    "fl": (25.77380, -80.19360),
}


def _local_tz() -> dt.tzinfo:
    tzname = (os.getenv("TZ") or "").strip() or "America/New_York"
    try:
        return ZoneInfo(tzname)
    except Exception:
        return dt.datetime.now().astimezone().tzinfo or dt.timezone.utc


def _now_iso_local() -> str:
    # ISO 8601 with offset, e.g. 2026-01-24T09:38:51-05:00
    return dt.datetime.now(tz=_local_tz()).isoformat()


def _safe_float(x) -> float | None:
    if x is None:
        return None
    try:
        s = str(x).strip()
    except Exception:
        return None
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _try_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


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
        if k == "lstm":
            continue
        fv = _safe_float(v)
        if fv is None:
            continue
        out[str(k)] = float(fv)
    # Renormalize
    s = sum(max(0.0, v) for v in out.values())
    if s <= 0:
        return {}
    return {k: float(v) / s for k, v in out.items()}


def _append_hourly_row(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path)
    fieldnames = [
        # minimum (plan)
        "timestamp",
        "city",
        "mean_forecast",
        "spread_at_time",
        # recommended extras
        "trade_date",
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
                _migrate_hourly_forecasts_schema(path, fieldnames)
        except Exception:
            # If migration fails, continue append; worst case the row is still preserved.
            pass
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def _migrate_hourly_forecasts_schema(path: str, new_fieldnames: list[str]) -> None:
    """
    Data/hourly_forecasts.csv was historically written with a smaller header; when we add new
    provider columns, appends can create shifted rows. This migration rewrites the file using
    the canonical header.
    """
    old_fieldnames = [
        "timestamp",
        "city",
        "mean_forecast",
        "spread_at_time",
        "trade_date",
        "tmax_open_meteo",
        "tmax_visual_crossing",
        "tmax_tomorrow",
        "tmax_weatherapi",
        "tmax_openweathermap",
        "tmax_pirateweather",
        "tmax_weather_gov",
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
            # best-effort: map the old schema prefix
            d = dict(zip(old_fieldnames, row[: len(old_fieldnames)]))
        out_rows.append(d)

    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=new_fieldnames)
        w.writeheader()
        for d in out_rows:
            w.writerow({k: d.get(k, "") for k in new_fieldnames})
    os.replace(tmp, path)


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
    # fallback if first item matches
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
      - GOOGLE (preferred, per your .env)
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
    p = argparse.ArgumentParser(description="Fetch per-source forecasts and append an hourly history row.")
    p.add_argument("--trade-date", type=str, required=True, help="YYYY-MM-DD (event date the forecasts target)")
    p.add_argument("--out-csv", type=str, default="Data/hourly_forecasts.csv")
    p.add_argument("--weights-json", type=str, default="Data/weights.json")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    trade_dt = dt.datetime.strptime(args.trade_date, "%Y-%m-%d").date()
    weights_all = _load_weights(args.weights_json)
    tomorrow_state: dict[str, float | None] = {"last_req_ts": None}

    started = _now_iso_local()
    wrote = 0
    for city in CITIES:
        # Per-source forecasts
        tmax_open_meteo = _try_call(forecast_tmax_open_meteo, city=city, trade_dt=trade_dt)
        tmax_visual_crossing = _try_call(forecast_tmax_visual_crossing, city=city, trade_dt=trade_dt)
        tmax_tomorrow = _try_call(forecast_tmax_tomorrow, city=city, trade_dt=trade_dt, _state=tomorrow_state)
        tmax_weatherapi = _try_call(forecast_tmax_weatherapi, city=city, trade_dt=trade_dt)
        tmax_google_weather = _try_call(forecast_tmax_google_weather, city=city, trade_dt=trade_dt)
        tmax_openweathermap = _try_call(forecast_tmax_openweathermap, city=city, trade_dt=trade_dt)
        tmax_pirateweather = _try_call(forecast_tmax_pirateweather, city=city, trade_dt=trade_dt)
        tmax_weather_gov = _try_call(forecast_tmax_weather_gov, city=city, trade_dt=trade_dt)

        vals = {
            "open-meteo": tmax_open_meteo,
            "visual-crossing": tmax_visual_crossing,
            "tomorrow": tmax_tomorrow,
            "weatherapi": tmax_weatherapi,
            "google-weather": tmax_google_weather,
            "openweathermap": tmax_openweathermap,
            "pirateweather": tmax_pirateweather,
            "weather.gov": tmax_weather_gov,
        }
        available = {k: float(v) for k, v in vals.items() if v is not None}

        if not available:
            # Still write a row so we can see "no data" periods in the history.
            row = {
                "timestamp": _now_iso_local(),
                "city": city,
                "trade_date": trade_dt.strftime("%Y-%m-%d"),
                "mean_forecast": "",
                "spread_at_time": "",
                "tmax_open_meteo": "" if tmax_open_meteo is None else f"{tmax_open_meteo:.4f}",
                "tmax_visual_crossing": "" if tmax_visual_crossing is None else f"{tmax_visual_crossing:.4f}",
                "tmax_tomorrow": "" if tmax_tomorrow is None else f"{tmax_tomorrow:.4f}",
                "tmax_weatherapi": "" if tmax_weatherapi is None else f"{tmax_weatherapi:.4f}",
                "tmax_google_weather": "" if tmax_google_weather is None else f"{tmax_google_weather:.4f}",
                "tmax_openweathermap": "" if tmax_openweathermap is None else f"{tmax_openweathermap:.4f}",
                "tmax_pirateweather": "" if tmax_pirateweather is None else f"{tmax_pirateweather:.4f}",
                "tmax_weather_gov": "" if tmax_weather_gov is None else f"{tmax_weather_gov:.4f}",
                "sources_used": "",
                "weights_used": "",
            }
            _append_hourly_row(args.out_csv, row)
            wrote += 1
            continue

        # Weights: prefer learned weights when available; otherwise uniform across available sources.
        w = _weights_for_city(weights_all, city)
        weights_used: dict[str, float] = {k: float(w[k]) for k in available.keys() if k in w}
        if not weights_used:
            u = 1.0 / len(available)
            weights_used = {k: u for k in available.keys()}
        else:
            s = sum(weights_used.values())
            weights_used = {k: v / s for k, v in weights_used.items()} if s > 0 else {k: 1.0 / len(available) for k in available.keys()}

        mean_forecast = sum(weights_used[k] * available[k] for k in weights_used.keys())
        spread_at_time = statistics.pstdev(list(available.values())) if len(available) > 1 else 0.0

        row = {
            "timestamp": _now_iso_local(),
            "city": city,
            "trade_date": trade_dt.strftime("%Y-%m-%d"),
            "mean_forecast": f"{mean_forecast:.4f}",
            "spread_at_time": f"{spread_at_time:.4f}",
            "tmax_open_meteo": "" if tmax_open_meteo is None else f"{tmax_open_meteo:.4f}",
            "tmax_visual_crossing": "" if tmax_visual_crossing is None else f"{tmax_visual_crossing:.4f}",
            "tmax_tomorrow": "" if tmax_tomorrow is None else f"{tmax_tomorrow:.4f}",
            "tmax_weatherapi": "" if tmax_weatherapi is None else f"{tmax_weatherapi:.4f}",
            "tmax_google_weather": "" if tmax_google_weather is None else f"{tmax_google_weather:.4f}",
            "tmax_openweathermap": "" if tmax_openweathermap is None else f"{tmax_openweathermap:.4f}",
            "tmax_pirateweather": "" if tmax_pirateweather is None else f"{tmax_pirateweather:.4f}",
            "tmax_weather_gov": "" if tmax_weather_gov is None else f"{tmax_weather_gov:.4f}",
            "sources_used": ",".join(sorted(weights_used.keys())),
            "weights_used": ",".join([f"{k}:{weights_used[k]:.4f}" for k in sorted(weights_used.keys())]),
        }
        _append_hourly_row(args.out_csv, row)
        wrote += 1

    print(
        f"[hourly_pulse.py] done start={started} trade_date={trade_dt.isoformat()} "
        f"cities_written={wrote} out={args.out_csv}"
    )
