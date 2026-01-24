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


# We are intentionally limiting API calls to a small set of providers.
SOURCES_ORDER = ["open-meteo", "visual-crossing", "tomorrow", "weatherapi"]

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


def _confidence_from_spread(spread_f: float) -> float:
    # Match run_daily.py behavior.
    if spread_f <= 1.5:
        return 1.0
    if spread_f >= 3.0:
        return 0.0
    return float((3.0 - float(spread_f)) / (3.0 - 1.5))


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
        "sources_used",
        "weights_used",
    ]
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def _write_predictions_latest(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "date",
        "city",
        "tmax_predicted",
        "spread_f",
        "confidence_score",
        "forecast_sources",
        "tmax_open_meteo",
        "tmax_visual_crossing",
        "tmax_tomorrow",
        "tmax_weatherapi",
        "sources_used",
        "weights_used",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _append_predictions_history(path: str, latest_rows: list[dict], *, extra_fields: dict[str, str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path)
    base_fields = list((latest_rows[0].keys()) if latest_rows else [])
    out_fields = base_fields + [k for k in extra_fields.keys() if k not in base_fields]
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        if write_header:
            w.writeheader()
        for r in latest_rows:
            rr = dict(r)
            rr.update(extra_fields)
            w.writerow(rr)


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


def _parse_args():
    p = argparse.ArgumentParser(description="Fetch limited-provider forecasts at specific intraday times.")
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
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    trade_dt = dt.datetime.strptime(args.trade_date, "%Y-%m-%d").date()
    weights_all = _load_weights(args.weights_json)
    tomorrow_state: dict[str, float | None] = {"last_req_ts": None}

    started = _now_iso_local()
    wrote = 0
    pred_rows: list[dict] = []

    for city in CITIES:
        tmax_open_meteo = _try_call(forecast_tmax_open_meteo, city=city, trade_dt=trade_dt)
        tmax_visual_crossing = _try_call(forecast_tmax_visual_crossing, city=city, trade_dt=trade_dt)
        tmax_tomorrow = _try_call(forecast_tmax_tomorrow, city=city, trade_dt=trade_dt, _state=tomorrow_state)
        tmax_weatherapi = _try_call(forecast_tmax_weatherapi, city=city, trade_dt=trade_dt)

        vals = {
            "open-meteo": tmax_open_meteo,
            "visual-crossing": tmax_visual_crossing,
            "tomorrow": tmax_tomorrow,
            "weatherapi": tmax_weatherapi,
        }
        available = {k: float(v) for k, v in vals.items() if v is not None}

        w_city = _weights_for_city(weights_all, city)
        weights_used: dict[str, float] = {k: float(w_city[k]) for k in available.keys() if k in w_city}
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

        mean_forecast = (
            sum(weights_used[k] * available[k] for k in weights_used.keys()) if weights_used else None
        )
        sigma = float(statistics.pstdev(list(available.values()))) if len(available) > 1 else (0.0 if available else None)

        row = {
            "timestamp": _now_iso_local(),
            "city": city,
            "trade_date": trade_dt.strftime("%Y-%m-%d"),
            "mean_forecast": "" if mean_forecast is None else f"{mean_forecast:.4f}",
            "current_sigma": "" if sigma is None else f"{sigma:.4f}",
            "tmax_open_meteo": "" if tmax_open_meteo is None else f"{tmax_open_meteo:.4f}",
            "tmax_visual_crossing": "" if tmax_visual_crossing is None else f"{tmax_visual_crossing:.4f}",
            "tmax_tomorrow": "" if tmax_tomorrow is None else f"{tmax_tomorrow:.4f}",
            "tmax_weatherapi": "" if tmax_weatherapi is None else f"{tmax_weatherapi:.4f}",
            "sources_used": ",".join([s for s in SOURCES_ORDER if s in available]),
            "weights_used": ",".join([f"{k}:{weights_used[k]:.4f}" for k in sorted(weights_used.keys())]) if weights_used else "",
        }
        _append_intraday_row(args.out_csv, row)
        wrote += 1

        if args.write_predictions:
            spread_f = sigma if sigma is not None else ""
            conf = "" if sigma is None else f"{_confidence_from_spread(float(sigma)):.4f}"
            pred_rows.append(
                {
                    "date": trade_dt.strftime("%Y-%m-%d"),
                    "city": city,
                    "tmax_predicted": "" if mean_forecast is None else f"{mean_forecast:.4f}",
                    "spread_f": "" if sigma is None else f"{float(sigma):.4f}",
                    "confidence_score": conf,
                    "forecast_sources": ",".join([s for s in SOURCES_ORDER if s in available]),
                    "tmax_open_meteo": "" if tmax_open_meteo is None else f"{tmax_open_meteo:.4f}",
                    "tmax_visual_crossing": "" if tmax_visual_crossing is None else f"{tmax_visual_crossing:.4f}",
                    "tmax_tomorrow": "" if tmax_tomorrow is None else f"{tmax_tomorrow:.4f}",
                    "tmax_weatherapi": "" if tmax_weatherapi is None else f"{tmax_weatherapi:.4f}",
                    "sources_used": ",".join([s for s in SOURCES_ORDER if s in available]),
                    "weights_used": ",".join([f"{k}:{weights_used[k]:.4f}" for k in sorted(weights_used.keys())]) if weights_used else "",
                }
            )

    if args.write_predictions:
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
        f"cities_written={wrote} out={args.out_csv}"
        + (f" predictions_latest={args.predictions_latest}" if args.write_predictions else "")
    )

