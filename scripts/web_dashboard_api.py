"""
Single process: background observation fetch + FastAPI API + dashboard at /.
Run: python scripts/web_dashboard_api.py
Then open http://localhost:8080/
"""
import os
import csv
import json
import datetime as dt
import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from typing import List, Dict, Any, Optional
from zoneinfo import ZoneInfo
import sys

# Project root on path for kalshi_trader
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
import numpy as np
from dotenv import load_dotenv
from astral import LocationInfo
from astral.sun import sun

load_dotenv()

# Per-city lat/lon and timezone for astral (same order as STATIONS: ny, il, tx, fl)
CITY_LATLON_TZ: Dict[str, Dict[str, Any]] = {
    "ny": {"lat": 40.79736, "lon": -73.97785, "tz": "America/New_York"},
    "il": {"lat": 41.78701, "lon": -87.77166, "tz": "America/Chicago"},
    "tx": {"lat": 30.1444, "lon": -97.66876, "tz": "America/Chicago"},
    "fl": {"lat": 25.7738, "lon": -80.1936, "tz": "America/New_York"},
}

# Lazy import so Kalshi code only loads when trading endpoints are hit
kalshi_trader = None

def _kalshi():
    global kalshi_trader
    if kalshi_trader is None:
        import kalshi_trader as kt
        kalshi_trader = kt
    return kalshi_trader

# --- Observation fetch (from observation_monitor, runs in background) ---
STATIONS = {"ny": "KNYC", "il": "KMDW", "tx": "KAUS", "fl": "KMIA"}
FETCH_INTERVAL_SEC = 120
DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
OBSERVATIONS_JSON = DATA_DIR / "observations_latest.json"
OBSERVATIONS_HISTORY_CSV = DATA_DIR / "observations_history.csv"
NWS_USER_AGENT = os.getenv("NWS_USER_AGENT") or "(weather-trader-bot, contact: larry.liquid@proton.me)"

def _calculate_slope(times: np.ndarray, values: np.ndarray) -> float:
    if len(times) < 2:
        return 0.0
    x = times / 3600.0
    y = values
    A = np.vstack([x, np.ones(len(x))]).T
    m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m)

def _get_observations(stid: str) -> Optional[List[Dict[str, Any]]]:
    url = f"https://api.weather.gov/stations/{stid}/observations"
    try:
        r = requests.get(url, params={"limit": 24}, headers={"User-Agent": NWS_USER_AGENT}, timeout=20)
        r.raise_for_status()
        features = r.json().get("features", [])
        if not features:
            return None
        out = []
        for feat in features:
            props = feat.get("properties", {})
            ts = props.get("timestamp")
            temp_c = props.get("temperature", {}).get("value")
            if ts and temp_c is not None:
                temp_f = (float(temp_c) * 9.0 / 5.0) + 32.0
                out.append({"timestamp": ts, "temp": temp_f})
        return out
    except Exception:
        return None

def _process_station(city: str, stid: str) -> Optional[Dict[str, Any]]:
    obs_list = _get_observations(stid)
    if not obs_list:
        return None
    obs_list.sort(key=lambda x: x["timestamp"])
    timestamps = [dt.datetime.fromisoformat(o["timestamp"].replace("Z", "+00:00")).timestamp() for o in obs_list]
    temps = np.array([o["temp"] for o in obs_list])
    times = np.array(timestamps)
    if len(times) == 0:
        return None
    now_ts = times[-1]
    current_temp = temps[-1]

    def slope_for(minutes: int) -> float:
        cutoff = now_ts - (minutes * 60 * 2)
        mask = times >= cutoff
        if np.sum(mask) < 2:
            return 0.0
        return _calculate_slope(times[mask], temps[mask])

    trend_10m = slope_for(10)
    trend_30m = slope_for(30)
    trend_1h = slope_for(60)
    return {
        "city": city,
        "stid": stid,
        "timestamp": dt.datetime.fromtimestamp(now_ts, tz=dt.timezone.utc).isoformat(),
        "temp": round(current_temp, 2),
        "trend_10m": round(trend_10m, 4),
        "trend_30m": round(trend_30m, 4),
        "trend_1h": round(trend_1h, 4),
        "acceleration": round(trend_10m - trend_30m, 4),
    }

def _run_observation_fetch() -> None:
    """Sync fetch; call from thread or asyncio.to_thread."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for city, stid in STATIONS.items():
        res = _process_station(city, stid)
        if res:
            results[city] = res
    if not results:
        return
    payload = {"last_update": dt.datetime.now(tz=dt.timezone.utc).isoformat(), "stations": results}
    with open(OBSERVATIONS_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    write_header = not OBSERVATIONS_HISTORY_CSV.exists()
    fieldnames = ["timestamp", "city", "stid", "temp", "trend_10m", "trend_30m", "trend_1h", "acceleration"]
    with open(OBSERVATIONS_HISTORY_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for r in results.values():
            w.writerow(r)

# --- FastAPI app ---
TRADES_CSV = DATA_DIR / "trades_history.csv"
PREDICTIONS_LATEST = DATA_DIR / "predictions_latest.csv"
INTRADAY_CSV = DATA_DIR / "intraday_forecasts.csv"
DASHBOARD_HTML_PATH = Path(__file__).resolve().parent / "dashboard_web.html"

app = FastAPI(title="Weather Trader")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_local_tz():
    tzname = (os.getenv("TZ") or "").strip() or "America/New_York"
    try:
        return ZoneInfo(tzname)
    except Exception:
        return dt.timezone.utc

def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))

async def observation_loop():
    """Background: fetch observations every FETCH_INTERVAL_SEC."""
    loop = asyncio.get_event_loop()
    while True:
        await asyncio.to_thread(_run_observation_fetch)
        await asyncio.sleep(FETCH_INTERVAL_SEC)

@app.on_event("startup")
async def startup():
    # First fetch immediately so / has data
    await asyncio.to_thread(_run_observation_fetch)
    asyncio.create_task(observation_loop())

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the dashboard HTML (same process, no extra server)."""
    if not DASHBOARD_HTML_PATH.exists():
        raise HTTPException(status_code=404, detail="dashboard_web.html not found")
    return HTMLResponse(content=DASHBOARD_HTML_PATH.read_text())

@app.get("/api/status")
async def get_status():
    return {"status": "ok", "timestamp": dt.datetime.now(dt.timezone.utc).isoformat()}

@app.get("/api/observations")
async def get_observations():
    if not OBSERVATIONS_JSON.exists():
        return {"stations": {}, "last_update": None}
    return json.loads(OBSERVATIONS_JSON.read_text())


@app.get("/api/sun")
async def get_sun():
    """Per-city solar position via astral: sunrise, solar noon, warming progress (0–100)."""
    today = dt.date.today()
    out: Dict[str, Dict[str, Any]] = {}
    for city, cfg in CITY_LATLON_TZ.items():
        tz = ZoneInfo(cfg["tz"])
        loc = LocationInfo(city, "", cfg["tz"], cfg["lat"], cfg["lon"])
        try:
            s = sun(loc.observer, date=today, tzinfo=tz)
        except Exception:
            out[city] = {"progress": 0, "status": "Unknown", "past_peak": False}
            continue
        sunrise = s["sunrise"]
        noon = s["noon"]
        # Typical high temp is ~1–3h after solar noon
        high_time = noon + dt.timedelta(hours=2)
        now = dt.datetime.now(tz)
        if now < sunrise:
            progress = 0.0
            status = "Before sunrise"
            past_peak = False
        elif now >= high_time:
            progress = 100.0
            status = "Post-peak (High set)"
            past_peak = True
        else:
            start_ts = sunrise.timestamp()
            end_ts = high_time.timestamp()
            progress = 100.0 * (now.timestamp() - start_ts) / (end_ts - start_ts)
            progress = max(0.0, min(100.0, progress))
            if progress > 80:
                status = "Approaching Daily High"
            else:
                status = "Warming phase"
            past_peak = False
        out[city] = {
            "progress": round(progress, 1),
            "status": status,
            "past_peak": past_peak,
            "sunrise": sunrise.isoformat(),
            "noon": noon.isoformat(),
            "high_time": high_time.isoformat(),
        }
    return {"by_city": out}

def _event_ticker_from_market(ticker: str) -> str:
    """KXHIGHNY-26JAN29-B21.5 -> KXHIGHNY-26JAN29"""
    if "-" in ticker:
        parts = ticker.split("-")
        if len(parts) >= 2:
            return f"{parts[0]}-{parts[1]}"
    return ticker

def _series_to_city() -> dict:
    kt = _kalshi()
    return {v: k for k, v in kt.SERIES_TICKERS.items()}

@app.get("/api/positions")
async def get_positions(env: str = "prod"):
    """Live positions from Kalshi with current prices and familiar names. Requires Kalshi credentials."""
    try:
        client = _kalshi_client(env)
    except HTTPException as e:
        return {"positions": [], "source": "none", "error": e.detail if hasattr(e, "detail") else "Kalshi credentials not configured"}
    try:
        raw = _kalshi().get_portfolio_positions(client, count_filter="position", limit=100)
    except Exception as e:
        return {"positions": [], "source": "kalshi", "error": str(e)}
    market_positions = raw.get("market_positions") or []
    series_to_city = _series_to_city()
    out = []
    for mp in market_positions:
        ticker = mp.get("ticker") or ""
        position = int(mp.get("position") or 0)
        if position == 0:
            continue
        event_ticker = _event_ticker_from_market(ticker)
        series = event_ticker.split("-")[0] if "-" in event_ticker else ""
        city = series_to_city.get(series, "")
        market_exposure_dollars = mp.get("market_exposure_dollars") or "0"
        position_fp = mp.get("position_fp") or str(position)
        row = {
            "ticker": ticker,
            "event_ticker": event_ticker,
            "city": city or series,
            "position": position,
            "position_fp": position_fp,
            "market_exposure_dollars": market_exposure_dollars,
            "realized_pnl_dollars": mp.get("realized_pnl_dollars") or "0",
        }
        try:
            m = _kalshi().get_market(client, ticker)
            row["market_subtitle"] = m.get("subtitle") or m.get("title") or ticker
            row["yes_bid"] = m.get("yes_bid")
            row["yes_ask"] = m.get("yes_ask")
            if row["yes_ask"] is None and m.get("no_bid") is not None:
                row["yes_ask"] = 100 - int(m["no_bid"])
        except Exception:
            row["market_subtitle"] = ticker
            row["yes_bid"] = None
            row["yes_ask"] = None
        out.append(row)
    return {"positions": out, "source": "kalshi"}

@app.get("/api/predictions")
async def get_predictions():
    return read_csv(PREDICTIONS_LATEST)


@app.get("/api/intraday")
async def get_intraday():
    """Latest mean_forecast per (city, trade_date) from intraday_forecasts.csv for bet context."""
    rows = read_csv(INTRADAY_CSV)
    if not rows:
        return {"by_city_date": {}, "rows": []}
    # Keep latest timestamp per (city, trade_date)
    key_to_row: Dict[tuple, Dict[str, Any]] = {}
    for r in rows:
        city = (r.get("city") or "").strip()
        trade_date = (r.get("trade_date") or "").strip()
        ts_str = r.get("timestamp") or ""
        if not city or not trade_date:
            continue
        try:
            ts = dt.datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
        except Exception:
            ts = 0.0
        key = (city, trade_date)
        if key not in key_to_row or ts > float((key_to_row[key].get("_ts") or 0)):
            mean_f = r.get("mean_forecast")
            sigma = r.get("current_sigma")
            key_to_row[key] = {
                "city": city,
                "trade_date": trade_date,
                "mean_forecast": float(mean_f) if mean_f and str(mean_f).strip() else None,
                "current_sigma": float(sigma) if sigma and str(sigma).strip() else None,
                "timestamp": ts_str,
                "_ts": ts,
            }
    out_rows = []
    for v in key_to_row.values():
        v.pop("_ts", None)
        out_rows.append(v)
    out_rows.sort(key=lambda x: (x["trade_date"], x["city"]))
    by_city_date: Dict[str, Dict[str, Any]] = {}
    for r in out_rows:
        by_city_date.setdefault(r["city"], {})[r["trade_date"]] = {
            "mean_forecast": r["mean_forecast"],
            "current_sigma": r["current_sigma"],
            "timestamp": r["timestamp"],
        }
    return {"by_city_date": by_city_date, "rows": out_rows}


def _kalshi_client(env: str = "prod"):
    api_key_id = os.getenv("KALSHI_API_KEY_ID") or os.getenv("API_KEY")
    private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH") or os.getenv("KALSHI_PRIVATE_KEY_FILE")
    if not api_key_id:
        raise HTTPException(status_code=500, detail="Kalshi credentials not configured in .env")
    # When not in container, env may point at /run/secrets/kalshi_key.pem; use project-root gooony.txt
    if not private_key_path or not Path(private_key_path).exists():
        fallback = Path(__file__).resolve().parent.parent / "gooony.txt"
        if fallback.exists():
            private_key_path = str(fallback)
        elif not private_key_path:
            raise HTTPException(status_code=500, detail="Kalshi private key not configured. Set KALSHI_PRIVATE_KEY_PATH or add gooony.txt to project root.")
        else:
            raise HTTPException(status_code=500, detail=f"Kalshi key file not found: {private_key_path} (when not in Docker, use gooony.txt in project root)")
    return _kalshi().KalshiHttpClient(env=env, api_key_id=api_key_id, private_key_path=private_key_path)

@app.get("/api/market/orderbook/{ticker}")
async def get_orderbook(ticker: str, env: str = "prod"):
    client = _kalshi_client(env)
    try:
        ob = _kalshi().get_market_orderbook(client, ticker)
        prices = _kalshi()._best_yes_prices_from_orderbook(ob)
        return {"ticker": ticker, "orderbook": ob, "prices": prices}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/trade/order")
async def place_order(
    ticker: str = Body(...),
    side: str = Body(...),
    action: str = Body(...),
    count: int = Body(...),
    price: int = Body(...),
    env: str = Body("prod"),
):
    client = _kalshi_client(env)
    order = {
        "ticker": ticker,
        "side": side,
        "action": action,
        "count": int(count),
        "type": "limit",
        "client_order_id": str(_kalshi().uuid.uuid4()),
    }
    if side == "yes":
        order["yes_price"] = int(price)
    else:
        order["no_price"] = int(price)
    try:
        status_resp = client.get("/trade-api/v2/exchange/status")
        if status_resp.status_code == 200 and not status_resp.json().get("trading_active", False):
            raise HTTPException(status_code=400, detail="Exchange trading is not active")
        resp = client.post("/trade-api/v2/portfolio/orders", order)
        if resp.status_code != 201:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return {"status": "success", "order": resp.json()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
