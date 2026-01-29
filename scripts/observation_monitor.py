import os
import time
import json
import csv
import datetime as dt
import requests
import numpy as np
from dotenv import load_dotenv
from typing import Dict, List, Optional, Any
import subprocess
import sys

load_dotenv()

# Configuration
STATIONS = {
    "ny": "KNYC",
    "il": "KMDW",
    "tx": "KAUS",
    "fl": "KMIA"
}

FETCH_INTERVAL_SEC = 120  # 2 minutes
DATA_DIR = "Data"
LATEST_JSON = os.path.join(DATA_DIR, "observations_latest.json")
HISTORY_CSV = os.path.join(DATA_DIR, "observations_history.csv")
USER_AGENT = os.getenv("NWS_USER_AGENT") or "(weather-trader-bot, contact: larry.liquid@proton.me)"

def calculate_slope(times: np.ndarray, values: np.ndarray) -> float:
    """Calculate slope using linear regression (degrees per hour)."""
    if len(times) < 2:
        return 0.0
    # Times are in seconds, convert to hours for the slope
    x = times / 3600.0
    y = values
    
    # Simple linear regression
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m)

def get_observations(stid: str) -> Optional[List[Dict[str, Any]]]:
    """Fetch recent observations for a station using the official NWS API."""
    url = f"https://api.weather.gov/stations/{stid}/observations"
    params = {"limit": 24} # last 24 observations (usually hourly or more frequent)
    headers = {"User-Agent": USER_AGENT}
    
    try:
        r = requests.get(url, params=params, headers=headers, timeout=20)
        r.raise_for_status()
        data = r.json()
        
        features = data.get("features", [])
        if not features:
            return None
            
        obs_list = []
        for feat in features:
            props = feat.get("properties", {})
            ts_str = props.get("timestamp")
            temp_c = props.get("temperature", {}).get("value")
            
            if ts_str and temp_c is not None:
                temp_f = (float(temp_c) * 9.0 / 5.0) + 32.0
                obs_list.append({
                    "timestamp": ts_str,
                    "temp": temp_f
                })
        
        return obs_list
    except Exception as e:
        print(f"Error fetching data for {stid}: {e}")
        return None

def process_station(city: str, stid: str) -> Optional[Dict[str, Any]]:
    obs_list = get_observations(stid)
    if not obs_list:
        return None
        
    # Sort by timestamp ascending
    obs_list.sort(key=lambda x: x["timestamp"])
    
    # Extract timestamps and temps
    timestamps = [dt.datetime.fromisoformat(o["timestamp"].replace("Z", "+00:00")).timestamp() for o in obs_list]
    temps = np.array([o["temp"] for o in obs_list])
    times = np.array(timestamps)
    
    if len(times) == 0:
        return None
        
    now_ts = times[-1]
    current_temp = temps[-1]
    
    # Calculate trends
    def get_slope_for_window(minutes: int) -> float:
        cutoff = now_ts - (minutes * 60 * 2) # Take 2 hours of data to be safe for 1h trend
        mask = times >= cutoff
        if np.sum(mask) < 2:
            return 0.0
        return calculate_slope(times[mask], temps[mask])

    trend_10m = get_slope_for_window(10)
    trend_30m = get_slope_for_window(30)
    trend_1h = get_slope_for_window(60)
    
    # Calculate acceleration
    acceleration = trend_10m - trend_30m
    
    return {
        "city": city,
        "stid": stid,
        "timestamp": dt.datetime.fromtimestamp(now_ts, tz=dt.timezone.utc).isoformat(),
        "temp": round(current_temp, 2),
        "trend_10m": round(trend_10m, 4),
        "trend_30m": round(trend_30m, 4),
        "trend_1h": round(trend_1h, 4),
        "acceleration": round(acceleration, 4)
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    print(f"Starting observation monitor. Interval: {FETCH_INTERVAL_SEC}s")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    while True:
        results = {}
        for city, stid in STATIONS.items():
            print(f"Processing {city} ({stid})...")
            res = process_station(city, stid)
            if res:
                results[city] = res
        
        if results:
            # Update latest JSON
            with open(LATEST_JSON, "w") as f:
                json.dump({
                    "last_update": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
                    "stations": results
                }, f, indent=2)
            
            # Append to history CSV
            write_header = not os.path.exists(HISTORY_CSV)
            with open(HISTORY_CSV, "a", newline="") as f:
                fieldnames = ["timestamp", "city", "stid", "temp", "trend_10m", "trend_30m", "trend_1h", "acceleration"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                for res in results.values():
                    writer.writerow(res)
                    
            print(f"Updated observations at {dt.datetime.now().isoformat()}")
        
        if args.once:
            break
            
        time.sleep(FETCH_INTERVAL_SEC)

if __name__ == "__main__":
    main()
