import argparse
import base64
import datetime
import math
import os
import csv
import re
import uuid
import json
import statistics

import requests
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from dotenv import load_dotenv

load_dotenv()


def _base_url(env_name: str) -> str:
    env_name = (env_name or "").strip().lower()
    if env_name in ("prod", "production"):
        # Per Kalshi docs/examples (production)
        return "https://api.elections.kalshi.com"
    # Per Kalshi docs (demo)
    return "https://demo-api.kalshi.co"


def load_private_key_from_file(file_path: str) -> rsa.RSAPrivateKey:
    with open(file_path, "rb") as key_file:
        private_key = serialization.load_pem_private_key(
            key_file.read(), password=None, backend=default_backend()
        )
    if not isinstance(private_key, rsa.RSAPrivateKey):
        raise TypeError("Private key is not an RSA private key")
    return private_key


def sign_pss_text(private_key: rsa.RSAPrivateKey, text: str) -> str:
    message = text.encode("utf-8")
    signature = private_key.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("utf-8")


def kalshi_headers(
    private_key: rsa.RSAPrivateKey, api_key_id: str, method: str, path: str
) -> dict:
    # IMPORTANT (Kalshi docs): when signing, strip query parameters from the path.
    path_without_query = path.split("?")[0]
    timestamp_ms = str(int(datetime.datetime.now().timestamp() * 1000))
    msg = timestamp_ms + method.upper() + path_without_query
    sig = sign_pss_text(private_key, msg)
    return {
        "KALSHI-ACCESS-KEY": api_key_id,
        "KALSHI-ACCESS-SIGNATURE": sig,
        "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
    }


class KalshiHttpClient:
    def __init__(self, *, env: str, api_key_id: str, private_key_path: str, base_url: str | None = None):
        self.base = (base_url or "").strip() or os.getenv("KALSHI_BASE_URL", "").strip() or _base_url(env)
        self.api_key_id = api_key_id
        self.private_key = load_private_key_from_file(private_key_path)

    def get(self, path: str, *, timeout_s: int = 30) -> requests.Response:
        headers = kalshi_headers(self.private_key, self.api_key_id, "GET", path)
        return requests.get(self.base + path, headers=headers, timeout=timeout_s)

    def post(self, path: str, data: dict, *, timeout_s: int = 30) -> requests.Response:
        headers = kalshi_headers(self.private_key, self.api_key_id, "POST", path)
        headers["Content-Type"] = "application/json"
        return requests.post(self.base + path, headers=headers, json=data, timeout=timeout_s)


SERIES_TICKERS = {
    # Default to the KX* series, which exist in the Kalshi demo environment.
    # Override via env vars if you want different series in prod.
    "ny": os.getenv("KALSHI_SERIES_NY", "KXHIGHNY"),
    "il": os.getenv("KALSHI_SERIES_IL", "KXHIGHCHI"),
    "tx": os.getenv("KALSHI_SERIES_TX", "KXHIGHAUS"),
    "fl": os.getenv("KALSHI_SERIES_FL", "KXHIGHMIA"),
}
CITY_ORDER = ["ny", "il", "tx", "fl"]


def get_event(client: KalshiHttpClient, event_ticker: str) -> dict:
    path = f"/trade-api/v2/events/{event_ticker}?with_nested_markets=true"
    resp = client.get(path)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch event {event_ticker}: {resp.status_code} {resp.text}")
    payload = resp.json()
    # Depending on API versioning, markets may live in payload["event"]["markets"]
    return payload


def get_market_orderbook(client: KalshiHttpClient, market_ticker: str, *, depth: int = 25) -> dict:
    depth = int(depth)
    if depth < 0:
        depth = 0
    if depth > 100:
        depth = 100
    path = f"/trade-api/v2/markets/{market_ticker}/orderbook?depth={depth}"
    resp = client.get(path)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Failed to fetch orderbook for {market_ticker}: {resp.status_code} {resp.text}"
        )
    return resp.json()


def get_market(client: KalshiHttpClient, market_ticker: str) -> dict:
    path = f"/trade-api/v2/markets/{market_ticker}"
    resp = client.get(path)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch market {market_ticker}: {resp.status_code} {resp.text}")
    payload = resp.json()
    return payload.get("market") or payload


def _load_city_metadata_mae(path: str) -> dict[str, float]:
    """
    Load Data/city_metadata.json and return city->historical_MAE (degrees F).
    Supported shapes:
      - {"cities": {"il": {"historical_MAE": 2.1}, ...}}
      - {"il": {"historical_MAE": 2.1}, ...}
      - {"il": 2.1, ...}
    """
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            payload = json.load(f) or {}
    except Exception:
        return {}

    cities = payload.get("cities") if isinstance(payload, dict) else None
    if isinstance(cities, dict):
        out: dict[str, float] = {}
        for city, meta in cities.items():
            if isinstance(meta, (int, float)):
                out[str(city)] = float(meta)
                continue
            if not isinstance(meta, dict):
                continue
            for k in ("historical_MAE", "historical_MAE_f", "historical_mae", "historical_mae_f"):
                if k in meta:
                    try:
                        out[str(city)] = float(meta[k])
                    except Exception:
                        pass
                    break
        return out

    if isinstance(payload, dict):
        out: dict[str, float] = {}
        for city, meta in payload.items():
            if city in ("as_of", "window_days", "updated_at", "source"):
                continue
            if isinstance(meta, (int, float)):
                out[str(city)] = float(meta)
                continue
            if isinstance(meta, dict):
                for k in ("historical_MAE", "historical_MAE_f", "historical_mae", "historical_mae_f"):
                    if k in meta:
                        try:
                            out[str(city)] = float(meta[k])
                        except Exception:
                            pass
                        break
        return out

    return {}


def get_temporal_stability(
    *,
    city: str,
    trade_date: str,
    hourly_csv: str = "Data/hourly_forecasts.csv",
    window: int = 6,
) -> dict[str, float | int | None] | None:
    """
    Compute temporal stability (jitter) and drift from the hourly forecast history.

    Jitter (temporal_stability): stddev of mean_forecast over the last N hourly samples.
    Drift: last_mean - first_mean over the same window.
    """
    if not hourly_csv or not os.path.exists(hourly_csv):
        return None
    window = max(1, int(window))

    samples: list[tuple[datetime.datetime, float]] = []
    try:
        with open(hourly_csv, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                if not row:
                    continue
                if (row.get("city") or "").strip() != (city or "").strip():
                    continue
                if (row.get("trade_date") or "").strip() != (trade_date or "").strip():
                    continue
                mf = row.get("mean_forecast")
                mf_f = None if mf in (None, "") else float(mf)
                if mf_f is None:
                    continue
                ts = (row.get("timestamp") or "").strip()
                if not ts:
                    continue
                try:
                    t = datetime.datetime.fromisoformat(ts)
                except Exception:
                    continue
                if t.tzinfo is None:
                    # Treat naive timestamps as UTC for safety.
                    t = t.replace(tzinfo=datetime.timezone.utc)
                samples.append((t, float(mf_f)))
    except Exception:
        return None

    if not samples:
        return None

    samples.sort(key=lambda x: x[0])
    samples = samples[-window:]
    means = [m for _, m in samples]
    n = len(means)
    if n < 2:
        return {"n": n, "temporal_stability": None, "drift": None, "first_mean": means[0], "last_mean": means[0]}

    jitter = float(statistics.pstdev(means))
    drift = float(means[-1] - means[0])
    return {
        "n": n,
        "temporal_stability": jitter,
        "drift": drift,
        "first_mean": float(means[0]),
        "last_mean": float(means[-1]),
    }


def get_balance(client: KalshiHttpClient) -> dict:
    """
    Fetch portfolio balance. Per Kalshi docs, values are returned in cents.

    We try to extract an "available cash" number for sizing.
    """
    path = "/trade-api/v2/portfolio/balance"
    resp = client.get(path)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch balance: {resp.status_code} {resp.text}")
    payload = resp.json() or {}

    # Common shapes across SDKs/versions:
    # - {"balance": {"available_cash": 12345, "portfolio_value": 23456, ...}}
    # - {"balance": 12345, "portfolio_value": 23456, ...}
    bal = payload.get("balance")
    if isinstance(bal, dict):
        return {
            "available_cash_cents": bal.get("available_cash")
            if bal.get("available_cash") is not None
            else bal.get("availableCash"),
            "portfolio_value_cents": bal.get("portfolio_value")
            if bal.get("portfolio_value") is not None
            else bal.get("portfolioValue"),
            "raw": payload,
        }
    return {
        "available_cash_cents": payload.get("available_cash")
        if payload.get("available_cash") is not None
        else payload.get("availableCash"),
        "portfolio_value_cents": payload.get("portfolio_value")
        if payload.get("portfolio_value") is not None
        else payload.get("portfolioValue"),
        "balance_cents": bal,
        "raw": payload,
    }


def _best_yes_prices_from_orderbook(orderbook_payload: dict) -> dict[str, float | int | None]:
    """
    Kalshi orderbook returns bids only. Use reciprocal relationship:
    - YES ask = 100 - best NO bid
    - YES bid = best YES bid
    """
    ob = (orderbook_payload or {}).get("orderbook") or {}
    yes = ob.get("yes") or []
    no = ob.get("no") or []

    def _price_qty(level) -> tuple[int | None, int | None]:
        try:
            price = int(level[0])
        except Exception:
            return (None, None)
        qty = None
        try:
            qty = int(level[1])
        except Exception:
            qty = None
        return (price, qty)

    best_yes_bid = None
    best_yes_bid_qty = None
    if yes:
        p, q = _price_qty(yes[-1])
        best_yes_bid, best_yes_bid_qty = p, q

    best_no_bid = None
    best_no_bid_qty = None
    if no:
        p, q = _price_qty(no[-1])
        best_no_bid, best_no_bid_qty = p, q

    yes_ask = None
    if best_no_bid is not None:
        yes_ask = 100 - int(best_no_bid)

    yes_spread = None
    if best_yes_bid is not None and yes_ask is not None:
        yes_spread = int(yes_ask) - int(best_yes_bid)

    # Liquidity at the implied ask: contracts resting at best NO bid (reciprocal).
    ask_qty = None
    if best_no_bid is not None:
        total = 0
        for lvl in reversed(no):
            p, q = _price_qty(lvl)
            if p is None:
                continue
            if int(p) != int(best_no_bid):
                break
            if q is not None:
                total += int(q)
        ask_qty = total

    return {
        "best_yes_bid": best_yes_bid,
        "best_yes_bid_qty": best_yes_bid_qty,
        "best_no_bid": best_no_bid,
        "best_no_bid_qty": best_no_bid_qty,
        "yes_ask": yes_ask,
        "yes_spread": yes_spread,
        "ask_qty": ask_qty,
    }


def get_yes_pricing(
    client: KalshiHttpClient,
    market_ticker: str,
    *,
    orderbook_depth: int,
    fallback_qty: int,
) -> tuple[dict[str, float | int | None], str]:
    """
    Return (pricing, source) where pricing contains:
    best_yes_bid, yes_ask, yes_spread, ask_qty, best_no_bid.

    Prefer orderbook. If orderbook has no actionable bids, fall back to market endpoint.
    """
    try:
        ob = get_market_orderbook(client, market_ticker, depth=orderbook_depth)
        px = _best_yes_prices_from_orderbook(ob)
        if px.get("yes_ask") is not None:
            return (px, "orderbook")
    except Exception:
        pass

    m = get_market(client, market_ticker)
    yes_ask = m.get("yes_ask")
    yes_bid = m.get("yes_bid")
    no_bid = m.get("no_bid")

    # If yes_ask missing but no_bid exists, infer ask via reciprocity.
    if yes_ask is None and no_bid is not None:
        try:
            yes_ask = 100 - int(no_bid)
        except Exception:
            yes_ask = None

    px = {
        "best_yes_bid": yes_bid,
        "best_yes_bid_qty": None,
        "best_no_bid": no_bid,
        "best_no_bid_qty": None,
        "yes_ask": yes_ask,
        "yes_spread": (None if yes_ask is None or yes_bid is None else int(yes_ask) - int(yes_bid)),
        "ask_qty": int(fallback_qty),
    }
    return (px, "market")


# for seriesTicker in seriesTickers:
#   temps = getTempRanges(seriesTicker)


# Function to determine the interval
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _parse_subtitle_to_range(subtitle: str) -> tuple[float | None, float | None]:
    """
    Convert a Kalshi temperature subtitle into a numeric range.

    Handles patterns like:
    - '15° or below'  -> (None, 15)
    - '40° or above'  -> (40, None)
    - '18° to 19°'    -> (18, 19)
    - '67° to 68°'    -> (67, 68)

    Returns (min_f, max_f) where either bound can be None to indicate +/- infinity.
    """
    s = (subtitle or "").strip().lower()
    nums = [float(x) for x in _NUM_RE.findall(s)]
    if not nums:
        return (None, None)
    if "or below" in s:
        return (None, nums[0])
    if "or above" in s:
        return (nums[0], None)
    if len(nums) >= 2:
        lo, hi = nums[0], nums[1]
        if hi < lo:
            lo, hi = hi, lo
        return (lo, hi)
    return (None, None)


def find_interval(value: float, intervals: list[dict]) -> int:
    """
    Pick the market bucket for a predicted temperature.

    Strategy:
    - Prefer buckets whose range contains value.
    - If multiple contain (boundary overlap), prefer the narrowest range.
    - If none contain (parsing gaps), pick the closest bucket by distance to range.
    """
    parsed: list[tuple[int, float | None, float | None, str]] = []
    for i, m in enumerate(intervals):
        subtitle = (m.get("subtitle") or "") if isinstance(m, dict) else ""
        lo, hi = _parse_subtitle_to_range(subtitle)
        parsed.append((i, lo, hi, subtitle))

    containing: list[tuple[float, int]] = []
    for i, lo, hi, _ in parsed:
        if lo is None and hi is None:
            continue
        lo_ok = True if lo is None else (value >= lo)
        hi_ok = True if hi is None else (value <= hi)
        if lo_ok and hi_ok:
            width = float("inf") if (lo is None or hi is None) else (hi - lo)
            containing.append((width, i))
    if containing:
        containing.sort(key=lambda t: (t[0], t[1]))
        return containing[0][1]

    # No containing bucket found: choose closest to a finite range.
    best = (float("inf"), 0)  # (distance, index)
    for i, lo, hi, _ in parsed:
        if lo is None and hi is None:
            continue
        if lo is None:
            dist = max(0.0, value - float(hi)) if hi is not None else float("inf")
        elif hi is None:
            dist = max(0.0, float(lo) - value)
        else:
            if value < lo:
                dist = lo - value
            elif value > hi:
                dist = value - hi
            else:
                dist = 0.0
        if dist < best[0]:
            best = (dist, i)
    return best[1]


def _norm_cdf(x: float, *, mu: float, sigma: float) -> float:
    if sigma <= 1e-9:
        return 1.0 if x >= mu else 0.0
    z = (x - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def bucket_probability(*, lo: float | None, hi: float | None, mu: float, sigma: float) -> float:
    """
    Probability that X ~ Normal(mu, sigma) lands in the bucket described by (lo, hi).
    lo/hi are inclusive bounds; for continuous distributions, boundary inclusivity is negligible.
    """
    if sigma <= 1e-9:
        if lo is not None and mu < lo:
            return 0.0
        if hi is not None and mu > hi:
            return 0.0
        return 1.0
    if lo is None and hi is None:
        return 0.0
    if lo is None:
        return max(0.0, min(1.0, _norm_cdf(float(hi), mu=mu, sigma=sigma)))
    if hi is None:
        return max(0.0, min(1.0, 1.0 - _norm_cdf(float(lo), mu=mu, sigma=sigma)))
    a = _norm_cdf(float(lo), mu=mu, sigma=sigma)
    b = _norm_cdf(float(hi), mu=mu, sigma=sigma)
    return max(0.0, min(1.0, b - a))


def _append_eval_row(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path)
    fieldnames = [
        "run_ts",
        "env",
        "trade_date",
        "city",
        "series_ticker",
        "event_ticker",
        "decision",
        "reason",
        "mu_tmax_f",
        "sigma_f",
        "spread_f",
        "confidence_score",
        "tmax_open_meteo",
        "tmax_visual_crossing",
        "tmax_tomorrow",
        "tmax_weatherapi",
        "tmax_lstm",
        "sources_used",
        "weights_used",
        "chosen_market_ticker",
        "chosen_market_subtitle",
        "bucket_lo",
        "bucket_hi",
        "model_prob_yes",
        "yes_ask",
        "yes_bid",
        "yes_spread",
        "ask_qty",
        "market_prob_yes",
        "edge_prob",
        "ev_cents",
        "count",
        "send_orders",
    ]
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def _append_decision(path: str, row: dict) -> None:
    # Keep the existing decisions_history.csv schema stable (reason is free-form).
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path)
    fieldnames = [
        "run_ts",
        "env",
        "trade_date",
        "city",
        "series_ticker",
        "event_ticker",
        "pred_tmax_f",
        "spread_f",
        "confidence_score",
        "decision",
        "reason",
    ]
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def _already_traded(*, trades_log: str | None, env: str, trade_date: str, city: str) -> bool:
    """
    Idempotency guard: return True if we have already placed (or attempted) a live trade
    for this env+trade_date+city.

    We only treat rows with send_orders==True as "already traded" so dry-runs don't block live runs.
    """
    if not trades_log:
        return False
    if not os.path.exists(trades_log):
        return False
    try:
        with open(trades_log, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                if (row.get("env") or "").strip() != (env or "").strip():
                    continue
                if (row.get("trade_date") or "").strip() != (trade_date or "").strip():
                    continue
                if (row.get("city") or "").strip() != (city or "").strip():
                    continue
                s = (row.get("send_orders") or "").strip().lower()
                if s in ("1", "true", "yes", "y"):
                    return True
    except Exception:
        # Best-effort: if we can't read it, don't block.
        return False
    return False


def _load_allocation_scores(
    *,
    metrics_csv: str,
    trade_dt: datetime.date,
    window_days: int,
) -> dict[str, float]:
    """
    Compute per-city allocation scores using historical feedback (best-effort).

    We use `Data/daily_metrics.csv` rows produced by the nightly settle/metrics job:
    - consensus MAE (metric_type=mae_f, source_name=consensus)
    - bucket hit rate (metric_type=bucket_hit_rate, source_name=trade)

    Score is higher when MAE is lower and hit-rate is higher.
    """
    if not metrics_csv or not os.path.exists(metrics_csv):
        return {}

    start = trade_dt - datetime.timedelta(days=int(window_days))
    end = trade_dt - datetime.timedelta(days=1)

    # Aggregate over the window.
    mae_sum: dict[str, float] = {}
    mae_n: dict[str, int] = {}
    hit_sum: dict[str, float] = {}
    hit_n: dict[str, int] = {}

    with open(metrics_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            d = (row.get("trade_date") or "").strip()
            city = (row.get("city") or "").strip()
            mtype = (row.get("metric_type") or "").strip()
            src = (row.get("source_name") or "").strip()
            v = (row.get("value") or "").strip()
            if not d or not city or not mtype or not src or not v:
                continue
            try:
                dd = datetime.datetime.strptime(d, "%Y-%m-%d").date()
            except Exception:
                continue
            if dd < start or dd > end:
                continue
            try:
                val = float(v)
            except Exception:
                continue

            if mtype == "mae_f" and src == "consensus":
                mae_sum[city] = mae_sum.get(city, 0.0) + val
                mae_n[city] = mae_n.get(city, 0) + 1
            if mtype == "bucket_hit_rate" and src == "trade":
                hit_sum[city] = hit_sum.get(city, 0.0) + val
                hit_n[city] = hit_n.get(city, 0) + 1

    scores: dict[str, float] = {}
    for city in CITY_ORDER:
        # Default to neutral if we have no history yet.
        mae = (mae_sum.get(city, 0.0) / mae_n[city]) if mae_n.get(city) else None
        hit = (hit_sum.get(city, 0.0) / hit_n[city]) if hit_n.get(city) else None

        # MAE term: smaller MAE => bigger score. Keep bounded and stable.
        # Example: mae=1 => ~0.5; mae=3 => ~0.1.
        mae_term = 1.0
        if mae is not None and mae >= 0.0:
            mae_term = 1.0 / (1.0 + float(mae) * float(mae))

        # Hit term: 0..1. If missing, neutral 1.0.
        hit_term = 1.0
        if hit is not None:
            hit_term = 0.5 + max(0.0, min(1.0, float(hit)))  # 0.5..1.5

        scores[city] = float(mae_term) * float(hit_term)
    return scores


def make_trade(
    *,
    client: KalshiHttpClient,
    pred: float,
    markets: list[dict],
    send_orders: bool,
    count: int,
    side: str,
    yes_price: int,
    no_price: int,
    trade_dt_str: str,
    city: str,
    series: str,
    event_ticker: str,
    trades_log: str | None,
    env: str,
):
    # Caller may pass either the full event markets list (legacy) or a single chosen market.
    # In the EV-based selection flow, we pass a list of length 1 containing the chosen market.
    chosen = markets[0] if markets else None
    if not chosen:
        raise RuntimeError("make_trade called with empty markets list")

    subtitle = chosen.get("subtitle", "")
    ticker = chosen.get("ticker")
    print(f"Pred_mean={pred:.2f} → chosen_bucket='{subtitle}' → market={ticker}")

    # NOTE: trades_log is written only when a trade is allowed to proceed.

    if trades_log:
        os.makedirs(os.path.dirname(trades_log) or ".", exist_ok=True)
        write_header = not os.path.exists(trades_log)
        with open(trades_log, "a", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "run_ts",
                    "env",
                    "trade_date",
                    "city",
                    "series_ticker",
                    "event_ticker",
                    "market_ticker",
                    "market_subtitle",
                    "pred_tmax_f",
                    "side",
                    "count",
                    "yes_price",
                    "no_price",
                    "send_orders",
                ],
            )
            if write_header:
                w.writeheader()
            w.writerow(
                {
                    "run_ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "env": env,
                    "trade_date": trade_dt_str,
                    "city": city,
                    "series_ticker": series,
                    "event_ticker": event_ticker,
                    "market_ticker": ticker,
                    "market_subtitle": subtitle,
                    "pred_tmax_f": f"{pred:.4f}",
                    "side": side,
                    "count": int(count),
                    "yes_price": int(yes_price),
                    "no_price": int(no_price),
                    "send_orders": bool(send_orders),
                }
            )

    if not send_orders:
        print(
            f"DRY RUN: would submit order action=buy side={side} count={count} "
            f"yes_price={yes_price} no_price={no_price}"
        )
        return

    # Check exchange status before sending
    status_resp = client.get("/trade-api/v2/exchange/status")
    if status_resp.status_code != 200:
        raise RuntimeError(
            f"Failed to fetch exchange status: {status_resp.status_code} {status_resp.text}"
        )
    status = status_resp.json()
    if not status.get("trading_active", False):
        print("Exchange trading is not active; refusing to send order.")
        return

    order = {
        "ticker": ticker,
        "side": side,
        "action": "buy",
        "count": int(count),
        "type": "limit",
        "client_order_id": str(uuid.uuid4()),
    }
    if side == "yes":
        order["yes_price"] = int(yes_price)
    else:
        order["no_price"] = int(no_price)

    resp = client.post("/trade-api/v2/portfolio/orders", order)
    if resp.status_code != 201:
        raise RuntimeError(f"Order failed: {resp.status_code} {resp.text}")
    print(f"Order submitted: {ticker}")


def _parse_args():
    p = argparse.ArgumentParser(description="Trade Kalshi HIGH{CITY} markets from predictions.")
    p.add_argument("--trade-date", type=str, default=None, help="YYYY-MM-DD (default: today)")
    p.add_argument(
        "--predictions-csv",
        type=str,
        default="predictions_final.csv",
        help="Predictions CSV written by daily_prediction.py",
    )
    p.add_argument(
        "--env",
        type=str,
        default=os.getenv("KALSHI_ENV", "demo"),
        help="Kalshi env: demo or prod (default from KALSHI_ENV, else demo)",
    )
    p.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Override API base URL (e.g. https://demo-api.kalshi.co or https://api.elections.kalshi.com)",
    )
    p.add_argument(
        "--api-key-id",
        type=str,
        default=None,
        help="Kalshi API key ID (overrides env vars)",
    )
    p.add_argument(
        "--private-key-path",
        type=str,
        default=None,
        help="Path to RSA private key PEM (overrides env vars)",
    )
    p.add_argument(
        "--send-orders",
        action="store_true",
        help="Actually submit orders (default is dry-run)",
    )
    p.add_argument(
        "--count",
        type=int,
        default=0,
        help="If >0, fixed contracts per order; if 0, auto-size up to city budget (default 0).",
    )
    p.add_argument("--side", type=str, default="yes", choices=["yes", "no"])
    # For orderbook-aware selection, yes-price is treated as a ceiling for the implied ask.
    p.add_argument("--yes-price", type=int, default=99, help="Max acceptable YES ask in cents (1-99)")
    p.add_argument("--no-price", type=int, default=99, help="Limit price in cents (1-99)")
    p.add_argument(
        "--trades-log",
        type=str,
        default=None,
        help="Append trade intents to this CSV (useful for run_daily.py).",
    )
    p.add_argument(
        "--decisions-log",
        type=str,
        default=None,
        help="Append trade decisions (including skips) to this CSV.",
    )
    # Safe defaults to match the VotingModel guardrails.
    p.add_argument("--min-confidence", type=float, default=0.5, help="Skip if confidence_score < this")
    p.add_argument("--max-spread", type=float, default=3.0, help="Skip if spread_f > this")
    p.add_argument("--eval-log", type=str, default=None, help="Append evaluation rows to this CSV.")

    p.add_argument("--orderbook-depth", type=int, default=25)
    # sigma is now computed per city as max(current_spread, historical_MAE), with sigma_floor as fallback
    p.add_argument("--sigma-floor", type=float, default=2.0, help="Fallback sigma when city metadata is unavailable")
    p.add_argument("--sigma-mult", type=float, default=1.0, help="Legacy (ignored when city metadata is present)")
    p.add_argument("--city-metadata-json", type=str, default="Data/city_metadata.json")
    p.add_argument("--min-ev-cents", type=float, default=3.0)
    p.add_argument("--max-yes-spread-cents", type=float, default=6.0)
    p.add_argument("--min-ask-depth", type=int, default=25)
    p.add_argument("--max-dollars-per-city", type=float, default=50.0)
    p.add_argument("--max-dollars-total", type=float, default=150.0)
    p.add_argument("--max-contracts-per-order", type=int, default=500)
    p.add_argument(
        "--max-balance-fraction",
        type=float,
        default=0.5,
        help="Hard cap: never spend more than this fraction of available cash balance per run (default 0.5).",
    )
    p.add_argument(
        "--daily-metrics-csv",
        type=str,
        default="Data/daily_metrics.csv",
        help="Historical metrics used for per-city budget allocation.",
    )
    p.add_argument(
        "--allocation-window-days",
        type=int,
        default=14,
        help="Lookback window (days) for allocation scoring from daily_metrics.csv.",
    )
    p.add_argument(
        "--allocation-mode",
        type=str,
        default="learned",
        choices=["learned", "equal"],
        help="How to split the per-run cap across cities. 'learned' uses confidence+history, 'equal' splits evenly.",
    )
    p.add_argument(
        "--allocation-min-city-fraction",
        type=float,
        default=0.05,
        help="Minimum fraction of per-run cap reserved per city (prevents zero budgets).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    trade_dt = (
        datetime.datetime.strptime(args.trade_date, "%Y-%m-%d").date()
        if args.trade_date
        else datetime.date.today()
    )
    trade_dt_str = trade_dt.strftime("%Y-%m-%d")
    event_suffix = trade_dt.strftime("%y%b%d").upper()

    api_key_id = (
        args.api_key_id
        or os.getenv("KALSHI_API_KEY_ID")
        or os.getenv("KALSHI_API_KEY")
        or os.getenv("KALSHI_KEY_ID")
        or os.getenv("API_KEY_ID")
        or os.getenv("API_KEY")
    )
    private_key_path = (
        args.private_key_path
        or os.getenv("KALSHI_PRIVATE_KEY_PATH")
        or os.getenv("KALSHI_PRIVATE_KEY_FILE")
        or os.getenv("PRIVATE_KEY_PATH")
        or os.getenv("PRIVATE_KEY_FILE")
    )
    if not api_key_id:
        raise RuntimeError("Missing KALSHI_API_KEY_ID (or KALSHI_API_KEY) in environment/.env")
    if not private_key_path:
        raise RuntimeError("Missing KALSHI_PRIVATE_KEY_PATH in environment/.env")

    client = KalshiHttpClient(
        env=args.env,
        api_key_id=api_key_id,
        private_key_path=private_key_path,
        base_url=args.base_url,
    )

    # Fetch balance once per run and enforce the "never bet more than half your balance" rule.
    available_cash_dollars = None
    try:
        bal = get_balance(client)
        cash_cents = bal.get("available_cash_cents")
        if cash_cents is None:
            # Fallback if payload uses a different field name.
            cash_cents = bal.get("balance_cents")
        if cash_cents is not None:
            available_cash_dollars = float(cash_cents) / 100.0
    except Exception as e:
        print(f"WARNING: could not fetch Kalshi balance; using configured caps only ({e})")

    configured_total_cap = float(args.max_dollars_total)
    balance_cap = None
    if available_cash_dollars is not None and float(args.max_balance_fraction) > 0:
        balance_cap = max(0.0, float(args.max_balance_fraction) * float(available_cash_dollars))
    effective_total_cap = configured_total_cap if balance_cap is None else min(configured_total_cap, balance_cap)
    if available_cash_dollars is not None:
        print(
            f"Balance: available_cash=${available_cash_dollars:.2f} "
            f"→ per-run cap=${effective_total_cap:.2f} (min(configured=${configured_total_cap:.2f}, "
            f"{args.max_balance_fraction:.2f}*balance))"
        )
    else:
        print(f"Per-run cap=${effective_total_cap:.2f} (configured; balance unavailable)")

    # Load historical MAE per city (used for sigma).
    city_mae = _load_city_metadata_mae(args.city_metadata_json)

    # Allocate per-city budgets.
    if str(args.allocation_mode).lower() == "equal":
        city_scores: dict[str, float] = {c: 1.0 for c in CITY_ORDER}
    else:
        # learned: use historical feedback (daily_metrics.csv) + today's confidence
        hist_scores = _load_allocation_scores(
            metrics_csv=args.daily_metrics_csv,
            trade_dt=trade_dt,
            window_days=int(args.allocation_window_days),
        )
        city_scores = {c: float(hist_scores.get(c, 1.0)) for c in CITY_ORDER}

    preds: dict[tuple[str, str], dict] = {}
    with open(args.predictions_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            d = (row.get("date") or "").strip()
            c = (row.get("city") or "").strip()
            t = row.get("tmax_predicted")
            if not d or not c or t is None:
                continue
            try:
                row["_tmax_predicted"] = float(t)
                preds[(d, c)] = row
            except ValueError:
                continue

    if args.side != "yes":
        raise RuntimeError("This build is configured for YES-only trade selection. Use --side yes.")

    spent_total_dollars = 0.0
    spent_by_city: dict[str, float] = {c: 0.0 for c in CITY_ORDER}

    # Blend in today's confidence_score into allocation weights.
    for city in CITY_ORDER:
        key = (trade_dt_str, city)
        row = preds.get(key)
        conf = None
        if row is not None:
            try:
                conf = float(row.get("confidence_score")) if row.get("confidence_score") not in (None, "") else None
            except Exception:
                conf = None
        # Missing/low confidence shouldn't zero the city's budget; it should just downweight it.
        # Clamp to a floor so we still consider trades unless the guardrails explicitly skip them.
        if conf is None:
            conf_term = 1.0
        else:
            conf_term = max(0.0, min(1.0, float(conf)))
        conf_term = max(0.25, float(conf_term))
        city_scores[city] = float(city_scores.get(city, 1.0)) * float(conf_term)

    s = sum(max(0.0, v) for v in city_scores.values())
    if s <= 0:
        city_budgets = {c: effective_total_cap / len(CITY_ORDER) for c in CITY_ORDER}
    else:
        city_budgets = {c: effective_total_cap * max(0.0, city_scores[c]) / s for c in CITY_ORDER}

    # Ensure each city has a small non-zero cap (prevents accidental starvation due to missing confidence/metrics).
    min_city = max(0.0, float(args.allocation_min_city_fraction) * float(effective_total_cap))
    if min_city > 0:
        # First, bump any city below the minimum.
        bumped = {c: max(float(city_budgets.get(c, 0.0)), min_city) for c in CITY_ORDER}
        total_bumped = sum(bumped.values())
        if total_bumped <= effective_total_cap + 1e-9:
            city_budgets = bumped
        else:
            # If minimums would exceed the cap, fall back to equal split.
            city_budgets = {c: effective_total_cap / len(CITY_ORDER) for c in CITY_ORDER}
    print(
        "City budget caps: "
        + ", ".join([f"{c}=${city_budgets.get(c, 0.0):.2f}" for c in CITY_ORDER])
        + f" (window_days={int(args.allocation_window_days)})"
    )

    for city in CITY_ORDER:
        series = SERIES_TICKERS[city]
        event_ticker = f"{series}-{event_suffix}"
        print(f"\n----------- {series} / city={city} / trade_date={trade_dt_str} -----------")

        # Idempotency: one live trade per city per date.
        if bool(args.send_orders) and _already_traded(
            trades_log=args.trades_log,
            env=args.env,
            trade_date=trade_dt_str,
            city=city,
        ):
            print("SKIP: already traded for this city/date (idempotency)")
            if args.decisions_log:
                _append_decision(
                    args.decisions_log,
                    {
                        "run_ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "env": args.env,
                        "trade_date": trade_dt_str,
                        "city": city,
                        "series_ticker": series,
                        "event_ticker": event_ticker,
                        "pred_tmax_f": "",
                        "spread_f": "",
                        "confidence_score": "",
                        "decision": "skip",
                        "reason": "already_traded",
                    },
                )
            if args.eval_log:
                _append_eval_row(
                    args.eval_log,
                    {
                        "run_ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "env": args.env,
                        "trade_date": trade_dt_str,
                        "city": city,
                        "series_ticker": series,
                        "event_ticker": event_ticker,
                        "decision": "skip",
                        "reason": "already_traded",
                        "mu_tmax_f": "",
                        "sigma_f": "",
                        "spread_f": "",
                        "confidence_score": "",
                        "tmax_open_meteo": "",
                        "tmax_visual_crossing": "",
                        "tmax_tomorrow": "",
                        "tmax_weatherapi": "",
                        "tmax_lstm": "",
                        "sources_used": "",
                        "weights_used": "",
                        "chosen_market_ticker": "",
                        "chosen_market_subtitle": "",
                        "bucket_lo": "",
                        "bucket_hi": "",
                        "model_prob_yes": "",
                        "yes_ask": "",
                        "yes_bid": "",
                        "yes_spread": "",
                        "ask_qty": "",
                        "market_prob_yes": "",
                        "edge_prob": "",
                        "ev_cents": "",
                        "count": "",
                        "send_orders": bool(args.send_orders),
                    },
                )
            continue

        key = (trade_dt_str, city)
        if key not in preds:
            raise RuntimeError(
                f"No prediction found for date={trade_dt_str} city={city} in {args.predictions_csv}"
            )
        row = preds[key]
        pred = float(row["_tmax_predicted"])

        spread_f = None
        confidence = None
        try:
            spread_f = float(row.get("spread_f")) if row.get("spread_f") not in (None, "") else None
        except Exception:
            spread_f = None
        try:
            confidence = float(row.get("confidence_score")) if row.get("confidence_score") not in (None, "") else None
        except Exception:
            confidence = None

        # Enforce guardrail before doing any API lookups for the event.
        if confidence is not None and confidence < args.min_confidence:
            print(f"SKIP: confidence_score={confidence:.3f} < {args.min_confidence}")
            if args.decisions_log:
                _append_decision(
                    args.decisions_log,
                    {
                        "run_ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "env": args.env,
                        "trade_date": trade_dt_str,
                        "city": city,
                        "series_ticker": series,
                        "event_ticker": event_ticker,
                        "pred_tmax_f": f"{pred:.4f}",
                        "spread_f": "" if spread_f is None else f"{spread_f:.4f}",
                        "confidence_score": "" if confidence is None else f"{confidence:.4f}",
                        "decision": "skip",
                        "reason": f"confidence<{args.min_confidence}",
                    },
                )
            continue
        if spread_f is not None and spread_f > args.max_spread:
            print(f"SKIP: spread_f={spread_f:.3f} > {args.max_spread}")
            if args.decisions_log:
                _append_decision(
                    args.decisions_log,
                    {
                        "run_ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "env": args.env,
                        "trade_date": trade_dt_str,
                        "city": city,
                        "series_ticker": series,
                        "event_ticker": event_ticker,
                        "pred_tmax_f": f"{pred:.4f}",
                        "spread_f": "" if spread_f is None else f"{spread_f:.4f}",
                        "confidence_score": "" if confidence is None else f"{confidence:.4f}",
                        "decision": "skip",
                        "reason": f"spread>{args.max_spread}",
                    },
                )
            continue

        # Temporal Stability Engine gate: only trade when the forecast is stable (low jitter).
        stability_window = 6
        st = get_temporal_stability(city=city, trade_date=trade_dt_str, window=stability_window)
        if st is None or st.get("n", 0) is None or int(st.get("n") or 0) < int(stability_window) or st.get("temporal_stability") is None:
            n = 0 if st is None else int(st.get("n") or 0)
            print(f"SKIP: insufficient hourly forecast history for stability check (n={n} < {stability_window})")
            if args.decisions_log:
                _append_decision(
                    args.decisions_log,
                    {
                        "run_ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "env": args.env,
                        "trade_date": trade_dt_str,
                        "city": city,
                        "series_ticker": series,
                        "event_ticker": event_ticker,
                        "pred_tmax_f": f"{pred:.4f}",
                        "spread_f": "" if spread_f is None else f"{spread_f:.4f}",
                        "confidence_score": "" if confidence is None else f"{confidence:.4f}",
                        "decision": "skip",
                        "reason": f"insufficient_hourly_forecasts;n={n};need={int(stability_window)}",
                    },
                )
            continue

        jitter = float(st.get("temporal_stability") or 0.0)
        drift = float(st.get("drift") or 0.0)
        threshold = float(city_mae.get(city, 1.5))
        if jitter > threshold:
            print(f"SKIP: high forecast jitter (jitter={jitter:.3f}F > threshold={threshold:.3f}F, drift={drift:+.3f}F)")
            if args.decisions_log:
                _append_decision(
                    args.decisions_log,
                    {
                        "run_ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "env": args.env,
                        "trade_date": trade_dt_str,
                        "city": city,
                        "series_ticker": series,
                        "event_ticker": event_ticker,
                        "pred_tmax_f": f"{pred:.4f}",
                        "spread_f": "" if spread_f is None else f"{spread_f:.4f}",
                        "confidence_score": "" if confidence is None else f"{confidence:.4f}",
                        "decision": "skip",
                        "reason": f"high_forecast_jitter;jitter={jitter:.4f};threshold={threshold:.4f};drift={drift:.4f};n={int(st.get('n') or 0)}",
                    },
                )
            continue

        event_payload = get_event(client, event_ticker)
        markets = (event_payload.get("event") or {}).get("markets") or event_payload.get("markets") or []
        if not markets:
            raise RuntimeError(f"No markets returned for event {event_ticker}")

        # New sigma: max(current_spread, historical_MAE). If metadata missing, fall back to sigma_floor.
        cur_spread = float(spread_f or 0.0)
        hist_mae = city_mae.get(city)
        if hist_mae is None:
            sigma = max(float(args.sigma_floor), float(args.sigma_mult) * cur_spread)
        else:
            sigma = max(cur_spread, float(hist_mae))
        best = None  # (ev_cents, market_dict, pricing_dict, prob, lo, hi, edge_prob, market_prob)
        best_any = None  # same tuple as best, but without EV/spread/ceiling filters
        for m in markets:
            subtitle = (m.get("subtitle") or "") if isinstance(m, dict) else ""
            lo, hi = _parse_subtitle_to_range(subtitle)
            if lo is None and hi is None:
                continue
            p_yes = bucket_probability(lo=lo, hi=hi, mu=float(pred), sigma=float(sigma))
            if p_yes <= 0.0:
                continue

            ticker = m.get("ticker")
            if not ticker:
                continue

            px, px_src = get_yes_pricing(
                client,
                ticker,
                orderbook_depth=args.orderbook_depth,
                fallback_qty=min(int(args.max_contracts_per_order), 5),
            )
            px["pricing_source"] = px_src
            yes_ask = px.get("yes_ask")
            yes_spread = px.get("yes_spread")
            ask_qty = px.get("ask_qty")
            yes_bid = px.get("best_yes_bid")

            if yes_ask is None:
                continue
            yes_ask = int(yes_ask)
            yes_spread = None if yes_spread is None else int(yes_spread)

            if yes_ask < 1 or yes_ask > 99:
                continue
            if yes_spread is not None and yes_spread < 0:
                continue
            if ask_qty is None or int(ask_qty) <= 0:
                continue

            market_prob = yes_ask / 100.0
            edge_prob = float(p_yes) - float(market_prob)
            ev_cents = float(100.0 * float(p_yes) - float(yes_ask))
            cand = (ev_cents, m, px, float(p_yes), lo, hi, float(edge_prob), float(market_prob))

            if best_any is None or cand[0] > best_any[0]:
                best_any = cand

            # Apply conservative filters for actual trading
            if yes_ask > int(args.yes_price):
                continue
            if yes_spread is not None and yes_spread > int(args.max_yes_spread_cents):
                continue
            if ev_cents < float(args.min_ev_cents):
                continue

            if best is None or cand[0] > best[0]:
                best = cand

        if best is None:
            print("SKIP: no qualifying market passed EV/spread/liquidity filters")
            reason = "no_qualifying_market"
            if best_any is not None:
                ev_c, bm, bpx, bp, blo, bhi, bedge, bmprob = best_any
                bt = bm.get("ticker")
                bask = int(bpx.get("yes_ask") or 0)
                bspr = int(bpx.get("yes_spread") or 0)
                bqty = int(bpx.get("ask_qty") or 0)
                src = bpx.get("pricing_source") or ""
                reason = f"no_qualifying_market;best={bt};ask={bask};spr={bspr};qty={bqty};ev={ev_c:.2f};px={src}"
            if args.decisions_log:
                _append_decision(
                    args.decisions_log,
                    {
                        "run_ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "env": args.env,
                        "trade_date": trade_dt_str,
                        "city": city,
                        "series_ticker": series,
                        "event_ticker": event_ticker,
                        "pred_tmax_f": f"{pred:.4f}",
                        "spread_f": "" if spread_f is None else f"{spread_f:.4f}",
                        "confidence_score": "" if confidence is None else f"{confidence:.4f}",
                        "decision": "skip",
                        "reason": reason,
                    },
                )
            if args.eval_log:
                chosen_ticker = ""
                chosen_subtitle = ""
                bucket_lo = ""
                bucket_hi = ""
                model_prob = ""
                yes_ask = ""
                yes_bid = ""
                yes_spread = ""
                ask_qty = ""
                market_prob = ""
                edge_prob = ""
                ev_cents = ""
                if best_any is not None:
                    ev_c, bm, bpx, bp, blo, bhi, bedge, bmprob = best_any
                    chosen_ticker = bm.get("ticker") or ""
                    chosen_subtitle = bm.get("subtitle") or ""
                    bucket_lo = "" if blo is None else f"{blo:.4f}"
                    bucket_hi = "" if bhi is None else f"{bhi:.4f}"
                    model_prob = f"{bp:.6f}"
                    yes_ask = int(bpx.get("yes_ask") or 0)
                    yes_bid = "" if bpx.get("best_yes_bid") is None else int(bpx.get("best_yes_bid"))
                    yes_spread = int(bpx.get("yes_spread") or 0)
                    ask_qty = int(bpx.get("ask_qty") or 0)
                    market_prob = f"{bmprob:.6f}"
                    edge_prob = f"{bedge:.6f}"
                    ev_cents = f"{ev_c:.4f}"
                _append_eval_row(
                    args.eval_log,
                    {
                        "run_ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "env": args.env,
                        "trade_date": trade_dt_str,
                        "city": city,
                        "series_ticker": series,
                        "event_ticker": event_ticker,
                        "decision": "skip",
                        "reason": reason,
                        "mu_tmax_f": f"{pred:.4f}",
                        "sigma_f": f"{sigma:.4f}",
                        "spread_f": "" if spread_f is None else f"{spread_f:.4f}",
                        "confidence_score": "" if confidence is None else f"{confidence:.4f}",
                        "tmax_open_meteo": row.get("tmax_open_meteo", ""),
                        "tmax_visual_crossing": row.get("tmax_visual_crossing", ""),
                        "tmax_tomorrow": row.get("tmax_tomorrow", ""),
                        "tmax_weatherapi": row.get("tmax_weatherapi", ""),
                        "tmax_lstm": row.get("tmax_lstm", ""),
                        "sources_used": row.get("sources_used", ""),
                        "weights_used": row.get("weights_used", ""),
                        "chosen_market_ticker": chosen_ticker,
                        "chosen_market_subtitle": chosen_subtitle,
                        "bucket_lo": bucket_lo,
                        "bucket_hi": bucket_hi,
                        "model_prob_yes": model_prob,
                        "yes_ask": yes_ask,
                        "yes_bid": yes_bid,
                        "yes_spread": yes_spread,
                        "ask_qty": ask_qty,
                        "market_prob_yes": market_prob,
                        "edge_prob": edge_prob,
                        "ev_cents": ev_cents,
                        "send_orders": bool(args.send_orders),
                    },
                )
            continue

        ev_cents, chosen_market, px, p_yes, lo, hi, edge_prob, market_prob = best
        ticker = chosen_market.get("ticker")
        subtitle = chosen_market.get("subtitle", "")
        yes_ask = int(px.get("yes_ask") or 0)
        yes_spread = int(px.get("yes_spread") or 0)
        yes_bid = px.get("best_yes_bid")
        ask_qty = int(px.get("ask_qty") or 0)

        remaining_total = max(0.0, float(effective_total_cap) - float(spent_total_dollars))
        remaining_city_budget = max(0.0, float(city_budgets.get(city, 0.0)) - float(spent_by_city.get(city, 0.0)))
        max_city = min(float(args.max_dollars_per_city), remaining_total, remaining_city_budget)
        cost_per_contract = yes_ask / 100.0
        count_cap_budget = int(max_city // cost_per_contract) if cost_per_contract > 0 else 0

        desired = int(args.count) if int(args.count) > 0 else 10**9
        count = min(desired, int(args.max_contracts_per_order), ask_qty, count_cap_budget)
        # Price impact cap: don't take more than 25% of the resting best-ask size.
        max_take = max(1, int(math.floor(float(ask_qty) * 0.25)))
        count = min(int(count), int(max_take))
        if count < 1:
            print("SKIP: budget/liquidity caps reduce count to 0")
            if args.decisions_log:
                _append_decision(
                    args.decisions_log,
                    {
                        "run_ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "env": args.env,
                        "trade_date": trade_dt_str,
                        "city": city,
                        "series_ticker": series,
                        "event_ticker": event_ticker,
                        "pred_tmax_f": f"{pred:.4f}",
                        "spread_f": "" if spread_f is None else f"{spread_f:.4f}",
                        "confidence_score": "" if confidence is None else f"{confidence:.4f}",
                        "decision": "skip",
                        "reason": "count_zero_after_caps",
                    },
                )
            if args.eval_log:
                _append_eval_row(
                    args.eval_log,
                    {
                        "run_ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "env": args.env,
                        "trade_date": trade_dt_str,
                        "city": city,
                        "series_ticker": series,
                        "event_ticker": event_ticker,
                        "decision": "skip",
                        "reason": "count_zero_after_caps",
                        "mu_tmax_f": f"{pred:.4f}",
                        "sigma_f": f"{sigma:.4f}",
                        "spread_f": "" if spread_f is None else f"{spread_f:.4f}",
                        "confidence_score": "" if confidence is None else f"{confidence:.4f}",
                        "tmax_open_meteo": row.get("tmax_open_meteo", ""),
                        "tmax_visual_crossing": row.get("tmax_visual_crossing", ""),
                        "tmax_tomorrow": row.get("tmax_tomorrow", ""),
                        "tmax_weatherapi": row.get("tmax_weatherapi", ""),
                        "tmax_lstm": row.get("tmax_lstm", ""),
                        "sources_used": row.get("sources_used", ""),
                        "weights_used": row.get("weights_used", ""),
                        "chosen_market_ticker": ticker,
                        "chosen_market_subtitle": subtitle,
                        "bucket_lo": "" if lo is None else f"{lo:.4f}",
                        "bucket_hi": "" if hi is None else f"{hi:.4f}",
                        "model_prob_yes": f"{p_yes:.6f}",
                        "yes_ask": yes_ask,
                        "yes_bid": "" if yes_bid is None else int(yes_bid),
                        "yes_spread": yes_spread,
                        "ask_qty": ask_qty,
                        "market_prob_yes": f"{market_prob:.6f}",
                        "edge_prob": f"{edge_prob:.6f}",
                        "ev_cents": f"{ev_cents:.4f}",
                        "count": count,
                        "send_orders": bool(args.send_orders),
                    },
                )
            continue

        reason = ""
        if ask_qty < int(args.min_ask_depth):
            reason = f"low_ask_depth({ask_qty}<{int(args.min_ask_depth)})"

        if args.decisions_log:
            _append_decision(
                args.decisions_log,
                {
                    "run_ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "env": args.env,
                    "trade_date": trade_dt_str,
                    "city": city,
                    "series_ticker": series,
                    "event_ticker": event_ticker,
                    "pred_tmax_f": f"{pred:.4f}",
                    "spread_f": "" if spread_f is None else f"{spread_f:.4f}",
                    "confidence_score": "" if confidence is None else f"{confidence:.4f}",
                    "decision": "trade",
                    "reason": (
                        (reason + ";" if reason else "")
                        + f"stability_ok;jitter={jitter:.4f};threshold={threshold:.4f};drift={drift:.4f};n={int(st.get('n') or 0)};"
                        + f"ticker={ticker};ask={yes_ask};ev={ev_cents:.2f}"
                    ),
                },
            )
        if args.eval_log:
            _append_eval_row(
                args.eval_log,
                {
                    "run_ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "env": args.env,
                    "trade_date": trade_dt_str,
                    "city": city,
                    "series_ticker": series,
                    "event_ticker": event_ticker,
                    "decision": "trade",
                    "reason": reason,
                    "mu_tmax_f": f"{pred:.4f}",
                    "sigma_f": f"{sigma:.4f}",
                    "spread_f": "" if spread_f is None else f"{spread_f:.4f}",
                    "confidence_score": "" if confidence is None else f"{confidence:.4f}",
                    "tmax_open_meteo": row.get("tmax_open_meteo", ""),
                    "tmax_visual_crossing": row.get("tmax_visual_crossing", ""),
                    "tmax_tomorrow": row.get("tmax_tomorrow", ""),
                    "tmax_weatherapi": row.get("tmax_weatherapi", ""),
                    "tmax_lstm": row.get("tmax_lstm", ""),
                    "sources_used": row.get("sources_used", ""),
                    "weights_used": row.get("weights_used", ""),
                    "chosen_market_ticker": ticker,
                    "chosen_market_subtitle": subtitle,
                    "bucket_lo": "" if lo is None else f"{lo:.4f}",
                    "bucket_hi": "" if hi is None else f"{hi:.4f}",
                    "model_prob_yes": f"{p_yes:.6f}",
                    "yes_ask": yes_ask,
                    "yes_bid": "" if yes_bid is None else int(yes_bid),
                    "yes_spread": yes_spread,
                    "ask_qty": ask_qty,
                    "market_prob_yes": f"{market_prob:.6f}",
                    "edge_prob": f"{edge_prob:.6f}",
                    "ev_cents": f"{ev_cents:.4f}",
                    "count": count,
                    "send_orders": bool(args.send_orders),
                },
            )

        make_trade(
            client=client,
            pred=pred,
            markets=[chosen_market],
            send_orders=args.send_orders,
            count=count,
            side="yes",
            yes_price=yes_ask,
            no_price=args.no_price,
            trade_dt_str=trade_dt_str,
            city=city,
            series=series,
            event_ticker=event_ticker,
            trades_log=args.trades_log,
            env=args.env,
        )

        spent_total_dollars += float(count) * float(cost_per_contract)
        spent_by_city[city] = float(spent_by_city.get(city, 0.0)) + float(count) * float(cost_per_contract)
