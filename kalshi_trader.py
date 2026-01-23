import argparse
import base64
import datetime
import os
import csv
import uuid

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


# for seriesTicker in seriesTickers:
#   temps = getTempRanges(seriesTicker)


# Function to determine the interval
def find_interval(value, intervals):
    for i in range(len(intervals)):
        subtitle = intervals[i].get("subtitle", "") if isinstance(intervals[i], dict) else ""
        if "or below" in subtitle:
            max_value = float(subtitle.split("°")[0])
            if value <= max_value:
                return i
        elif "or above" in subtitle:
            min_value = float(subtitle.split("°")[0])
            if value >= min_value:
                return i
        else:
            bounds = subtitle.replace("°", "").split(" to ")
            min_value, max_value = map(float, bounds)
            if min_value <= value <= max_value:
                return i
    return 0


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
):
    # Check the interval for pred
    interval = find_interval(pred, markets)
    chosen = markets[interval]

    subtitle = chosen.get("subtitle", "")
    ticker = chosen.get("ticker")
    print(f"Pred={pred:.2f} → interval='{subtitle}' → market={ticker}")

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
    p.add_argument("--count", type=int, default=10, help="Contracts per order")
    p.add_argument("--side", type=str, default="yes", choices=["yes", "no"])
    p.add_argument("--yes-price", type=int, default=50, help="Limit price in cents (1-99)")
    p.add_argument("--no-price", type=int, default=50, help="Limit price in cents (1-99)")
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

    preds: dict[tuple[str, str], float] = {}
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
                preds[(d, c)] = float(t)
            except ValueError:
                continue

    for city in CITY_ORDER:
        series = SERIES_TICKERS[city]
        event_ticker = f"{series}-{event_suffix}"
        print(f"\n----------- {series} / city={city} / trade_date={trade_dt_str} -----------")

        key = (trade_dt_str, city)
        if key not in preds:
            raise RuntimeError(
                f"No prediction found for date={trade_dt_str} city={city} in {args.predictions_csv}"
            )
        pred = preds[key]

        event_payload = get_event(client, event_ticker)
        markets = (event_payload.get("event") or {}).get("markets") or event_payload.get("markets") or []
        if not markets:
            raise RuntimeError(f"No markets returned for event {event_ticker}")

        make_trade(
            client=client,
            pred=pred,
            markets=markets,
            send_orders=args.send_orders,
            count=args.count,
            side=args.side,
            yes_price=args.yes_price,
            no_price=args.no_price,
        )
