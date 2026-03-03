"""
log_market_prices.py — snapshot Kalshi market prices for all buckets across all cities.

Runs at every intraday pulse (wired into run_intraday_pulse.sh). Appends one row per
bucket per run to Data/market_price_history.csv. This CSV is the training data for the
morning Kelly exit-price learning model.

Usage:
    python log_market_prices.py --trade-date 2026-03-03 --env prod
"""

import argparse
import csv
import datetime
import os
import sys
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

# Re-use the full Kalshi client and helpers from kalshi_trader.
from kalshi_trader import (
    CITY_CONFIG,
    CITY_ORDER,
    SERIES_TICKERS,
    KalshiHttpClient,
    _parse_subtitle_to_range,
    get_event,
    get_yes_pricing,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MARKET_PRICE_CSV = "Data/market_price_history.csv"
MARKET_PRICE_COLS = [
    "logged_at",
    "trade_date",
    "city",
    "series_ticker",
    "event_ticker",
    "market_ticker",
    "market_subtitle",
    "bucket_lo",
    "bucket_hi",
    "yes_ask",
    "yes_bid",
    "ask_qty",
    "bid_qty",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    tz = ZoneInfo(os.getenv("TZ", "America/New_York"))
    return datetime.datetime.now(tz=tz).isoformat()


def _append_rows(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MARKET_PRICE_COLS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _event_ticker_for(series: str, trade_date: str) -> str:
    # e.g. KXHIGHNY + 2026-03-03 → KXHIGHNY-26MAR03
    d = datetime.date.fromisoformat(trade_date)
    return f"{series}-{d.strftime('%y%b%d').upper()}"


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def log_prices(
    client: KalshiHttpClient,
    *,
    trade_date: str,
    cities: list[str],
    out_csv: str = MARKET_PRICE_CSV,
) -> int:
    """Fetch all bucket prices for each city and append to out_csv. Returns row count."""
    now = _now_iso()
    rows: list[dict] = []

    for city in cities:
        series = SERIES_TICKERS.get(city, "")
        if not series:
            continue
        event_ticker = _event_ticker_for(series, trade_date)

        try:
            payload = get_event(client, event_ticker)
        except Exception as exc:
            print(f"[log_market_prices] {city}: get_event failed — {exc}", file=sys.stderr)
            continue

        markets = (
            (payload.get("event") or {}).get("markets")
            or payload.get("markets")
            or []
        )
        if not markets:
            print(f"[log_market_prices] {city}: no markets in event {event_ticker}")
            continue

        for mkt in markets:
            ticker = mkt.get("ticker") or mkt.get("market_ticker") or ""
            subtitle = mkt.get("subtitle") or mkt.get("market_subtitle") or ""
            if not ticker:
                continue

            lo, hi = _parse_subtitle_to_range(subtitle)

            try:
                pricing, _ = get_yes_pricing(client, ticker, orderbook_depth=5, fallback_qty=10)
            except Exception as exc:
                print(f"[log_market_prices] {ticker}: pricing failed — {exc}", file=sys.stderr)
                continue

            rows.append({
                "logged_at": now,
                "trade_date": trade_date,
                "city": city,
                "series_ticker": series,
                "event_ticker": event_ticker,
                "market_ticker": ticker,
                "market_subtitle": subtitle,
                "bucket_lo": "" if lo is None else f"{lo:.1f}",
                "bucket_hi": "" if hi is None else f"{hi:.1f}",
                "yes_ask": "" if pricing.get("yes_ask") is None else str(pricing["yes_ask"]),
                "yes_bid": "" if pricing.get("best_yes_bid") is None else str(pricing["best_yes_bid"]),
                "ask_qty": "" if pricing.get("ask_qty") is None else str(pricing["ask_qty"]),
                "bid_qty": "" if pricing.get("bid_qty") is None else str(pricing.get("bid_qty", "")),
            })

    _append_rows(out_csv, rows)
    print(f"[log_market_prices] logged {len(rows)} bucket prices → {out_csv}")
    return len(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Snapshot Kalshi bucket prices for all cities.")
    p.add_argument("--trade-date", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument("--env", type=str, default=os.getenv("WT_ENV", "demo"))
    p.add_argument("--cities", type=str, default="", help="Comma-separated city codes; default=all")
    p.add_argument("--out-csv", type=str, default=MARKET_PRICE_CSV)
    p.add_argument(
        "--api-key-id",
        type=str,
        default=os.getenv("KALSHI_API_KEY_ID", ""),
    )
    p.add_argument(
        "--private-key-path",
        type=str,
        default=os.getenv("KALSHI_PRIVATE_KEY_PATH", ""),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not args.api_key_id or not args.private_key_path:
        print(
            "[log_market_prices] KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH not set — skipping.",
            file=sys.stderr,
        )
        sys.exit(0)

    client = KalshiHttpClient(
        env=args.env,
        api_key_id=args.api_key_id,
        private_key_path=args.private_key_path,
    )

    cities = [c.strip() for c in args.cities.split(",") if c.strip()] if args.cities else CITY_ORDER

    log_prices(client, trade_date=args.trade_date, cities=cities, out_csv=args.out_csv)
