"""
morning_trader.py — Morning Kelly-criterion bucket scanner and entry engine.

Runs at 07:15 ET after the 7AM intraday pulse has written predictions_latest.csv.
For each city, scans every Kalshi temperature bucket and buys those where the model
assigns more probability than the market price implies (positive Kelly) AND the price
is cheap enough to swing (default ≤ 25¢).

Exit targets start at 2.5× entry price. Once Data/market_price_history.csv has ≥10
historical data points for a city/price band, the target is replaced by the learned
median price at noon for similar entries.

This strategy supplements — not replaces — the existing 1 PM confident-mu trades.

Usage (dry run):
    python morning_trader.py --env prod --trade-date 2026-03-03

Usage (live):
    python morning_trader.py --env prod --trade-date 2026-03-03 --send-orders
"""

import argparse
import csv
import datetime
import math
import os
import statistics
import sys
import uuid
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

from kalshi_trader import (
    CITY_CONFIG,
    CITY_ORDER,
    SERIES_TICKERS,
    KalshiHttpClient,
    _parse_subtitle_to_range,
    bucket_probability,
    get_event,
    get_yes_pricing,
    make_trade,
    place_sell_order,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MORNING_ENTRIES_CSV = "Data/morning_entries.csv"
MARKET_PRICE_HISTORY_CSV = "Data/market_price_history.csv"
PREDICTIONS_LATEST_CSV = "Data/predictions_latest.csv"

MORNING_ENTRIES_COLS = [
    "logged_at",
    "trade_date",
    "city",
    "series_ticker",
    "event_ticker",
    "market_ticker",
    "market_subtitle",
    "bucket_lo",
    "bucket_hi",
    "entry_price",
    "model_prob",
    "kelly",
    "count",
    "target_exit_price",
    "exit_order_id",
    "exit_price_filled",
    "exit_ts",
    "send_orders",
    "status",
]

# Price bands used for grouping when learning exit targets (¢).
PRICE_BANDS = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 25)]
MIN_HISTORY_FOR_LEARNED_TARGET = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    tz = ZoneInfo(os.getenv("TZ", "America/New_York"))
    return datetime.datetime.now(tz=tz).isoformat()


def _event_ticker_for(series: str, trade_date: str) -> str:
    d = datetime.date.fromisoformat(trade_date)
    return f"{series}-{d.strftime('%y%b%d').upper()}"


def _load_predictions(path: str, trade_date: str) -> dict[str, dict]:
    """Return {city: {mu, sigma, spread_f, confidence_score}} from predictions_latest.csv."""
    result: dict[str, dict] = {}
    if not os.path.exists(path):
        return result
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("date") != trade_date:
                continue
            city = (row.get("city") or "").strip().lower()
            if not city:
                continue
            try:
                mu = float(row.get("tmax_predicted") or row.get("tmax_forecast") or "")
            except (ValueError, TypeError):
                continue
            try:
                sigma_raw = float(row.get("spread_f") or "")
            except (ValueError, TypeError):
                sigma_raw = 2.0
            # Use at least 2°F sigma so tail probabilities aren't near-zero.
            sigma = max(2.0, sigma_raw)
            result[city] = {
                "mu": mu,
                "sigma": sigma,
                "spread_f": sigma_raw,
                "confidence_score": row.get("confidence_score", ""),
            }
    return result


def _load_existing_entries(path: str, trade_date: str) -> set[str]:
    """Return set of market_tickers already traded today (idempotency guard)."""
    seen: set[str] = set()
    if not os.path.exists(path):
        return seen
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("trade_date") == trade_date:
                ticker = row.get("market_ticker", "").strip()
                if ticker:
                    seen.add(ticker)
    return seen


def _append_entry(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MORNING_ENTRIES_COLS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _price_band(price: int) -> tuple[int, int] | None:
    for lo, hi in PRICE_BANDS:
        if lo <= price <= hi:
            return (lo, hi)
    return None


def _compute_exit_target(
    entry_price: int,
    city: str,
    history_csv: str,
) -> int:
    """
    Compute limit sell target price.
    Default: 2.5× entry, capped at 75¢.
    Learned: median price_at_noon from market_price_history for this city/price band
             once ≥10 historical data points exist.
    """
    default = min(int(entry_price * 2.5), 75)
    band = _price_band(entry_price)
    if band is None or not os.path.exists(history_csv):
        return default

    band_lo, band_hi = band
    noon_prices: list[int] = []

    try:
        with open(history_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("city") != city:
                    continue
                # Only use noon snapshots (12:00 local) as the "exit window" reference.
                logged_at = row.get("logged_at", "")
                if len(logged_at) >= 13 and logged_at[11:13] != "12":
                    continue
                ask_raw = row.get("yes_ask", "")
                if not ask_raw:
                    continue
                try:
                    ask = int(ask_raw)
                except ValueError:
                    continue
                if band_lo <= ask <= band_hi:
                    noon_prices.append(ask)
    except Exception:
        return default

    if len(noon_prices) < MIN_HISTORY_FOR_LEARNED_TARGET:
        return default

    learned = min(int(statistics.median(noon_prices)), 75)
    # Never set target lower than 1.5× entry — ensure minimum reward.
    return max(learned, min(int(entry_price * 1.5), 75))


def _kelly(model_prob: float, ask_cents: int) -> float:
    """
    Full Kelly fraction for a binary YES contract at ask_cents.
    b = net odds (profit per dollar risked if win)
    f* = (b*p - q) / b
    """
    if ask_cents <= 0 or ask_cents >= 100:
        return 0.0
    b = (100 - ask_cents) / ask_cents
    p = model_prob
    q = 1.0 - p
    if b <= 0:
        return 0.0
    return (b * p - q) / b


# ---------------------------------------------------------------------------
# Core scan
# ---------------------------------------------------------------------------

def scan_and_enter(
    client: KalshiHttpClient,
    *,
    trade_date: str,
    city_predictions: dict[str, dict],
    morning_budget_dollars: float,
    max_entry_price: int,
    kelly_fraction: float,
    min_kelly: float,
    min_model_prob: float,
    max_buckets_per_city: int,
    min_depth: int,
    max_contracts: int,
    send_orders: bool,
    entries_csv: str,
    history_csv: str,
    dry_run_print: bool = True,
) -> list[dict]:
    """
    For each city, scan all Kalshi buckets, find positive-Kelly opportunities,
    size using fractional Kelly, and place limit buy orders.
    Returns list of entry dicts that were logged.
    """
    already_entered = _load_existing_entries(entries_csv, trade_date)
    city_budget = morning_budget_dollars / max(1, len(CITY_ORDER))
    all_entries: list[dict] = []

    for city in CITY_ORDER:
        pred = city_predictions.get(city)
        if pred is None:
            print(f"[morning_trader] {city}: no prediction for {trade_date} — skipping")
            continue

        mu: float = pred["mu"]
        sigma: float = pred["sigma"]
        series = SERIES_TICKERS.get(city, "")
        event_ticker = _event_ticker_for(series, trade_date)

        try:
            payload = get_event(client, event_ticker)
        except Exception as exc:
            print(f"[morning_trader] {city}: get_event failed — {exc}", file=sys.stderr)
            continue

        markets = (
            (payload.get("event") or {}).get("markets")
            or payload.get("markets")
            or []
        )
        if not markets:
            print(f"[morning_trader] {city}: no markets found for {event_ticker}")
            continue

        # Score every bucket.
        candidates: list[dict] = []
        for mkt in markets:
            ticker = mkt.get("ticker") or mkt.get("market_ticker") or ""
            subtitle = mkt.get("subtitle") or mkt.get("market_subtitle") or ""
            if not ticker or ticker in already_entered:
                continue

            lo, hi = _parse_subtitle_to_range(subtitle)
            model_prob = bucket_probability(lo=lo, hi=hi, mu=mu, sigma=sigma)

            # Gate 0: minimum model probability.
            # Prevents buying far-OTM tail bets that are only cheap because they're
            # unlikely — adjacent buckets near mu often exceed the price ceiling, leaving
            # only tails passing the price filter. Require meaningful forecast coverage.
            if model_prob < min_model_prob:
                continue

            try:
                pricing, _ = get_yes_pricing(client, ticker, orderbook_depth=5, fallback_qty=10)
            except Exception as exc:
                print(f"[morning_trader] {ticker}: pricing error — {exc}", file=sys.stderr)
                continue

            yes_ask = pricing.get("yes_ask")
            ask_qty = pricing.get("ask_qty")
            yes_bid = pricing.get("best_yes_bid")

            if yes_ask is None:
                continue
            yes_ask_i = int(yes_ask)

            # Gate 1: price ceiling.
            if yes_ask_i > max_entry_price or yes_ask_i <= 0:
                continue

            # Gate 2: liquidity floor.
            if ask_qty is not None and int(ask_qty) < min_depth:
                continue

            k = _kelly(model_prob, yes_ask_i)

            # Gate 3: positive Kelly with minimum edge.
            if k < min_kelly:
                continue

            candidates.append({
                "ticker": ticker,
                "subtitle": subtitle,
                "lo": lo,
                "hi": hi,
                "model_prob": model_prob,
                "kelly": k,
                "yes_ask": yes_ask_i,
                "yes_bid": yes_bid,
                "ask_qty": ask_qty,
            })

        # Rank by kelly × model_prob (edge × probability mass).
        candidates.sort(key=lambda c: c["kelly"] * c["model_prob"], reverse=True)
        top = candidates[:max_buckets_per_city]

        if not top:
            print(f"[morning_trader] {city}: no positive-Kelly buckets ≤ {max_entry_price}¢ found")
            continue

        for cand in top:
            ask = cand["yes_ask"]
            k = cand["kelly"]
            model_prob = cand["model_prob"]
            ticker = cand["ticker"]

            # Fractional Kelly sizing.
            cost_per = ask / 100.0
            raw_count = int(math.floor(kelly_fraction * k * city_budget / cost_per))
            count = min(raw_count, max_contracts)
            if count <= 0:
                print(
                    f"[morning_trader] {city} {ticker}: kelly sizing gives 0 contracts "
                    f"(budget=${city_budget:.2f}, k={k:.3f}, ask={ask}¢) — skipping"
                )
                continue

            target_exit = _compute_exit_target(ask, city, history_csv)

            entry = {
                "logged_at": _now_iso(),
                "trade_date": trade_date,
                "city": city,
                "series_ticker": series,
                "event_ticker": event_ticker,
                "market_ticker": ticker,
                "market_subtitle": cand["subtitle"],
                "bucket_lo": "" if cand["lo"] is None else f"{cand['lo']:.1f}",
                "bucket_hi": "" if cand["hi"] is None else f"{cand['hi']:.1f}",
                "entry_price": ask,
                "model_prob": f"{model_prob:.4f}",
                "kelly": f"{k:.4f}",
                "count": count,
                "target_exit_price": target_exit,
                "exit_order_id": "",
                "exit_price_filled": "",
                "exit_ts": "",
                "send_orders": str(send_orders),
                "status": "open",
            }

            cost_dollars = count * cost_per
            print(
                f"[morning_trader] {city} {ticker} '{cand['subtitle']}' — "
                f"ask={ask}¢ model_p={model_prob:.3f} kelly={k:.3f} "
                f"count={count} cost=${cost_dollars:.2f} target_exit={target_exit}¢"
            )

            if send_orders:
                try:
                    make_trade(
                        client=client,
                        markets=[{"ticker": ticker, "subtitle": cand["subtitle"]}],
                        pred=mu,
                        count=count,
                        yes_price=ask,
                        no_price=100 - ask,
                        side="yes",
                        trade_dt_str=trade_date,
                        city=city,
                        series=series,
                        event_ticker=event_ticker,
                        trades_log=None,
                        env="prod" if "prod" in str(client.base) else "demo",
                        send_orders=True,
                    )
                    print(f"[morning_trader] ✓ order submitted: {ticker} × {count}")
                except Exception as exc:
                    print(f"[morning_trader] order failed for {ticker}: {exc}", file=sys.stderr)
                    entry["status"] = "error"
            else:
                print(f"[morning_trader] DRY RUN — would buy {ticker} × {count} @ {ask}¢")

            _append_entry(entries_csv, entry)
            all_entries.append(entry)

    return all_entries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        return default


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Morning Kelly bucket scanner — buy cheap, exit on price drift.")
    p.add_argument("--trade-date", type=str, required=True)
    p.add_argument("--env", type=str, default=os.getenv("WT_ENV", "demo"))
    p.add_argument("--predictions-csv", type=str, default=PREDICTIONS_LATEST_CSV)
    p.add_argument("--entries-csv", type=str, default=MORNING_ENTRIES_CSV)
    p.add_argument("--history-csv", type=str, default=MARKET_PRICE_HISTORY_CSV)
    p.add_argument(
        "--morning-budget",
        type=float,
        default=_env_float("WT_MORNING_BUDGET", 10.0),
        help="Total dollars available for morning Kelly trades (default $10, from WT_MORNING_BUDGET).",
    )
    p.add_argument(
        "--max-entry-price",
        type=int,
        default=25,
        help="Maximum ask price in cents to consider (default 25¢).",
    )
    p.add_argument(
        "--kelly-fraction",
        type=float,
        default=0.25,
        help="Fraction of full Kelly to bet (default 0.25 = quarter-Kelly).",
    )
    p.add_argument(
        "--min-kelly",
        type=float,
        default=0.05,
        help="Minimum Kelly fraction to consider a trade (default 0.05).",
    )
    p.add_argument(
        "--min-model-prob",
        type=float,
        default=0.10,
        help="Minimum model probability for a bucket to be tradeable (default 0.10). "
             "Prevents buying far-OTM tails when adjacent buckets are priced above the entry ceiling.",
    )
    p.add_argument(
        "--max-buckets-per-city",
        type=int,
        default=2,
        help="Max number of buckets to enter per city (default 2).",
    )
    p.add_argument(
        "--min-depth",
        type=int,
        default=10,
        help="Minimum contracts at ask needed to enter (default 10).",
    )
    p.add_argument(
        "--max-contracts",
        type=int,
        default=500,
        help="Hard cap on contracts per order (default 500).",
    )
    p.add_argument(
        "--send-orders",
        action="store_true",
        default=False,
        help="Actually submit orders (default: dry run).",
    )
    p.add_argument("--api-key-id", type=str, default=os.getenv("KALSHI_API_KEY_ID", ""))
    p.add_argument("--private-key-path", type=str, default=os.getenv("KALSHI_PRIVATE_KEY_PATH", ""))
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not args.api_key_id or not args.private_key_path:
        print(
            "[morning_trader] KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH not set.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = KalshiHttpClient(
        env=args.env,
        api_key_id=args.api_key_id,
        private_key_path=args.private_key_path,
    )

    city_predictions = _load_predictions(args.predictions_csv, args.trade_date)
    if not city_predictions:
        print(
            f"[morning_trader] No predictions found for {args.trade_date} in {args.predictions_csv}",
            file=sys.stderr,
        )
        sys.exit(0)

    print(
        f"[morning_trader] {args.trade_date} — "
        f"budget=${args.morning_budget:.2f} max_entry={args.max_entry_price}¢ "
        f"kelly_fraction={args.kelly_fraction} send_orders={args.send_orders}"
    )
    for city, pred in city_predictions.items():
        print(f"  {city}: mu={pred['mu']:.2f}°F sigma={pred['sigma']:.2f}°F")

    entries = scan_and_enter(
        client,
        trade_date=args.trade_date,
        city_predictions=city_predictions,
        morning_budget_dollars=args.morning_budget,
        max_entry_price=args.max_entry_price,
        kelly_fraction=args.kelly_fraction,
        min_kelly=args.min_kelly,
        min_model_prob=args.min_model_prob,
        max_buckets_per_city=args.max_buckets_per_city,
        min_depth=args.min_depth,
        max_contracts=args.max_contracts,
        send_orders=args.send_orders,
        entries_csv=args.entries_csv,
        history_csv=args.history_csv,
    )

    print(f"[morning_trader] done — {len(entries)} position(s) logged to {args.entries_csv}")
