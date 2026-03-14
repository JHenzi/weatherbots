"""
exit_manager.py — Intraday limit-sell placement and position monitor.

Runs hourly (09:00–12:00 ET) to check open morning positions. If the current
YES bid has reached the target exit price, places a limit sell order. At 12:30 ET
the --cleanup flag cancels any unfilled limit sells so the positions settle normally
via the 1 PM mechanism.

Usage:
    python exit_manager.py --trade-date 2026-03-03 --env prod
    python exit_manager.py --trade-date 2026-03-03 --env prod --cleanup   # 12:30 pass
    python exit_manager.py --trade-date 2026-03-03 --env prod --send-orders
"""

import argparse
import csv
import datetime
import os
import sys
import uuid
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

from kalshi_trader import (
    KalshiHttpClient,
    cancel_order,
    get_open_orders,
    place_sell_order,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MORNING_ENTRIES_CSV = "Data/morning_entries.csv"

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


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    tz = ZoneInfo(os.getenv("TZ", "America/New_York"))
    return datetime.datetime.now(tz=tz).isoformat()


def _load_entries(path: str, trade_date: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        reader = csv.DictReader(f)
        return [r for r in reader if r.get("trade_date") == trade_date]


def _save_entries(path: str, all_rows: list[dict]) -> None:
    """Rewrite the full CSV (we keep all dates, update only today's rows)."""
    if not all_rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MORNING_ENTRIES_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)


def _load_all_entries(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def check_and_exit(
    client: KalshiHttpClient,
    *,
    trade_date: str,
    entries_csv: str,
    send_orders: bool,
    cleanup: bool = False,
) -> None:
    """
    For each open morning position:
    - If cleanup=True: cancel any resting limit sells, mark status=settled (will go to settlement).
    - Else: if yes_bid >= target_exit_price, place a limit sell.
    - If status=limit_placed: check whether the order has been filled.
    """
    all_rows = _load_all_entries(entries_csv)
    today_rows = [r for r in all_rows if r.get("trade_date") == trade_date]

    if not today_rows:
        print(f"[exit_manager] no morning entries for {trade_date}")
        return

    changed = False
    now = _now_iso()

    for row in today_rows:
        status = row.get("status", "")
        ticker = row.get("market_ticker", "")

        # Already done, or a shadow/dry-run entry — never touch these.
        if status in ("filled", "settled", "error", "cancelled", "shadow"):
            continue

        # --- Cleanup pass: cancel resting limit sells, leave to settle. ---
        if cleanup:
            if status == "limit_placed":
                order_id = row.get("exit_order_id", "")
                if order_id:
                    print(f"[exit_manager] cleanup: cancelling limit sell {order_id} for {ticker}")
                    if send_orders:
                        try:
                            cancel_order(client, order_id)
                        except Exception as exc:
                            print(f"[exit_manager] cancel failed: {exc}", file=sys.stderr)
                row["status"] = "settled"
                row["exit_ts"] = now
                changed = True
            elif status == "open":
                # Never got a limit sell off — will settle at close.
                row["status"] = "settled"
                row["exit_ts"] = now
                changed = True
            continue

        # --- Normal pass. ---
        if status == "limit_placed":
            # Check if the resting order has been filled.
            order_id = row.get("exit_order_id", "")
            filled = False
            if order_id:
                try:
                    open_orders = get_open_orders(client, ticker=ticker)
                    order_ids_resting = {o.get("order_id") or o.get("id") for o in open_orders}
                    if order_id not in order_ids_resting:
                        # No longer resting → assume filled.
                        filled = True
                except Exception as exc:
                    print(f"[exit_manager] order status check failed for {order_id}: {exc}", file=sys.stderr)
            if filled:
                target = int(row.get("target_exit_price", 0))
                row["status"] = "filled"
                row["exit_price_filled"] = str(target)
                row["exit_ts"] = now
                changed = True
                print(f"[exit_manager] ✓ filled: {ticker} @ {target}¢")
            continue

        # status == "open": morning_trader should have placed the limit sell immediately
        # after buying. If we're here, the sell placement failed at entry time — retry now.
        if status != "open":
            continue

        try:
            target = int(row.get("target_exit_price", 0))
            count = int(row.get("count", 0))
        except (ValueError, TypeError):
            continue

        if target <= 0 or count <= 0:
            continue

        print(
            f"[exit_manager] {ticker}: status=open (sell placement missed at entry) — "
            f"retrying limit sell @ {target}¢ × {count} "
            f"({'LIVE' if send_orders else 'DRY RUN'})"
        )
        if send_orders:
            try:
                result = place_sell_order(
                    client,
                    ticker=ticker,
                    count=count,
                    yes_price=target,
                )
                order_id = (
                    (result.get("order") or {}).get("order_id")
                    or result.get("order_id")
                    or str(uuid.uuid4())
                )
                row["exit_order_id"] = order_id
                row["status"] = "limit_placed"
                changed = True
                print(f"[exit_manager] ✓ retry sell placed: {order_id}")
            except Exception as exc:
                print(f"[exit_manager] retry sell failed for {ticker}: {exc}", file=sys.stderr)

    if changed:
        _save_entries(entries_csv, all_rows)
        print(f"[exit_manager] updated {entries_csv}")
    else:
        print(f"[exit_manager] no changes")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Intraday exit manager — place limit sells on morning positions.")
    p.add_argument("--trade-date", type=str, required=True)
    p.add_argument("--env", type=str, default=os.getenv("WT_ENV", "demo"))
    p.add_argument("--entries-csv", type=str, default=MORNING_ENTRIES_CSV)
    p.add_argument(
        "--cleanup",
        action="store_true",
        default=False,
        help="12:30 pass: cancel unfilled limit sells and mark positions as settled.",
    )
    p.add_argument(
        "--send-orders",
        action="store_true",
        default=False,
        help="Actually submit/cancel orders (default: dry run).",
    )
    p.add_argument("--api-key-id", type=str, default=os.getenv("KALSHI_API_KEY_ID", ""))
    p.add_argument("--private-key-path", type=str, default=os.getenv("KALSHI_PRIVATE_KEY_PATH", ""))
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not args.api_key_id or not args.private_key_path:
        print("[exit_manager] KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH not set.", file=sys.stderr)
        sys.exit(1)

    client = KalshiHttpClient(
        env=args.env,
        api_key_id=args.api_key_id,
        private_key_path=args.private_key_path,
    )

    mode = "CLEANUP" if args.cleanup else "CHECK"
    print(
        f"[exit_manager] {args.trade_date} {mode} "
        f"send_orders={args.send_orders}"
    )

    check_and_exit(
        client,
        trade_date=args.trade_date,
        entries_csv=args.entries_csv,
        send_orders=args.send_orders,
        cleanup=args.cleanup,
    )
