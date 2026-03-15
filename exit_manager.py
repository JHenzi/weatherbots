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
    get_yes_pricing,
    place_sell_order,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MORNING_ENTRIES_CSV = "Data/morning_entries.csv"
OBSERVATIONS_LATEST_JSON = "Data/observations_latest.json"

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
    "mu_pred",
    "sigma_pred",
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

def _load_projected_highs(obs_json_path: str) -> dict[str, float]:
    """Return {city: projected_high} from observations_latest.json.

    The JSON shape is:
        {"last_update": "...", "stations": {"ny": {"projected_high": 48.2, ...}, ...}}
    Returns an empty dict if the file is missing or malformed.
    """
    import json
    result: dict[str, float] = {}
    if not os.path.exists(obs_json_path):
        return result
    try:
        with open(obs_json_path) as f:
            data = json.load(f)
        stations = data.get("stations") or {}
        for city, obs in stations.items():
            if not isinstance(obs, dict):
                continue
            ph = obs.get("projected_high")
            if ph is not None:
                try:
                    result[str(city).lower()] = float(ph)
                except (ValueError, TypeError):
                    pass
    except Exception as exc:
        print(f"[exit_manager] could not load observations from {obs_json_path}: {exc}",
              file=sys.stderr)
    return result


def check_and_exit(
    client: KalshiHttpClient,
    *,
    trade_date: str,
    entries_csv: str,
    send_orders: bool,
    cleanup: bool = False,
    obs_json: str = OBSERVATIONS_LATEST_JSON,
    danger_threshold_f: float = 1.5,
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
    projected_highs = _load_projected_highs(obs_json)

    for row in today_rows:
        status = row.get("status", "")
        ticker = row.get("market_ticker", "")

        # Already done, or a shadow/dry-run entry — never touch these.
        if status in ("filled", "settled", "error", "cancelled", "shadow", "obs_exit"):
            continue

        # --- Observation-based exit (danger or win-capture). ---
        # Fires before the cleanup pass so we can place an aggressive sell rather
        # than just cancelling to settle.
        if not cleanup and status in ("open", "limit_placed"):
            city = (row.get("city") or "").lower()
            ph = projected_highs.get(city)
            try:
                bucket_lo = float(row.get("bucket_lo") or "nan")
                bucket_hi = float(row.get("bucket_hi") or "nan")
            except (ValueError, TypeError):
                bucket_lo = float("nan")
                bucket_hi = float("nan")

            obs_trigger: str | None = None
            if ph is not None and not (bucket_lo != bucket_lo or bucket_hi != bucket_hi):
                # NaN check above: if either bound is missing, skip obs logic.
                if ph > bucket_hi + danger_threshold_f:
                    obs_trigger = f"too_hot;proj={ph:.2f}>hi={bucket_hi:.1f}+{danger_threshold_f}"
                elif ph < bucket_lo - danger_threshold_f:
                    obs_trigger = f"too_cold;proj={ph:.2f}<lo={bucket_lo:.1f}-{danger_threshold_f}"
                else:
                    # Within bucket — check if we can capture a win aggressively.
                    # Win capture fires when the resting sell hasn't filled yet but
                    # the current YES bid is at or above 85% of target exit price.
                    try:
                        target = int(row.get("target_exit_price") or 0)
                        entry_p = int(row.get("entry_price") or 0)
                    except (ValueError, TypeError):
                        target = 0
                        entry_p = 0
                    if target > 0 and entry_p > 0 and status == "limit_placed":
                        try:
                            px, _ = get_yes_pricing(
                                client, ticker, orderbook_depth=5, fallback_qty=5
                            )
                            bid = px.get("best_yes_bid")
                            if bid is not None and int(bid) >= int(target * 0.85):
                                obs_trigger = (
                                    f"win_capture;proj={ph:.2f} in bucket;"
                                    f"bid={int(bid)}>={int(target*0.85)}"
                                )
                        except Exception:
                            pass

            if obs_trigger:
                try:
                    count = int(row.get("count") or 0)
                except (ValueError, TypeError):
                    count = 0
                if count > 0:
                    # Get current best bid for aggressive sell pricing.
                    sell_price = 1
                    try:
                        px, _ = get_yes_pricing(
                            client, ticker, orderbook_depth=5, fallback_qty=5
                        )
                        bid = px.get("best_yes_bid")
                        if bid is not None:
                            # Sell 1¢ below bid for danger exits to improve fill odds;
                            # at bid for win captures (price is already favourable).
                            discount = 0 if obs_trigger.startswith("win_capture") else 1
                            sell_price = max(int(bid) - discount, 1)
                    except Exception:
                        pass

                    print(
                        f"[exit_manager] {ticker} OBS-EXIT {obs_trigger} — "
                        f"selling {count}× @ {sell_price}¢ "
                        f"({'LIVE' if send_orders else 'DRY RUN'})"
                    )
                    # Cancel existing resting limit sell first to avoid two competing orders.
                    existing_order_id = row.get("exit_order_id", "")
                    if status == "limit_placed" and existing_order_id and send_orders:
                        try:
                            cancel_order(client, existing_order_id)
                        except Exception as exc:
                            print(f"[exit_manager] cancel before obs-exit failed: {exc}",
                                  file=sys.stderr)

                    if send_orders:
                        try:
                            result = place_sell_order(
                                client, ticker=ticker, count=count, yes_price=sell_price
                            )
                            new_order_id = (
                                (result.get("order") or {}).get("order_id")
                                or result.get("order_id")
                                or str(uuid.uuid4())
                            )
                            row["exit_order_id"] = new_order_id
                            row["status"] = "obs_exit"
                            row["exit_ts"] = now
                            changed = True
                            print(f"[exit_manager] ✓ obs-exit sell placed: {new_order_id}")
                        except Exception as exc:
                            print(f"[exit_manager] obs-exit sell failed for {ticker}: {exc}",
                                  file=sys.stderr)
                    else:
                        row["status"] = "obs_exit"
                        row["exit_ts"] = now
                        changed = True
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
    p.add_argument(
        "--obs-json",
        type=str,
        default=OBSERVATIONS_LATEST_JSON,
        help="Path to observations_latest.json (default: Data/observations_latest.json).",
    )
    p.add_argument(
        "--danger-threshold-f",
        type=float,
        default=1.5,
        help=(
            "Degrees F outside bucket bounds that triggers an observation-based exit. "
            "Default 1.5°F — fires when projected_high exceeds bucket_hi+1.5 or "
            "falls below bucket_lo-1.5."
        ),
    )
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
        obs_json=args.obs_json,
        danger_threshold_f=args.danger_threshold_f,
    )
