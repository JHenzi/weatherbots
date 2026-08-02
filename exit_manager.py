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
    SERIES_TICKERS,
    KalshiHttpClient,
    cancel_order,
    get_market,
    get_open_orders,
    get_portfolio_positions,
    get_yes_pricing,
    place_sell_order,
)

# series ticker (KXHIGHCHI) -> city code (il), for mapping positions to obs.
_SERIES_TO_CITY = {v: k for k, v in SERIES_TICKERS.items()}
# Only these series are ever managed. Any other portfolio position (e.g. an
# election market the user holds manually) is never considered or touched.
_WEATHER_SERIES = frozenset(SERIES_TICKERS.values())


def _is_managed_ticker(ticker: str) -> bool:
    """True only for the daily high-temp weather markets this bot trades."""
    series = ticker.split("-")[0] if "-" in ticker else ticker
    return series in _WEATHER_SERIES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MORNING_ENTRIES_CSV = "Data/morning_entries.csv"
OBSERVATIONS_LATEST_JSON = "Data/observations_latest.json"
TRAILING_STATE_JSON = "Data/exit_trailing_state.json"

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
# Live-position exit engine (covers the 1–2 PM daily-trade path)
# ---------------------------------------------------------------------------
#
# The morning path above manages positions it recorded in morning_entries.csv.
# The daily-trade path (run_trade.sh -> kalshi_trader.py, logged only to
# trades_history.csv) records no exit metadata, so we manage those positions
# directly from Kalshi's live portfolio. Two triggers, both YES-long only:
#   * obs bucket-breach stop-loss — projected_high has moved outside the bucket
#     by danger_threshold_f (the temp is breaching against us).
#   * trailing stop — YES bid has retraced trail_cents from its peak since entry,
#     once the position was up by at least trail_arm_gain_cents.
# Peak-bid state persists in TRAILING_STATE_JSON across the 30-min cron cycles.


def _parse_bucket_bounds(subtitle: str) -> tuple[float | None, float | None, str]:
    """Parse a Kalshi high-temp market subtitle into (lo, hi, kind).

    "79° or below" -> (None, 79.0, "below")   YES wins if actual <= hi
    "80° or above" -> (80.0, None, "above")    YES wins if actual >= lo
    "75° to 76°"   -> (75.0, 76.0, "range")    YES wins if lo <= actual <= hi
    Returns (None, None, "unknown") if it can't be parsed.
    """
    import re

    s = (subtitle or "").lower()
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", s)]
    if "below" in s or "under" in s or "or less" in s:
        return (None, nums[0], "below") if nums else (None, None, "unknown")
    if "above" in s or "over" in s or "or more" in s or "or higher" in s:
        return (nums[0], None, "above") if nums else (None, None, "unknown")
    if len(nums) >= 2:
        return (min(nums[0], nums[1]), max(nums[0], nums[1]), "range")
    return (None, None, "unknown")


def _to_cents_int(v) -> int | None:
    """Best-effort cents from a Kalshi field that may be int-cents or a
    dollar string like '0.76'. Values in (0, ~1] are treated as dollars."""
    if v is None:
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    if fv == 0:
        return 0
    if 0 < fv <= 1.0:
        return int(round(fv * 100))
    return int(round(fv))


def _position_count(mp: dict) -> int:
    """Signed contract count across Kalshi field variants.

    The newer "dollars" API omits the integer ``position`` field and returns
    ``position_fp`` (a string like "3.00") instead. Missing/blank -> 0.
    """
    v = mp.get("position")
    if v is None or v == "":
        v = mp.get("position_fp")
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return 0


def _position_cost_cents(mp: dict) -> int | None:
    """Cost basis (cents) of a market position across Kalshi field variants."""
    for key in ("market_exposure", "total_traded", "cost_basis"):
        c = _to_cents_int(mp.get(key))
        if c:
            return abs(c)
    for key in ("market_exposure_dollars", "total_traded_dollars"):
        v = mp.get(key)
        if v is not None:
            try:
                return abs(int(round(float(v) * 100)))
            except (TypeError, ValueError):
                pass
    return None


def _load_trailing_state(path: str) -> dict:
    import json

    if not os.path.exists(path):
        return {"positions": {}}
    try:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict) or "positions" not in data:
            return {"positions": {}}
        return data
    except Exception:
        return {"positions": {}}


def _save_trailing_state(path: str, state: dict) -> None:
    import json

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def check_and_exit_live(
    client: KalshiHttpClient,
    *,
    send_orders: bool,
    obs_json: str = OBSERVATIONS_LATEST_JSON,
    state_path: str = TRAILING_STATE_JSON,
    danger_threshold_f: float = 1.5,
    trail_cents: int = 10,
    trail_arm_gain_cents: int = 8,
    resell_cooldown_min: int = 20,
) -> None:
    """Manage live Kalshi YES positions with obs bucket-breach + trailing stops."""
    try:
        raw = get_portfolio_positions(client, count_filter="position", limit=100)
    except Exception as exc:
        print(f"[exit_manager] live: could not fetch positions: {exc}", file=sys.stderr)
        return

    market_positions = raw.get("market_positions") or []
    projected_highs = _load_projected_highs(obs_json)
    state = _load_trailing_state(state_path)
    positions_state = state.setdefault("positions", {})
    now = _now_iso()
    changed = False
    seen: set[str] = set()

    open_yes = [
        mp for mp in market_positions
        if _position_count(mp) > 0 and _is_managed_ticker(mp.get("ticker") or "")
    ]
    if not open_yes:
        print("[exit_manager] live: no open YES positions")
    for mp in market_positions:
        ticker = mp.get("ticker") or ""
        position = _position_count(mp)
        if not ticker or position == 0:
            continue
        # Never consider or manage anything outside our weather series (e.g. a
        # manually-held election market). Hard restriction — no exits, no state.
        if not _is_managed_ticker(ticker):
            print(f"[exit_manager] live: {ticker} not a managed weather market — ignoring")
            continue
        seen.add(ticker)
        if position < 0:
            # System only ever goes long YES; skip NO positions we didn't open.
            print(f"[exit_manager] live: {ticker} position={position} (NO/short) — skipping")
            continue

        ps = positions_state.setdefault(ticker, {})
        ps["last_seen"] = now

        # Current YES bid / market metadata.
        try:
            px, _ = get_yes_pricing(client, ticker, orderbook_depth=5, fallback_qty=5)
            bid = px.get("best_yes_bid")
            bid = int(bid) if bid is not None else None
        except Exception as exc:
            print(f"[exit_manager] live: pricing failed for {ticker}: {exc}", file=sys.stderr)
            bid = None
        try:
            m = get_market(client, ticker)
            subtitle = m.get("subtitle") or m.get("title") or ""
        except Exception:
            subtitle = ""

        avg_entry = _position_cost_cents(mp)
        avg_entry = int(avg_entry / position) if avg_entry else None

        # Update trailing peak.
        if bid is not None:
            ps["peak_bid"] = max(int(ps.get("peak_bid") or 0), bid)
            ps["last_bid"] = bid
        peak = int(ps.get("peak_bid") or 0)

        # Skip if we already fired a sell recently (avoid stacking orders before fill).
        exited_ts = ps.get("exited_ts")
        if exited_ts:
            try:
                dt_exit = datetime.datetime.fromisoformat(exited_ts)
                age_min = (datetime.datetime.now(tz=dt_exit.tzinfo) - dt_exit).total_seconds() / 60.0
                if age_min < resell_cooldown_min:
                    print(f"[exit_manager] live: {ticker} sell placed {age_min:.0f}m ago — cooldown")
                    changed = True
                    continue
            except Exception:
                pass

        # --- Trigger evaluation ---
        trigger: str | None = None
        lo, hi, kind = _parse_bucket_bounds(subtitle)
        # projected_high is keyed by city code (ny/il/tx/fl); map via series ticker.
        series = ticker.split("-")[0] if "-" in ticker else ""
        city = _SERIES_TO_CITY.get(series, series)
        ph = projected_highs.get(city)

        # 1) Obs bucket-breach stop-loss.
        if ph is not None and kind != "unknown":
            if kind in ("below", "range") and hi is not None and ph > hi + danger_threshold_f:
                trigger = f"bucket_breach;too_hot;proj={ph:.2f}>hi={hi:.1f}+{danger_threshold_f}"
            elif kind in ("above", "range") and lo is not None and ph < lo - danger_threshold_f:
                trigger = f"bucket_breach;too_cold;proj={ph:.2f}<lo={lo:.1f}-{danger_threshold_f}"

        # 2) Trailing stop (only if not already breaching).
        if trigger is None and bid is not None and avg_entry is not None:
            armed = peak >= avg_entry + trail_arm_gain_cents
            if armed and bid <= peak - trail_cents:
                trigger = (
                    f"trailing_stop;bid={bid}<=peak={peak}-{trail_cents};"
                    f"entry={avg_entry}"
                )

        if trigger is None:
            print(
                f"[exit_manager] live: {ticker} hold "
                f"(bid={bid} peak={peak} entry={avg_entry} proj={ph} "
                f"bucket={kind}:{lo}-{hi})"
            )
            changed = True  # peak/last_seen may have updated
            continue

        # Aggressive sell 1¢ below bid to improve fill odds (floor at 1¢).
        sell_price = max((bid - 1) if bid is not None else 1, 1)
        print(
            f"[exit_manager] live: {ticker} EXIT {trigger} — selling {position}× "
            f"@ {sell_price}¢ ({'LIVE' if send_orders else 'DRY RUN'})"
        )
        if send_orders:
            try:
                place_sell_order(client, ticker=ticker, count=position, yes_price=sell_price)
                ps["exited_ts"] = now
                ps["exit_reason"] = trigger
                changed = True
                print(f"[exit_manager] live: ✓ sell placed for {ticker}")
            except Exception as exc:
                print(f"[exit_manager] live: sell failed for {ticker}: {exc}", file=sys.stderr)
        else:
            ps["would_exit"] = trigger
            changed = True

    # Prune state for tickers no longer held (and not recently seen).
    for tk in list(positions_state.keys()):
        if tk in seen:
            continue
        last = positions_state[tk].get("last_seen")
        drop = True
        if last:
            try:
                dt_last = datetime.datetime.fromisoformat(last)
                age_h = (datetime.datetime.now(tz=dt_last.tzinfo) - dt_last).total_seconds() / 3600.0
                drop = age_h > 48
            except Exception:
                drop = True
        if drop:
            del positions_state[tk]
            changed = True

    if changed:
        _save_trailing_state(state_path, state)
        print(f"[exit_manager] live: updated {state_path}")


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
    p.add_argument(
        "--live",
        action="store_true",
        default=False,
        help=(
            "Manage live Kalshi YES positions directly (obs bucket-breach + trailing "
            "stop) instead of the morning_entries.csv path. Use for the 1–2 PM "
            "daily-trade positions, which record no exit metadata."
        ),
    )
    p.add_argument(
        "--trail-cents",
        type=int,
        default=int(os.getenv("WT_EXIT_TRAIL_CENTS", "10")),
        help="Trailing stop: sell when YES bid retraces this many ¢ from its peak (default 10).",
    )
    p.add_argument(
        "--trail-arm-gain-cents",
        type=int,
        default=int(os.getenv("WT_EXIT_TRAIL_ARM_GAIN_CENTS", "8")),
        help="Trailing stop arms only once peak bid is this many ¢ above entry (default 8).",
    )
    p.add_argument(
        "--trailing-state",
        type=str,
        default=TRAILING_STATE_JSON,
        help="Path to persistent trailing-peak state (default: Data/exit_trailing_state.json).",
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

    if args.live:
        print(
            f"[exit_manager] {args.trade_date} LIVE (daily-trade positions) "
            f"send_orders={args.send_orders} trail={args.trail_cents}¢ "
            f"arm={args.trail_arm_gain_cents}¢ danger={args.danger_threshold_f}°F"
        )
        check_and_exit_live(
            client,
            send_orders=args.send_orders,
            obs_json=args.obs_json,
            state_path=args.trailing_state,
            danger_threshold_f=args.danger_threshold_f,
            trail_cents=args.trail_cents,
            trail_arm_gain_cents=args.trail_arm_gain_cents,
        )
    else:
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
