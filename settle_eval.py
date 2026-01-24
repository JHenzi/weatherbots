import argparse
import csv
import datetime as dt
import os

from truth_engine import get_actual_tmax_from_nws_cli


def _local_tz() -> dt.tzinfo:
    tzname = (os.getenv("TZ") or "").strip() or "America/New_York"
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(tzname)
    except Exception:
        return dt.datetime.now().astimezone().tzinfo or dt.timezone.utc


def _parse_args():
    p = argparse.ArgumentParser(description="Backfill settlement temperature and realized PnL into Data/eval_history.csv.")
    p.add_argument("--trade-date", type=str, required=True, help="YYYY-MM-DD (event date to settle)")
    p.add_argument("--eval-csv", type=str, default="Data/eval_history.csv")
    return p.parse_args()


def _in_bucket(actual: float, lo: str, hi: str) -> bool | None:
    lo_v = None
    hi_v = None
    try:
        lo_v = float(lo) if lo not in ("", None) else None
    except Exception:
        lo_v = None
    try:
        hi_v = float(hi) if hi not in ("", None) else None
    except Exception:
        hi_v = None

    if lo_v is None and hi_v is None:
        return None
    if lo_v is not None and actual < lo_v:
        return False
    if hi_v is not None and actual > hi_v:
        return False
    return True


if __name__ == "__main__":
    args = _parse_args()
    trade_dt = dt.datetime.strptime(args.trade_date, "%Y-%m-%d").date()

    if not os.path.exists(args.eval_csv):
        raise RuntimeError(f"Missing eval CSV: {args.eval_csv}")

    with open(args.eval_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
        fieldnames = list(r.fieldnames or [])

    # Columns to backfill/update (in-place rewrite).
    new_cols = [
        "settlement_tmax_f",
        "settlement_source_url",
        "bucket_hit",
        "realized_pnl_cents",
        "realized_pnl_dollars",
        "settled_at",
    ]
    for c in new_cols:
        if c not in fieldnames:
            fieldnames.append(c)

    updated = 0
    for row in rows:
        if (row.get("trade_date") or "").strip() != args.trade_date:
            continue
        city = (row.get("city") or "").strip()
        if not city:
            continue

        # Only settle trades where a bucket was selected and we have a count/ask.
        chosen = (row.get("chosen_market_ticker") or "").strip()
        if not chosen:
            continue
        try:
            count = int(float(row.get("count") or 0))
        except Exception:
            count = 0
        if count <= 0:
            continue
        try:
            yes_ask = int(float(row.get("yes_ask") or 0))
        except Exception:
            yes_ask = 0
        if yes_ask <= 0:
            continue

        # If already settled, skip.
        if (row.get("settlement_tmax_f") or "").strip():
            continue

        try:
            truth = get_actual_tmax_from_nws_cli(city, trade_dt)
        except Exception:
            # Not available yet; leave row unchanged.
            continue

        actual = float(truth.observed_max_f)
        hit = _in_bucket(actual, row.get("bucket_lo", ""), row.get("bucket_hi", ""))
        if hit is None:
            continue

        # Assume we bought YES at yes_ask. Payout is 100 cents if hit else 0.
        pnl_cents = (count * (100 - yes_ask)) if hit else (-count * yes_ask)

        row["settlement_tmax_f"] = f"{actual:.1f}"
        row["settlement_source_url"] = truth.source_url
        row["bucket_hit"] = "1" if hit else "0"
        row["realized_pnl_cents"] = str(int(pnl_cents))
        row["realized_pnl_dollars"] = f"{pnl_cents/100.0:.2f}"
        row["settled_at"] = dt.datetime.now(tz=_local_tz()).isoformat()
        updated += 1

    # Rewrite file atomically-ish (best effort).
    tmp_path = args.eval_csv + ".tmp"
    with open(tmp_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    os.replace(tmp_path, args.eval_csv)

    print(f"Updated {updated} rows in {args.eval_csv}")

