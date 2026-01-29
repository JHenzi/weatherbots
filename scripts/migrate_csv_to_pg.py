import argparse
import csv
import os
from typing import Any, Dict

import psycopg2

# Ensure db helpers are enabled for this one-off migration script.
os.environ.setdefault("ENABLE_PG_WRITE", "1")

import db  # type: ignore  # local module


def _get_conn():
    """
    Reuse the db module's cached connection for both existence checks and inserts.
    """
    # db._get_conn is intentionally internal; we use it here to avoid duplicating DSN logic.
    conn = db._get_conn()  # type: ignore[attr-defined]
    if conn is None:
        raise RuntimeError(
            "Could not connect to Postgres. Ensure PGDATABASE_URL (or DATABASE_URL) is set "
            "and Postgres is running."
        )
    return conn


def _parse_args():
    p = argparse.ArgumentParser(
        description="Backfill historical CSV logs into Postgres (idempotent best-effort)."
    )
    p.add_argument("--data-dir", type=str, default="Data", help="Base data directory.")
    p.add_argument(
        "--predictions-history",
        type=str,
        default="Data/predictions_history.csv",
        help="Path to predictions_history.csv.",
    )
    p.add_argument(
        "--intraday-forecasts",
        type=str,
        default="Data/intraday_forecasts.csv",
        help="Path to intraday_forecasts.csv.",
    )
    p.add_argument(
        "--trades-history",
        type=str,
        default="Data/trades_history.csv",
        help="Path to trades_history.csv.",
    )
    p.add_argument(
        "--decisions-history",
        type=str,
        default="Data/decisions_history.csv",
        help="Path to decisions_history.csv.",
    )
    p.add_argument(
        "--eval-history",
        type=str,
        default="Data/eval_history.csv",
        help="Path to eval_history.csv.",
    )
    p.add_argument(
        "--daily-metrics",
        type=str,
        default="Data/daily_metrics.csv",
        help="Path to daily_metrics.csv.",
    )
    p.add_argument(
        "--source-performance",
        type=str,
        default="Data/source_performance.csv",
        help="Path to source_performance.csv.",
    )
    p.add_argument(
        "--weights-history",
        type=str,
        default="Data/weights_history.csv",
        help="Path to weights_history.csv.",
    )
    return p.parse_args()


def _file_exists(path: str) -> bool:
    return bool(path) and os.path.exists(path)


def migrate_trades(path: str) -> None:
    if not _file_exists(path):
        print(f"[migrate] trades: missing {path}, skipping")
        return
    conn = _get_conn()
    cur = conn.cursor()
    inserted = 0
    seen = 0
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            seen += 1
            env = (row.get("env") or "").strip()
            trade_date = (row.get("trade_date") or "").strip()
            city = (row.get("city") or "").strip()
            series = (row.get("series_ticker") or "").strip()
            event = (row.get("event_ticker") or "").strip()
            market = (row.get("market_ticker") or "").strip()
            run_ts = (row.get("run_ts") or "").strip()
            if not (env and trade_date and city and series and event and market and run_ts):
                continue
            try:
                cur.execute(
                    """
                    SELECT 1
                    FROM trades t
                    JOIN cities c ON t.city_id = c.id
                    WHERE t.env = %s
                      AND t.trade_date = %s
                      AND c.code = %s
                      AND t.series_ticker = %s
                      AND t.event_ticker = %s
                      AND t.market_ticker = %s
                      AND t.run_ts = %s
                    LIMIT 1
                    """,
                    (env, trade_date, city, series, event, market, run_ts),
                )
                if cur.fetchone():
                    continue
                # Coerce a few obvious types that were stored numerically.
                row = dict(row)
                if "count" in row and row["count"] not in ("", None):
                    try:
                        row["count"] = int(float(row["count"]))
                    except Exception:
                        pass
                if "yes_price" in row and row["yes_price"] not in ("", None):
                    try:
                        row["yes_price"] = int(float(row["yes_price"]))
                    except Exception:
                        pass
                if "no_price" in row and row["no_price"] not in ("", None):
                    try:
                        row["no_price"] = int(float(row["no_price"]))
                    except Exception:
                        pass
                db.insert_trade_row(row)
                inserted += 1
            except Exception as e:
                conn.rollback()
                print(f"[migrate] trades: skipping malformed row (seen={seen}): {e}")
                continue
    print(f"[migrate] trades: {inserted} inserted (rows seen={seen})")


def migrate_decisions(path: str) -> None:
    if not _file_exists(path):
        print(f"[migrate] decisions: missing {path}, skipping")
        return
    conn = _get_conn()
    cur = conn.cursor()
    inserted = 0
    seen = 0
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            seen += 1
            env = (row.get("env") or "").strip()
            trade_date = (row.get("trade_date") or "").strip()
            city = (row.get("city") or "").strip()
            series = (row.get("series_ticker") or "").strip()
            event = (row.get("event_ticker") or "").strip()
            run_ts = (row.get("run_ts") or "").strip()
            if not (env and trade_date and city and series and event and run_ts):
                continue
            try:
                cur.execute(
                    """
                    SELECT 1
                    FROM decisions d
                    JOIN cities c ON d.city_id = c.id
                    WHERE d.env = %s
                      AND d.trade_date = %s
                      AND c.code = %s
                      AND d.series_ticker = %s
                      AND d.event_ticker = %s
                      AND d.run_ts = %s
                    LIMIT 1
                    """,
                    (env, trade_date, city, series, event, run_ts),
                )
                if cur.fetchone():
                    continue
                db.insert_decision_row(row)
                inserted += 1
            except Exception as e:
                conn.rollback()
                print(f"[migrate] decisions: skipping malformed row (seen={seen}): {e}")
                continue
    print(f"[migrate] decisions: {inserted} inserted (rows seen={seen})")


def migrate_eval_history(path: str) -> None:
    if not _file_exists(path):
        print(f"[migrate] eval_events: missing {path}, skipping")
        return
    conn = _get_conn()
    cur = conn.cursor()
    inserted = 0
    seen = 0
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            seen += 1
            env = (row.get("env") or "").strip()
            trade_date = (row.get("trade_date") or "").strip()
            city = (row.get("city") or "").strip()
            event = (row.get("event_ticker") or "").strip()
            run_ts = (row.get("run_ts") or "").strip()
            if not (env and trade_date and city and event and run_ts):
                continue
            try:
                cur.execute(
                    """
                    SELECT 1
                    FROM eval_events e
                    JOIN cities c ON e.city_id = c.id
                    WHERE e.env = %s
                      AND e.trade_date = %s
                      AND c.code = %s
                      AND e.event_ticker = %s
                      AND e.run_ts = %s
                    LIMIT 1
                    """,
                    (env, trade_date, city, event, run_ts),
                )
                if cur.fetchone():
                    continue
                db.insert_eval_event_row(row)
                inserted += 1
            except Exception as e:
                conn.rollback()
                print(f"[migrate] eval_events: skipping malformed row (seen={seen}): {e}")
                continue
    print(f"[migrate] eval_events: {inserted} inserted (rows seen={seen})")


def migrate_daily_metrics(path: str) -> None:
    if not _file_exists(path):
        print(f"[migrate] eval_metrics: missing {path}, skipping")
        return
    conn = _get_conn()
    cur = conn.cursor()
    inserted = 0
    seen = 0
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            seen += 1
            trade_date = (row.get("trade_date") or "").strip()
            city = (row.get("city") or "").strip()
            metric_type = (row.get("metric_type") or "").strip()
            source_name = (row.get("source_name") or "").strip()
            if not (trade_date and city and metric_type and source_name):
                continue
            try:
                cur.execute(
                    """
                    SELECT 1
                    FROM eval_metrics m
                    JOIN cities c ON m.city_id = c.id
                    WHERE m.trade_date = %s
                      AND c.code = %s
                      AND m.metric_type = %s
                      AND m.source_name = %s
                    LIMIT 1
                    """,
                    (trade_date, city, metric_type, source_name),
                )
                if cur.fetchone():
                    continue
                db.insert_eval_metric_row(row)
                inserted += 1
            except Exception as e:
                conn.rollback()
                print(f"[migrate] eval_metrics: skipping malformed row (seen={seen}): {e}")
                continue
    print(f"[migrate] eval_metrics: {inserted} inserted (rows seen={seen})")


def migrate_source_performance(path: str) -> None:
    if not _file_exists(path):
        print(f"[migrate] source_performance: missing {path}, skipping")
        return
    conn = _get_conn()
    cur = conn.cursor()
    inserted = 0
    seen = 0
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            seen += 1
            date = (row.get("date") or "").strip()
            city = (row.get("city") or "").strip()
            source_name = (row.get("source_name") or "").strip()
            if not (date and city and source_name):
                continue
            try:
                cur.execute(
                    """
                    SELECT 1
                    FROM source_performance sp
                    JOIN cities c ON sp.city_id = c.id
                    WHERE sp.date = %s
                      AND c.code = %s
                      AND sp.source_name = %s
                    LIMIT 1
                    """,
                    (date, city, source_name),
                )
                if cur.fetchone():
                    continue
                db.insert_source_performance_row(row)
                inserted += 1
            except Exception as e:
                conn.rollback()
                print(f"[migrate] source_performance: skipping malformed row (seen={seen}): {e}")
                continue
    print(f"[migrate] source_performance: {inserted} inserted (rows seen={seen})")


def migrate_weights_history(path: str) -> None:
    if not _file_exists(path):
        print(f"[migrate] weights_history: missing {path}, skipping")
        return
    conn = _get_conn()
    cur = conn.cursor()
    inserted = 0
    seen = 0
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            seen += 1
            as_of = (row.get("as_of") or "").strip()
            city = (row.get("city") or "").strip()
            window_days = (row.get("window_days") or "").strip()
            if not (as_of and city):
                continue
            try:
                cur.execute(
                    """
                    SELECT 1
                    FROM weights_history w
                    JOIN cities c ON w.city_id = c.id
                    WHERE w.as_of = %s
                      AND c.code = %s
                      AND COALESCE(w.window_days::text, '') = %s
                    LIMIT 1
                    """,
                    (as_of, city, window_days),
                )
                if cur.fetchone():
                    continue
                db.insert_weights_history_row(row)
                inserted += 1
            except Exception as e:
                conn.rollback()
                print(f"[migrate] weights_history: skipping malformed row (seen={seen}): {e}")
                continue
    print(f"[migrate] weights_history: {inserted} inserted (rows seen={seen})")


def migrate_intraday(path: str) -> None:
    if not _file_exists(path):
        print(f"[migrate] intraday_snapshots: missing {path}, skipping")
        return
    conn = _get_conn()
    cur = conn.cursor()
    inserted = 0
    seen = 0
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            seen += 1
            ts = (row.get("timestamp") or "").strip()
            city = (row.get("city") or "").strip()
            trade_date = (row.get("trade_date") or "").strip()
            if not (ts and city and trade_date):
                continue
            try:
                cur.execute(
                    """
                    SELECT 1
                    FROM intraday_snapshots s
                    JOIN cities c ON s.city_id = c.id
                    WHERE s.trade_date = %s
                      AND c.code = %s
                      AND s.snapshot_ts = %s
                    LIMIT 1
                    """,
                    (trade_date, city, ts),
                )
                if cur.fetchone():
                    continue
                db.insert_intraday_snapshot_row(row)
                inserted += 1
            except Exception as e:
                conn.rollback()
                print(f"[migrate] intraday_snapshots: skipping malformed row (seen={seen}): {e}")
                continue
    print(f"[migrate] intraday_snapshots: {inserted} inserted (rows seen={seen})")


def migrate_predictions_history(path: str) -> None:
    if not _file_exists(path):
        print(f"[migrate] predictions: missing {path}, skipping")
        return
    conn = _get_conn()
    cur = conn.cursor()
    inserted = 0
    seen = 0
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            seen += 1
            date = (row.get("date") or "").strip()
            city = (row.get("city") or "").strip()
            run_ts = (row.get("run_ts") or "").strip() or (row.get("timestamp") or "").strip()
            if not (date and city and run_ts):
                continue
            try:
                cur.execute(
                    """
                    SELECT 1
                    FROM predictions p
                    JOIN cities c ON p.city_id = c.id
                    WHERE p.trade_date = %s
                      AND c.code = %s
                      AND p.run_ts = %s
                    LIMIT 1
                    """,
                    (date, city, run_ts),
                )
                if cur.fetchone():
                    continue
                row = dict(row)
                row.setdefault("run_ts", run_ts)
                db.insert_prediction_row(row)
                inserted += 1
            except Exception as e:
                conn.rollback()
                print(f"[migrate] predictions: skipping malformed row (seen={seen}): {e}")
                continue
    print(f"[migrate] predictions: {inserted} inserted (rows seen={seen})")


if __name__ == "__main__":
    args = _parse_args()
    # Ensure we can connect before doing any I/O.
    _get_conn()
    migrate_predictions_history(args.predictions_history)
    migrate_intraday(args.intraday_forecasts)
    migrate_trades(args.trades_history)
    migrate_decisions(args.decisions_history)
    migrate_eval_history(args.eval_history)
    migrate_daily_metrics(args.daily_metrics)
    migrate_source_performance(args.source_performance)
    migrate_weights_history(args.weights_history)

