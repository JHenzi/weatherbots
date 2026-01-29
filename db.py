import os
from typing import Any, Dict, Mapping, Optional

import psycopg2
from psycopg2.extras import Json


"""
Lightweight Postgres access layer for weather-trader.

Goals:
- Centralize connection handling via PGDATABASE_URL / DATABASE_URL.
- Provide small helpers for inserting rows that mirror the CSV log schemas.
- Be safe to disable via ENABLE_PG_WRITE (no-ops when false/misconfigured).
"""


_PG_CONN = None  # type: ignore[var-annotated]
_PG_DISABLED = False
_PG_ERROR_LOGGED = False


def _pg_enabled() -> bool:
    """
    Global gate for Postgres writes.

    - ENABLE_PG_WRITE=1/true/yes/y enables writes.
    - Missing/false => all insert helpers become no-ops.
    """
    flag = os.getenv("ENABLE_PG_WRITE", "")
    return str(flag).strip().lower() in ("1", "true", "yes", "y")


def _get_dsn() -> Optional[str]:
    url = os.getenv("PGDATABASE_URL") or os.getenv("DATABASE_URL")
    if url:
        return str(url).strip()

    # Try to determine if we are inside docker or on host.
    # 'postgres' is the service name in docker-compose.
    import socket

    host = "postgres"
    port = 5432
    try:
        socket.gethostbyname(host)
    except socket.gaierror:
        # Fallback for host-based execution
        host = "localhost"
        port = 5433  # Mapped port in docker-compose.yml

    return f"postgresql://weather:weather@{host}:{port}/weather"


def _log_once(msg: str) -> None:
    global _PG_ERROR_LOGGED
    if not _PG_ERROR_LOGGED:
        print(f"[db] {msg}")
        _PG_ERROR_LOGGED = True


def _get_conn():
    """
    Return a cached psycopg2 connection, or None if disabled/misconfigured.
    """
    global _PG_CONN, _PG_DISABLED
    if _PG_DISABLED or not _pg_enabled():
        return None

    if _PG_CONN is not None:
        return _PG_CONN

    dsn = _get_dsn()
    if not dsn:
        _PG_DISABLED = True
        _log_once("PGDATABASE_URL/DATABASE_URL not set; skipping Postgres writes.")
        return None

    try:
        _PG_CONN = psycopg2.connect(dsn)
    except Exception as e:  # pragma: no cover - defensive
        _PG_DISABLED = True
        _log_once(f"Failed to connect to Postgres ({e}); disabling ENABLE_PG_WRITE for this run.")
        return None
    return _PG_CONN


def _with_cursor(fn) -> None:
    """
    Helper to run a small unit of work with a cursor and commit on success.
    Swallows connection errors after logging once, so the caller's main job
    (CSV writing, trading, etc.) is never blocked by Postgres issues.
    """
    conn = _get_conn()
    if conn is None:
        return
    try:
        with conn.cursor() as cur:
            fn(conn, cur)
        conn.commit()
    except Exception as e:  # pragma: no cover - defensive
        _log_once(f"Error during Postgres write ({e}); further writes will be skipped this run.")
        try:
            conn.rollback()
        except Exception:
            pass


_CITY_NAMES: Dict[str, str] = {
    "ny": "New York",
    "il": "Chicago",
    "tx": "Austin",
    "fl": "Miami",
}


def _ensure_city_id(cur, code: str | None) -> Optional[int]:
    """
    Resolve a city code (ny/il/tx/fl) to a cities.id, creating the row if needed.
    Returns None when code is empty.
    """
    if not code:
        return None
    c = str(code).strip().lower()
    if not c:
        return None
    name = _CITY_NAMES.get(c, c.upper())
    cur.execute(
        """
        INSERT INTO cities (code, name)
        VALUES (%s, %s)
        ON CONFLICT (code) DO UPDATE SET name = EXCLUDED.name
        RETURNING id
        """,
        (c, name),
    )
    row = cur.fetchone()
    return int(row[0]) if row and row[0] is not None else None


def insert_trade_row(row: Mapping[str, Any]) -> None:
    """
    Mirror of trades_history.csv writer.
    Expected keys: run_ts, env, trade_date, city, series_ticker, event_ticker,
    market_ticker, market_subtitle, pred_tmax_f, side, count, yes_price, no_price,
    send_orders.
    """

    def _work(conn, cur) -> None:  # pragma: no cover - small wrapper
        city_id = _ensure_city_id(cur, row.get("city"))
        cur.execute(
            """
            INSERT INTO trades (
                run_ts,
                env,
                trade_date,
                city_id,
                series_ticker,
                event_ticker,
                market_ticker,
                market_subtitle,
                pred_tmax_f,
                side,
                count,
                yes_price,
                no_price,
                send_orders
            )
            VALUES (%(run_ts)s, %(env)s, %(trade_date)s, %(city_id)s,
                    %(series_ticker)s, %(event_ticker)s,
                    %(market_ticker)s, %(market_subtitle)s,
                    NULLIF(%(pred_tmax_f)s, '')::double precision,
                    %(side)s,
                    %(count)s,
                    %(yes_price)s,
                    %(no_price)s,
                    %(send_orders)s)
            """,
            {
                "run_ts": row.get("run_ts"),
                "env": row.get("env"),
                "trade_date": row.get("trade_date"),
                "city_id": city_id,
                "series_ticker": row.get("series_ticker"),
                "event_ticker": row.get("event_ticker"),
                "market_ticker": row.get("market_ticker"),
                "market_subtitle": row.get("market_subtitle"),
                "pred_tmax_f": row.get("pred_tmax_f"),
                "side": row.get("side"),
                "count": row.get("count"),
                "yes_price": row.get("yes_price"),
                "no_price": row.get("no_price"),
                "send_orders": bool(row.get("send_orders")),
            },
        )

    _with_cursor(_work)


def insert_decision_row(row: Mapping[str, Any]) -> None:
    """
    Mirror of decisions_history.csv writer.
    Expected keys: run_ts, env, trade_date, city, series_ticker, event_ticker,
    pred_tmax_f, spread_f, confidence_score, decision, reason.
    """

    def _work(conn, cur) -> None:  # pragma: no cover - small wrapper
        city_id = _ensure_city_id(cur, row.get("city"))
        cur.execute(
            """
            INSERT INTO decisions (
                run_ts,
                env,
                trade_date,
                city_id,
                series_ticker,
                event_ticker,
                pred_tmax_f,
                spread_f,
                confidence_score,
                decision,
                reason
            )
            VALUES (%(run_ts)s, %(env)s, %(trade_date)s, %(city_id)s,
                    %(series_ticker)s, %(event_ticker)s,
                    NULLIF(%(pred_tmax_f)s, '')::double precision,
                    NULLIF(%(spread_f)s, '')::double precision,
                    NULLIF(%(confidence_score)s, '')::double precision,
                    %(decision)s,
                    %(reason)s)
            """,
            {
                "run_ts": row.get("run_ts"),
                "env": row.get("env"),
                "trade_date": row.get("trade_date"),
                "city_id": city_id,
                "series_ticker": row.get("series_ticker"),
                "event_ticker": row.get("event_ticker"),
                "pred_tmax_f": row.get("pred_tmax_f"),
                "spread_f": row.get("spread_f"),
                "confidence_score": row.get("confidence_score"),
                "decision": row.get("decision"),
                "reason": row.get("reason"),
            },
        )

    _with_cursor(_work)


def insert_eval_event_row(row: Mapping[str, Any]) -> None:
    """
    Mirror of eval_history.csv writer.
    Expects the keys produced by kalshi_trader._append_eval_row.
    """

    def _work(conn, cur) -> None:  # pragma: no cover - small wrapper
        city_id = _ensure_city_id(cur, row.get("city"))
        cur.execute(
            """
            INSERT INTO eval_events (
                run_ts,
                env,
                trade_date,
                city_id,
                series_ticker,
                event_ticker,
                decision,
                reason,
                mu_tmax_f,
                sigma_f,
                spread_f,
                confidence_score,
                tmax_open_meteo,
                tmax_visual_crossing,
                tmax_tomorrow,
                tmax_weatherapi,
                tmax_lstm,
                sources_used,
                weights_used,
                chosen_market_ticker,
                chosen_market_subtitle,
                bucket_lo,
                bucket_hi,
                model_prob_yes,
                yes_ask,
                yes_bid,
                yes_spread,
                ask_qty,
                market_prob_yes,
                edge_prob,
                ev_cents,
                count,
                send_orders
            )
            VALUES (
                %(run_ts)s,
                %(env)s,
                %(trade_date)s,
                %(city_id)s,
                %(series_ticker)s,
                %(event_ticker)s,
                %(decision)s,
                %(reason)s,
                NULLIF(%(mu_tmax_f)s, '')::double precision,
                NULLIF(%(sigma_f)s, '')::double precision,
                NULLIF(%(spread_f)s, '')::double precision,
                NULLIF(%(confidence_score)s, '')::double precision,
                NULLIF(%(tmax_open_meteo)s, '')::double precision,
                NULLIF(%(tmax_visual_crossing)s, '')::double precision,
                NULLIF(%(tmax_tomorrow)s, '')::double precision,
                NULLIF(%(tmax_weatherapi)s, '')::double precision,
                NULLIF(%(tmax_lstm)s, '')::double precision,
                %(sources_used)s,
                %(weights_used)s,
                %(chosen_market_ticker)s,
                %(chosen_market_subtitle)s,
                NULLIF(%(bucket_lo)s, '')::double precision,
                NULLIF(%(bucket_hi)s, '')::double precision,
                NULLIF(%(model_prob_yes)s, '')::double precision,
                NULLIF(%(yes_ask)s, '')::integer,
                NULLIF(%(yes_bid)s, '')::integer,
                NULLIF(%(yes_spread)s, '')::integer,
                NULLIF(%(ask_qty)s, '')::integer,
                NULLIF(%(market_prob_yes)s, '')::double precision,
                NULLIF(%(edge_prob)s, '')::double precision,
                NULLIF(%(ev_cents)s, '')::double precision,
                NULLIF(%(count)s, '')::integer,
                %(send_orders)s
            )
            """,
            {
                "run_ts": row.get("run_ts"),
                "env": row.get("env"),
                "trade_date": row.get("trade_date"),
                "city_id": city_id,
                "series_ticker": row.get("series_ticker"),
                "event_ticker": row.get("event_ticker"),
                "decision": row.get("decision"),
                "reason": row.get("reason"),
                "mu_tmax_f": row.get("mu_tmax_f"),
                "sigma_f": row.get("sigma_f"),
                "spread_f": row.get("spread_f"),
                "confidence_score": row.get("confidence_score"),
                "tmax_open_meteo": row.get("tmax_open_meteo"),
                "tmax_visual_crossing": row.get("tmax_visual_crossing"),
                "tmax_tomorrow": row.get("tmax_tomorrow"),
                "tmax_weatherapi": row.get("tmax_weatherapi"),
                "tmax_lstm": row.get("tmax_lstm"),
                "sources_used": row.get("sources_used"),
                "weights_used": row.get("weights_used"),
                "chosen_market_ticker": row.get("chosen_market_ticker"),
                "chosen_market_subtitle": row.get("chosen_market_subtitle"),
                "bucket_lo": row.get("bucket_lo"),
                "bucket_hi": row.get("bucket_hi"),
                "model_prob_yes": row.get("model_prob_yes"),
                "yes_ask": row.get("yes_ask"),
                "yes_bid": row.get("yes_bid"),
                "yes_spread": row.get("yes_spread"),
                "ask_qty": row.get("ask_qty"),
                "market_prob_yes": row.get("market_prob_yes"),
                "edge_prob": row.get("edge_prob"),
                "ev_cents": row.get("ev_cents"),
                "count": row.get("count"),
                "send_orders": bool(row.get("send_orders")),
            },
        )

    _with_cursor(_work)


def insert_intraday_snapshot_row(row: Mapping[str, Any]) -> None:
    """
    Mirror of intraday_forecasts.csv rows.
    Expected keys include:
      - timestamp, city, trade_date, mean_forecast, current_sigma,
        tmax_open_meteo, tmax_visual_crossing, tmax_tomorrow, tmax_weatherapi,
        tmax_google_weather, tmax_openweathermap, tmax_pirateweather,
        tmax_weather_gov, sources_used, weights_used.
    Provider values are packed into provider_values JSONB.
    """

    def _work(conn, cur) -> None:  # pragma: no cover - small wrapper
        city_id = _ensure_city_id(cur, row.get("city"))
        provider_values = {
            "open-meteo": row.get("tmax_open_meteo"),
            "visual-crossing": row.get("tmax_visual_crossing"),
            "tomorrow": row.get("tmax_tomorrow"),
            "weatherapi": row.get("tmax_weatherapi"),
            "google-weather": row.get("tmax_google_weather"),
            "openweathermap": row.get("tmax_openweathermap"),
            "pirateweather": row.get("tmax_pirateweather"),
            "weather.gov": row.get("tmax_weather_gov"),
        }
        cur.execute(
            """
            INSERT INTO intraday_snapshots (
                city_id,
                trade_date,
                snapshot_ts,
                mean_forecast,
                current_sigma,
                provider_values,
                sources_used,
                weights_used
            )
            VALUES (
                %(city_id)s,
                %(trade_date)s,
                %(snapshot_ts)s,
                NULLIF(%(mean_forecast)s, '')::double precision,
                NULLIF(%(current_sigma)s, '')::double precision,
                %(provider_values)s,
                %(sources_used)s,
                %(weights_used)s
            )
            """,
            {
                "city_id": city_id,
                "trade_date": row.get("trade_date"),
                "snapshot_ts": row.get("timestamp"),
                "mean_forecast": row.get("mean_forecast"),
                "current_sigma": row.get("current_sigma"),
                "provider_values": Json(provider_values),
                "sources_used": row.get("sources_used"),
                "weights_used": row.get("weights_used"),
            },
        )

    _with_cursor(_work)


def insert_prediction_row(row: Mapping[str, Any]) -> None:
    """
    Insert a single predictions_history-style row into predictions.
    Expected keys: date, city, run_ts, env, tmax_predicted, tmax_lstm,
    tmax_forecast, spread_f, confidence_score, conviction_score,
    forecast_sources, tmax_* provider columns, sources_used, weights_used,
    prediction_mode, blend_forecast_weight, refresh_history, retrain_lstm.
    """

    def _work(conn, cur) -> None:  # pragma: no cover - small wrapper
        city_id = _ensure_city_id(cur, row.get("city"))
        provider_values = {
            "open-meteo": row.get("tmax_open_meteo"),
            "visual-crossing": row.get("tmax_visual_crossing"),
            "tomorrow": row.get("tmax_tomorrow"),
            "weatherapi": row.get("tmax_weatherapi"),
            "google-weather": row.get("tmax_google_weather"),
            "openweathermap": row.get("tmax_openweathermap"),
            "pirateweather": row.get("tmax_pirateweather"),
            "weather.gov": row.get("tmax_weather_gov"),
        }
        cur.execute(
            """
            INSERT INTO predictions (
                city_id,
                trade_date,
                run_ts,
                env,
                tmax_predicted,
                tmax_lstm,
                tmax_forecast,
                spread_f,
                confidence_score,
                conviction_score,
                forecast_sources,
                provider_values,
                sources_used,
                weights_used,
                prediction_mode,
                blend_forecast_weight,
                refresh_history,
                retrain_lstm
            )
            VALUES (
                %(city_id)s,
                %(trade_date)s,
                %(run_ts)s,
                %(env)s,
                NULLIF(%(tmax_predicted)s, '')::double precision,
                NULLIF(%(tmax_lstm)s, '')::double precision,
                NULLIF(%(tmax_forecast)s, '')::double precision,
                NULLIF(%(spread_f)s, '')::double precision,
                NULLIF(%(confidence_score)s, '')::double precision,
                NULLIF(%(conviction_score)s, '')::double precision,
                %(forecast_sources)s,
                %(provider_values)s,
                %(sources_used)s,
                %(weights_used)s,
                %(prediction_mode)s,
                NULLIF(%(blend_forecast_weight)s, '')::double precision,
                %(refresh_history)s,
                %(retrain_lstm)s
            )
            """,
            {
                "city_id": city_id,
                "trade_date": row.get("date"),
                "run_ts": row.get("run_ts") or row.get("timestamp"),
                "env": row.get("env"),
                "tmax_predicted": row.get("tmax_predicted"),
                "tmax_lstm": row.get("tmax_lstm"),
                "tmax_forecast": row.get("tmax_forecast"),
                "spread_f": row.get("spread_f"),
                "confidence_score": row.get("confidence_score"),
                "conviction_score": row.get("conviction_score"),
                "forecast_sources": row.get("forecast_sources"),
                "provider_values": Json(provider_values),
                "sources_used": row.get("sources_used"),
                "weights_used": row.get("weights_used"),
                "prediction_mode": row.get("prediction_mode"),
                "blend_forecast_weight": row.get("blend_forecast_weight"),
                "refresh_history": _coerce_bool(row.get("refresh_history")),
                "retrain_lstm": _coerce_bool(row.get("retrain_lstm")),
            },
        )

    _with_cursor(_work)


def insert_eval_metric_row(row: Mapping[str, Any]) -> None:
    """
    Mirror of daily_metrics.csv writer.
    Expected keys: run_ts, trade_date, city, metric_type, source_name, value.
    """

    def _work(conn, cur) -> None:  # pragma: no cover - small wrapper
        city_id = _ensure_city_id(cur, row.get("city"))
        cur.execute(
            """
            INSERT INTO eval_metrics (
                run_ts,
                trade_date,
                city_id,
                metric_type,
                source_name,
                value
            )
            VALUES (
                %(run_ts)s,
                %(trade_date)s,
                %(city_id)s,
                %(metric_type)s,
                %(source_name)s,
                NULLIF(%(value)s, '')::double precision
            )
            """,
            {
                "run_ts": row.get("run_ts"),
                "trade_date": row.get("trade_date"),
                "city_id": city_id,
                "metric_type": row.get("metric_type"),
                "source_name": row.get("source_name"),
                "value": row.get("value"),
            },
        )

    _with_cursor(_work)


def insert_source_performance_row(row: Mapping[str, Any]) -> None:
    """
    Mirror of source_performance.csv writer.
    Expected keys: date, city, source_name, predicted_tmax, actual_tmax, absolute_error.
    """

    def _work(conn, cur) -> None:  # pragma: no cover - small wrapper
        city_id = _ensure_city_id(cur, row.get("city"))
        cur.execute(
            """
            INSERT INTO source_performance (
                date,
                city_id,
                source_name,
                predicted_tmax,
                actual_tmax,
                absolute_error
            )
            VALUES (
                %(date)s,
                %(city_id)s,
                %(source_name)s,
                NULLIF(%(predicted_tmax)s, '')::double precision,
                NULLIF(%(actual_tmax)s, '')::double precision,
                NULLIF(%(absolute_error)s, '')::double precision
            )
            """,
            {
                "date": row.get("date"),
                "city_id": city_id,
                "source_name": row.get("source_name"),
                "predicted_tmax": row.get("predicted_tmax"),
                "actual_tmax": row.get("actual_tmax"),
                "absolute_error": row.get("absolute_error"),
            },
        )

    _with_cursor(_work)


def insert_weights_history_row(row: Mapping[str, Any]) -> None:
    """
    Mirror of weights_history.csv writer.
    Expected keys: run_ts, as_of, city, window_days, weights_json.
    """

    def _work(conn, cur) -> None:  # pragma: no cover - small wrapper
        city_id = _ensure_city_id(cur, row.get("city"))
        weights = row.get("weights_json") or "{}"
        cur.execute(
            """
            INSERT INTO weights_history (
                run_ts,
                as_of,
                city_id,
                window_days,
                weights
            )
            VALUES (
                %(run_ts)s,
                %(as_of)s,
                %(city_id)s,
                NULLIF(%(window_days)s, '')::integer,
                %(weights)s
            )
            """,
            {
                "run_ts": row.get("run_ts"),
                "as_of": row.get("as_of"),
                "city_id": city_id,
                "window_days": row.get("window_days"),
                "weights": Json(_parse_json_safe(weights)),
            },
        )

    _with_cursor(_work)


def _parse_json_safe(raw: Any) -> Dict[str, Any]:
    import json

    if isinstance(raw, dict):
        return raw  # type: ignore[return-value]
    try:
        return json.loads(str(raw))
    except Exception:
        return {}


def _coerce_bool(x: Any) -> Optional[bool]:
    if x is None or x == "":
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("1", "true", "yes", "y"):
        return True
    if s in ("0", "false", "no", "n"):
        return False
    return None

