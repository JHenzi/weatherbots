import csv
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _iso_sort_key(value: Optional[str]) -> float:
    text = (value or "").strip()
    if not text:
        return float("-inf")
    try:
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return float("-inf")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.timestamp()


def _coerce_float(value: Any) -> Optional[float]:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> Optional[int]:
    num = _coerce_float(value)
    if num is None:
        return None
    try:
        return int(num)
    except (TypeError, ValueError):
        return None


def _csv_truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def build_recent_trade_log(
    *,
    decisions_path: Path,
    trades_path: Path,
    city_order: Sequence[str],
    days: int = 3,
    env: str = "prod",
) -> Dict[str, Any]:
    try:
        days = int(days)
    except (TypeError, ValueError):
        days = 3
    days = max(1, min(days, 14))
    env_filter = (env or "").strip()

    decisions = [
        row for row in read_csv(decisions_path)
        if not env_filter or (row.get("env") or "").strip() == env_filter
    ]
    trades = [
        row for row in read_csv(trades_path)
        if not env_filter or (row.get("env") or "").strip() == env_filter
    ]

    all_dates = sorted(
        {
            (row.get("trade_date") or "").strip()
            for row in (decisions + trades)
            if (row.get("trade_date") or "").strip()
        }
    )
    if not all_dates:
        return {"env": env_filter or None, "requested_days": days, "groups": []}

    recent_dates = all_dates[-days:]
    recent_dates_set = set(recent_dates)
    decisions_by_key: Dict[tuple, List[Dict[str, str]]] = {}
    trades_by_key: Dict[tuple, List[Dict[str, str]]] = {}

    for row in decisions:
        trade_date = (row.get("trade_date") or "").strip()
        city = (row.get("city") or "").strip().lower()
        if trade_date in recent_dates_set and city:
            decisions_by_key.setdefault((trade_date, city), []).append(row)

    for row in trades:
        trade_date = (row.get("trade_date") or "").strip()
        city = (row.get("city") or "").strip().lower()
        if trade_date in recent_dates_set and city:
            trades_by_key.setdefault((trade_date, city), []).append(row)

    city_rank = {city: idx for idx, city in enumerate(city_order)}
    groups: List[Dict[str, Any]] = []
    for trade_date in reversed(recent_dates):
        cities = {
            key[1] for key in decisions_by_key.keys() if key[0] == trade_date
        } | {
            key[1] for key in trades_by_key.keys() if key[0] == trade_date
        }
        items: List[Dict[str, Any]] = []
        for city in sorted(cities, key=lambda x: (city_rank.get(x, len(city_rank)), x)):
            decision_rows = sorted(
                decisions_by_key.get((trade_date, city), []),
                key=lambda row: (_iso_sort_key(row.get("run_ts")), row.get("run_ts") or ""),
            )
            trade_rows = sorted(
                trades_by_key.get((trade_date, city), []),
                key=lambda row: (_iso_sort_key(row.get("run_ts")), row.get("run_ts") or ""),
            )
            live_trade_rows = [row for row in trade_rows if _csv_truthy(row.get("send_orders"))]
            latest_trade = live_trade_rows[-1] if live_trade_rows else (trade_rows[-1] if trade_rows else None)
            trade_decisions = [
                row for row in decision_rows
                if (row.get("decision") or "").strip().lower() == "trade"
            ]
            primary_decision = trade_decisions[-1] if trade_decisions else (decision_rows[-1] if decision_rows else None)

            if live_trade_rows:
                status = "executed"
            elif latest_trade is not None or (
                primary_decision is not None
                and (primary_decision.get("decision") or "").strip().lower() == "trade"
            ):
                status = "planned"
            else:
                status = "skipped"

            decision_value = ""
            if primary_decision is not None:
                decision_value = (primary_decision.get("decision") or "").strip().lower()
            elif latest_trade is not None:
                decision_value = "trade"

            items.append(
                {
                    "trade_date": trade_date,
                    "city": city,
                    "decision": decision_value,
                    "status": status,
                    "run_ts": (
                        (primary_decision or {}).get("run_ts")
                        or (latest_trade or {}).get("run_ts")
                        or ""
                    ),
                    "attempt_count": len(decision_rows),
                    "reason": (
                        (primary_decision or {}).get("reason")
                        or ("trade_logged" if latest_trade is not None else "")
                    ),
                    "pred_tmax_f": _coerce_float(
                        (primary_decision or {}).get("pred_tmax_f")
                        or (latest_trade or {}).get("pred_tmax_f")
                    ),
                    "spread_f": _coerce_float((primary_decision or {}).get("spread_f")),
                    "confidence_score": _coerce_float((primary_decision or {}).get("confidence_score")),
                    "trade": (
                        None
                        if latest_trade is None
                        else {
                            "run_ts": latest_trade.get("run_ts") or "",
                            "market_ticker": latest_trade.get("market_ticker") or "",
                            "market_subtitle": latest_trade.get("market_subtitle") or "",
                            "side": (latest_trade.get("side") or "").strip().lower(),
                            "count": _coerce_int(latest_trade.get("count")),
                            "yes_price": _coerce_int(latest_trade.get("yes_price")),
                            "no_price": _coerce_int(latest_trade.get("no_price")),
                            "send_orders": _csv_truthy(latest_trade.get("send_orders")),
                        }
                    ),
                }
            )
        groups.append({"trade_date": trade_date, "items": items})

    return {
        "env": env_filter or None,
        "requested_days": days,
        "groups": groups,
    }
