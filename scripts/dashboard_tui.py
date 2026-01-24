import argparse
import csv
import datetime as dt
import os
import time
from dataclasses import dataclass
from typing import Any


def _truthy(x: Any) -> bool:
    s = str(x or "").strip().lower()
    return s in ("1", "true", "yes", "y")


def _read_csv(path: str) -> list[dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        return [row for row in r if row]


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(str(x).strip()))
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(str(x).strip())
    except Exception:
        return default


def _fmt_money(x: float) -> str:
    return f"${x:,.2f}"


def _fmt_dt(ts: str) -> str:
    s = (ts or "").strip()
    if not s:
        return ""
    return s.replace("T", " ").replace("+00:00", "Z")


@dataclass(frozen=True)
class Schedule:
    trade_min: int
    trade_hour: int
    calibrate_min: int
    calibrate_hour: int
    settle_min: int
    settle_hour: int


def _parse_crontab(path: str) -> Schedule | None:
    """
    Parse ops/docker/crontab for the 3 daily jobs we define:
      - run_trade.sh
      - run_calibrate.sh
      - run_settle.sh
    Expected format: "M H * * * ...run_*.sh"
    """
    if not os.path.exists(path):
        return None
    trade = None
    cal = None
    settle = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 6:
                continue
            try:
                minute = int(parts[0])
                hour = int(parts[1])
            except Exception:
                continue
            cmd = " ".join(parts[5:])
            if "run_trade.sh" in cmd:
                trade = (minute, hour)
            elif "run_calibrate.sh" in cmd:
                cal = (minute, hour)
            elif "run_settle.sh" in cmd:
                settle = (minute, hour)
    if not (trade and cal and settle):
        return None
    return Schedule(
        trade_min=trade[0],
        trade_hour=trade[1],
        calibrate_min=cal[0],
        calibrate_hour=cal[1],
        settle_min=settle[0],
        settle_hour=settle[1],
    )


def _next_daily(now: dt.datetime, hour: int, minute: int) -> dt.datetime:
    cand = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if cand <= now:
        cand = cand + dt.timedelta(days=1)
    return cand


def _get_tz() -> dt.tzinfo:
    tzname = os.getenv("TZ", "").strip()
    if not tzname:
        return dt.datetime.now().astimezone().tzinfo or dt.timezone.utc
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(tzname)
    except Exception:
        return dt.datetime.now().astimezone().tzinfo or dt.timezone.utc


def _draw_section(stdscr, y: int, title: str, lines: list[str], width: int) -> int:
    h, w = stdscr.getmaxyx()
    if y >= h:
        return y
    try:
        stdscr.addstr(y, 0, title[: max(0, min(width, w - 1))])
    except Exception:
        return h  # give up drawing further
    y += 1
    for ln in lines:
        if y >= h:
            return y
        try:
            stdscr.addstr(y, 0, ln[: max(0, min(width, w - 1))])
        except Exception:
            return h
        y += 1
    y += 1
    return y


def _tabulate(headers: list[str], rows: list[list[Any]], width: int) -> list[str]:
    if not rows:
        return ["(none)"]
    cols = len(headers)
    w = [len(h) for h in headers]
    for row in rows:
        for i in range(cols):
            w[i] = min(max(w[i], len(str(row[i]))), 30)
    def fmt_row(row):
        parts = []
        for i in range(cols):
            parts.append(str(row[i]).ljust(w[i])[: w[i]])
        s = "  ".join(parts)
        return s[:width]
    out = [fmt_row(headers), fmt_row(["-" * min(30, x) for x in w])]
    out.extend(fmt_row(r) for r in rows)
    return out


def _render(env: str, limit: int) -> dict[str, Any]:
    pred_path = "Data/predictions_latest.csv"
    trades_path = "Data/trades_history.csv"
    decisions_path = "Data/decisions_history.csv"
    eval_path = "Data/eval_history.csv"

    preds = _read_csv(pred_path)
    trades = _read_csv(trades_path)
    decisions = _read_csv(decisions_path)
    evals = _read_csv(eval_path)

    live_trades = [t for t in trades if (t.get("env") or "").strip() == env and _truthy(t.get("send_orders"))]
    live_trades.sort(key=lambda r: (r.get("run_ts") or ""))

    env_decisions = [d for d in decisions if (d.get("env") or "").strip() == env]
    env_decisions.sort(key=lambda r: (r.get("run_ts") or ""))

    env_evals = [e for e in evals if (e.get("env") or "").strip() == env]
    env_evals.sort(key=lambda r: (r.get("run_ts") or ""))

    # Settled index (date, city) for live sends
    settled = set()
    for e in env_evals:
        if not _truthy(e.get("send_orders")):
            continue
        if (e.get("settlement_tmax_f") or "").strip():
            settled.add(((e.get("trade_date") or "").strip(), (e.get("city") or "").strip()))

    open_trades = []
    for t in live_trades:
        key = ((t.get("trade_date") or "").strip(), (t.get("city") or "").strip())
        if key in settled:
            continue
        qty = _safe_int(t.get("count"), 0)
        ask_c = _safe_int(t.get("yes_price"), 0)
        cost = qty * (ask_c / 100.0)
        max_win = qty * ((100 - ask_c) / 100.0)
        open_trades.append(
            {
                "trade_date": t.get("trade_date", ""),
                "city": t.get("city", ""),
                "ticker": t.get("market_ticker", ""),
                "qty": qty,
                "ask_c": ask_c,
                "at_risk": cost,
                "max_win": max_win,
                "run_ts": t.get("run_ts", ""),
            }
        )

    # Latest predictions table
    pred_rows = []
    for p in preds:
        pred_rows.append(
            [
                p.get("date", ""),
                p.get("city", ""),
                f"{_safe_float(p.get('tmax_predicted'), 0.0):.2f}" if p.get("tmax_predicted") else "",
                f"{_safe_float(p.get('spread_f'), 0.0):.2f}" if p.get("spread_f") else "",
                f"{_safe_float(p.get('confidence_score'), 0.0):.2f}" if p.get("confidence_score") else "",
            ]
        )

    # Last live trades table
    trade_rows = []
    for t in live_trades[-limit:]:
        qty = _safe_int(t.get("count"), 0)
        ask_c = _safe_int(t.get("yes_price"), 0)
        cost = qty * (ask_c / 100.0)
        max_win = qty * ((100 - ask_c) / 100.0)
        trade_rows.append(
            [
                _fmt_dt(t.get("run_ts", "")),
                t.get("trade_date", ""),
                t.get("city", ""),
                t.get("market_ticker", ""),
                str(qty),
                str(ask_c),
                _fmt_money(cost),
                _fmt_money(max_win),
            ]
        )

    # Open trades table
    open_rows = []
    for t in sorted(open_trades, key=lambda x: (x["trade_date"], x["city"])):
        open_rows.append(
            [
                t["trade_date"],
                t["city"],
                t["ticker"],
                str(t["qty"]),
                str(t["ask_c"]),
                _fmt_money(t["at_risk"]),
                _fmt_money(t["max_win"]),
                _fmt_dt(t["run_ts"]),
            ]
        )

    # Last decisions
    decision_rows = []
    for d in env_decisions[-limit:]:
        decision_rows.append(
            [
                _fmt_dt(d.get("run_ts", "")),
                d.get("trade_date", ""),
                d.get("city", ""),
                d.get("decision", ""),
                (d.get("reason") or "")[:70],
            ]
        )

    # Next scheduled times
    tz = _get_tz()
    now = dt.datetime.now(tz)
    sched = _parse_crontab("ops/docker/crontab")
    next_lines = []
    if sched:
        nt = _next_daily(now, sched.trade_hour, sched.trade_min)
        nc = _next_daily(now, sched.calibrate_hour, sched.calibrate_min)
        ns = _next_daily(now, sched.settle_hour, sched.settle_min)
        next_lines = [
            f"Trade:     {nt.isoformat(timespec='minutes')}",
            f"Calibrate: {nc.isoformat(timespec='minutes')}",
            f"Settle:    {ns.isoformat(timespec='minutes')}",
            f"TZ: {getattr(tz, 'key', str(tz))}",
        ]
    else:
        next_lines = ["(could not parse ops/docker/crontab)"]

    # File row counts
    def count_rows(path: str) -> str:
        if not os.path.exists(path):
            return "missing"
        try:
            n = sum(1 for _ in open(path, "r", encoding="utf-8")) - 1
            return f"{n} rows"
        except Exception:
            return "present"

    files = {
        "predictions_latest": count_rows(pred_path),
        "trades_history": count_rows(trades_path),
        "decisions_history": count_rows(decisions_path),
        "eval_history": count_rows(eval_path),
    }

    return {
        "files": files,
        "pred_table": pred_rows,
        "trade_table": trade_rows,
        "open_table": open_rows,
        "decision_table": decision_rows,
        "next_lines": next_lines,
    }


def main():
    ap = argparse.ArgumentParser(description="Ncurses-like dashboard for weather-trader.")
    ap.add_argument("--env", type=str, default="prod")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument(
        "--city",
        type=str,
        default="",
        help="Optional: focus forecast comparison on one city code (ny/il/tx/fl). Default shows all cities.",
    )
    args = ap.parse_args()

    import curses

    def loop(stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(0)

        while True:
            h, w = stdscr.getmaxyx()
            stdscr.erase()

            payload = _render(args.env, int(args.limit))
            now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

            header = f"weather-trader status  env={args.env}  refresh={args.interval:.1f}s  now={now}   (q to quit)"
            try:
                stdscr.addstr(0, 0, header[: max(0, w - 1)])
            except Exception:
                pass
            file_line = (
                "files: "
                f"preds={payload['files']['predictions_latest']}, "
                f"trades={payload['files']['trades_history']}, "
                f"decisions={payload['files']['decisions_history']}, "
                f"eval={payload['files']['eval_history']}"
            )
            try:
                if h > 1:
                    stdscr.addstr(1, 0, file_line[: max(0, w - 1)])
            except Exception:
                pass

            y = 3
            if h < 6 or w < 20:
                try:
                    stdscr.addstr(3, 0, "Terminal too small. Resize and retry. (q to quit)"[: max(0, w - 1)])
                except Exception:
                    pass
                stdscr.refresh()
                ch = stdscr.getch()
                if ch in (ord("q"), ord("Q")):
                    break
                time.sleep(max(0.5, float(args.interval)))
                continue

            y = _draw_section(
                stdscr,
                y,
                "Next automatic runs:",
                payload["next_lines"],
                w,
            )
            if y >= h:
                stdscr.refresh()
                ch = stdscr.getch()
                if ch in (ord("q"), ord("Q")):
                    break
                time.sleep(max(0.5, float(args.interval)))
                continue

            pred_lines = _tabulate(["date", "city", "pred(F)", "spread(F)", "conf"], payload["pred_table"], w)
            y = _draw_section(stdscr, y, "Latest predictions (what the bot believes):", pred_lines, w)
            if y >= h:
                stdscr.refresh()
                ch = stdscr.getch()
                if ch in (ord("q"), ord("Q")):
                    break
                time.sleep(max(0.5, float(args.interval)))
                continue

            # Forecast comparison (all cities by default; optional focus with --city).
            city_focus = (args.city or "").strip().lower()
            preds_latest = _read_csv("Data/predictions_latest.csv")
            if city_focus:
                preds_latest = [r for r in preds_latest if (r.get("city") or "").strip().lower() == city_focus]

            def f(row: dict[str, str], key: str) -> str:
                v = (row.get(key) or "").strip()
                return f"{_safe_float(v, 0.0):.2f}" if v else ""

            # Choose columns based on available width.
            all_cols = [
                ("city", "city"),
                ("cons", "tmax_predicted"),
                ("om", "tmax_open_meteo"),
                ("vc", "tmax_visual_crossing"),
                ("tom", "tmax_tomorrow"),
                ("wapi", "tmax_weatherapi"),
                ("owm", "tmax_openweathermap"),
                ("pw", "tmax_pirateweather"),
                ("nws", "tmax_weather_gov"),
                ("lstm", "tmax_lstm"),
                ("spr", "spread_f"),
                ("conf", "confidence_score"),
            ]
            # Always include these.
            base = [all_cols[0], all_cols[1], all_cols[-2], all_cols[-1]]  # city, cons, spr, conf
            optional = [c for c in all_cols[2:-2]]  # sources

            def est_width(cols):
                # rough estimate: 2 spaces between cols, and ~6 chars per value
                return len(cols) * 8 + (len(cols) - 1) * 2

            chosen = list(base)
            for c in optional:
                if est_width(chosen + [c]) <= max(40, w - 2):
                    chosen.append(c)
            title = "Forecast comparison (all cities)" if not city_focus else f"Forecast comparison (city={city_focus})"

            comp_rows = []
            for r in preds_latest:
                comp_rows.append([f(r, key) for _, key in chosen])
            comp_lines = _tabulate([h for h, _ in chosen], comp_rows, w)
            y = _draw_section(stdscr, y, title + ":", comp_lines, w)
            if y >= h:
                stdscr.refresh()
                ch = stdscr.getch()
                if ch in (ord("q"), ord("Q")):
                    break
                time.sleep(max(0.5, float(args.interval)))
                continue

            open_lines = _tabulate(
                ["date", "city", "market", "qty", "price", "paid", "could_win", "time"],
                payload["open_table"],
                w,
            )
            y = _draw_section(stdscr, y, "Open bets (not settled yet):", open_lines, w)
            if y >= h:
                stdscr.refresh()
                ch = stdscr.getch()
                if ch in (ord("q"), ord("Q")):
                    break
                time.sleep(max(0.5, float(args.interval)))
                continue

            trade_lines = _tabulate(
                ["time", "date", "city", "market", "qty", "price", "paid", "could_win"],
                payload["trade_table"],
                w,
            )
            y = _draw_section(stdscr, y, f"Last {args.limit} placed bets (real orders):", trade_lines, w)
            if y >= h:
                stdscr.refresh()
                ch = stdscr.getch()
                if ch in (ord("q"), ord("Q")):
                    break
                time.sleep(max(0.5, float(args.interval)))
                continue

            dec_lines = _tabulate(["run_ts", "date", "city", "decision", "reason"], payload["decision_table"], w)
            y = _draw_section(stdscr, y, f"Recent bot actions (trade vs skip + why):", dec_lines, w)

            stdscr.refresh()

            ch = stdscr.getch()
            if ch in (ord("q"), ord("Q")):
                break

            time.sleep(max(0.5, float(args.interval)))

    curses.wrapper(loop)


if __name__ == "__main__":
    main()

