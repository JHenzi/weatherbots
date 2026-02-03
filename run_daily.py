import argparse
import csv
import datetime as dt
import json
import math
import os
import subprocess
import sys

from dotenv import load_dotenv

load_dotenv()


def _local_tz() -> dt.tzinfo:
    tzname = (os.getenv("TZ") or "").strip() or "America/New_York"
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(tzname)
    except Exception:
        return dt.datetime.now().astimezone().tzinfo or dt.timezone.utc


def _now_iso() -> str:
    # ISO with offset in local TZ, e.g. 2026-01-24T09:55:17-05:00
    return dt.datetime.now(tz=_local_tz()).isoformat()


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _append_with_metadata(
    *,
    src_csv: str,
    dst_csv: str,
    extra_fields: dict[str, str],
) -> None:
    os.makedirs(os.path.dirname(dst_csv) or ".", exist_ok=True)
    write_header = not os.path.exists(dst_csv)
    with open(src_csv, "r", newline="") as src_f:
        reader = csv.DictReader(src_f)
        src_fields = reader.fieldnames or []
        out_fields = list(src_fields) + [k for k in extra_fields.keys() if k not in src_fields]
        with open(dst_csv, "a", newline="") as dst_f:
            writer = csv.DictWriter(dst_f, fieldnames=out_fields)
            if write_header:
                writer.writeheader()
            for row in reader:
                if not row:
                    continue
                row.update(extra_fields)
                writer.writerow(row)


def _load_weights(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f) or {}


def _confidence_from_spread(spread_f: float) -> float:
    # VotingModel.md guardrail: <1.5F high confidence, >3.0F abort.
    if spread_f <= 1.5:
        return 1.0
    if spread_f >= 3.0:
        return 0.0
    # Linear interpolation between 1.5 and 3.0
    return float((3.0 - float(spread_f)) / (3.0 - 1.5))


def _skill_from_weights(weights_used: dict[str, float]) -> float:
    """
    Derive a per-city skill factor from the learned weights.

    Heuristic:
    - Interpret weights as a probability distribution over providers.
    - Use normalized Shannon entropy to capture how many competent sources
      contribute meaningfully to the ensemble:
        * High entropy (diversified, multiple good sources) → skill_conf ~ 1.0
        * Low entropy (one dominant source) → skill_conf ~ 0.0
    - This does NOT introduce extra recency bias: it relies solely on the
      existing MAE-based weights for the calibration window.
    """
    if not weights_used:
        # Neutral when we have no view on provider skill.
        return 0.5

    ws = [max(0.0, float(v)) for v in weights_used.values()]
    s = sum(ws)
    if s <= 0:
        return 0.5

    probs = [w / s for w in ws if w > 0.0]
    if len(probs) <= 1:
        # Single provider (or effectively single) → treat as moderate/unknown skill.
        return 0.5

    H = -sum(p * math.log(p) for p in probs)
    H_max = math.log(len(probs))
    if H_max <= 0:
        return 0.5

    entropy_norm = max(0.0, min(1.0, H / H_max))
    return float(entropy_norm)


def _postprocess_voting(
    predictions_csv: str,
    weights_json: str = "Data/weights.json",
    performance_csv: str | None = None,
    mae_window_days: int = 7,
    end_date: dt.date | None = None,
) -> None:
    """
    VotingModel consensus: MAE-weighted prediction and smart-spread confidence when
    performance_csv is provided and has data; otherwise fallback to weights.json + all-sources spread.
    """
    import statistics

    weights_all = _load_weights(weights_json)
    rolling_mae: dict[str, dict[str, float]] = {}
    if performance_csv and os.path.exists(performance_csv):
        from prediction_mae import get_rolling_mae_per_city_source
        rolling_mae = get_rolling_mae_per_city_source(
            performance_csv, window_days=mae_window_days, end_date=end_date
        )

    with open(predictions_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    extra_fields = ["spread_f", "confidence_score", "sources_used", "weights_used"]
    for ef in extra_fields:
        if ef not in fieldnames:
            fieldnames.append(ef)

    source_cols = {
        "open-meteo": "tmax_open_meteo",
        "visual-crossing": "tmax_visual_crossing",
        "tomorrow": "tmax_tomorrow",
        "weatherapi": "tmax_weatherapi",
        "google-weather": "tmax_google_weather",
        "openweathermap": "tmax_openweathermap",
        "pirateweather": "tmax_pirateweather",
        "weather.gov": "tmax_weather_gov",
        "lstm": "tmax_lstm",
    }
    for col in source_cols.values():
        if col not in fieldnames:
            fieldnames.append(col)

    for row in rows:
        city = (row.get("city") or "").strip()
        vals: dict[str, float] = {}
        for src, col in source_cols.items():
            v = row.get(col)
            if v is None or v == "":
                continue
            try:
                vals[src] = float(v)
            except ValueError:
                continue

        if not vals:
            continue

        mae_map = rolling_mae.get(city, {}) if rolling_mae else {}
        # Inverse-MAE weights: w_i = 1 / MAE_i^2 (floor MAE at 0.01)
        weights_used: dict[str, float] = {}
        for src in vals:
            mae = mae_map.get(src)
            if mae is not None:
                mae_safe = max(float(mae), 0.01)
                weights_used[src] = 1.0 / (mae_safe * mae_safe)
        if weights_used:
            s = sum(weights_used.values())
            weights_used = {k: v / s for k, v in weights_used.items()}
            consensus = sum(weights_used[src] * vals[src] for src in weights_used.keys())
        else:
            # Fallback: weights.json or equal
            w_city = (weights_all.get(city) or {}).get("weights") if isinstance(weights_all.get(city), dict) else None
            w_city = w_city or (weights_all.get(city) if isinstance(weights_all.get(city), dict) else None)
            if isinstance(w_city, dict):
                for src in vals:
                    if src in w_city:
                        try:
                            weights_used[src] = float(w_city[src])
                        except Exception:
                            pass
            if not weights_used:
                vote_sources = [s for s in vals if s != "lstm"] or list(vals)
                u = 1.0 / len(vote_sources)
                weights_used = {s: u for s in vote_sources}
            else:
                s = sum(weights_used.values())
                weights_used = {k: v / s for k, v in weights_used.items()} if s > 0 else weights_used
            consensus = sum(weights_used[src] * vals[src] for src in weights_used.keys())

        # Smart spread: only reliable sources (MAE within 1.5x of best). Exclude LSTM from spread.
        spread_vals = {k: v for k, v in vals.items() if k != "lstm"}
        if not spread_vals:
            spread_vals = dict(vals)
        sources_with_mae = [s for s in spread_vals if s in mae_map]
        if sources_with_mae:
            best_mae = min(mae_map[s] for s in sources_with_mae)
            reliable = [s for s in sources_with_mae if mae_map[s] <= 1.5 * best_mae]
            if len(reliable) >= 2:
                sigma = float(statistics.pstdev([vals[s] for s in reliable]))
            else:
                sigma = 0.0
            # Optional bonus: best source >20% better than runner-up
            mae_sorted = sorted(mae_map[s] for s in sources_with_mae)
            bonus = 0.1 if len(mae_sorted) >= 2 and mae_sorted[0] < 0.8 * mae_sorted[1] else 0.0
        else:
            sigma = float(statistics.pstdev(list(spread_vals.values()))) if len(spread_vals) > 1 else 0.0
            bonus = 0.0
        spread_conf_raw = _confidence_from_spread(sigma)
        spread_conf = min(0.9, max(0.0, float(spread_conf_raw)) + bonus)
        skill_conf = _skill_from_weights(weights_used)
        conf_final = spread_conf * (0.5 + 0.5 * skill_conf)

        row["tmax_predicted"] = f"{consensus}"
        row["spread_f"] = f"{sigma}"
        row["confidence_score"] = f"{conf_final}"
        row["sources_used"] = ",".join(sorted(weights_used.keys()))
        row["weights_used"] = ",".join([f"{k}:{weights_used[k]:.4f}" for k in sorted(weights_used.keys())])

    with open(predictions_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _env_first(*names: str) -> str | None:
    for n in names:
        v = os.getenv(n)
        if v is not None and str(v).strip():
            return str(v).strip()
    return None


def validate_config(*, trade_dt: dt.date, env: str) -> None:
    # Trade date sanity
    if trade_dt.year < 2020 or trade_dt.year > 2100:
        raise RuntimeError(f"trade_date looks wrong: {trade_dt.isoformat()}")

    # Kalshi creds are required even in dry-run (we still fetch events/markets).
    api_key_id = _env_first(
        "KALSHI_API_KEY_ID",
        "KALSHI_API_KEY",
        "KALSHI_KEY_ID",
        "API_KEY_ID",
        "API_KEY",
    )
    private_key_path = _env_first(
        "KALSHI_PRIVATE_KEY_PATH",
        "KALSHI_PRIVATE_KEY_FILE",
        "PRIVATE_KEY_PATH",
        "PRIVATE_KEY_FILE",
    )
    if not api_key_id:
        raise RuntimeError(
            "Missing Kalshi API key id. Set one of: KALSHI_API_KEY_ID / KALSHI_API_KEY / KALSHI_KEY_ID"
        )
    if not private_key_path:
        raise RuntimeError("Missing Kalshi private key path. Set KALSHI_PRIVATE_KEY_PATH to a PEM file.")
    if not os.path.exists(private_key_path):
        raise RuntimeError(f"Kalshi private key path does not exist: {private_key_path}")

    env_norm = (env or "").strip().lower()
    if env_norm not in ("demo", "prod", "production"):
        raise RuntimeError(f"--env must be demo|prod (got {env})")

    # Forecast provider keys are optional (Open-Meteo needs no key), but warn if none are configured
    # besides Open-Meteo (gives fewer votes / higher spread risk).
    vc = bool(_env_first("VISUAL_CROSSING_API_KEY"))
    tm = bool(_env_first("TOMORROW"))
    wa = bool(_env_first("WEATHERAPI"))
    owm = bool(_env_first("OPENWEATHERMAP_API_KEY"))
    pw = bool(_env_first("PIRATE_WEATHER_API_KEY", "PIRATE_WEATER_API_KEY"))
    if not (vc or tm or wa or owm or pw):
        print(
            "WARNING: No provider keys found for VISUAL_CROSSING_API_KEY/TOMORROW/WEATHERAPI/OPENWEATHERMAP_API_KEY/PIRATE_WEATHER_API_KEY. "
            "Forecast will rely on Open-Meteo only."
        )


def _parse_args():
    p = argparse.ArgumentParser(description="Monolithic daily weather→Kalshi trading run.")
    p.add_argument("--trade-date", type=str, default=None, help="YYYY-MM-DD (default: today)")
    p.add_argument("--env", type=str, default=os.getenv("KALSHI_ENV", "demo"))
    p.add_argument("--send-orders", action="store_true", help="Actually submit orders (default dry-run)")

    p.add_argument(
        "--prediction-mode",
        type=str,
        default="blend",
        choices=["lstm", "forecast", "blend"],
    )
    p.add_argument("--blend-forecast-weight", type=float, default=0.8)

    # Default behavior: fetch recent observed window (keep LSTM inputs current).
    # You can opt out with --skip-fetch if you want a faster run.
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--refresh-history",
        dest="skip_fetch",
        action="store_false",
        help="Fetch recent observed window first (default).",
    )
    g.add_argument(
        "--skip-fetch",
        dest="skip_fetch",
        action="store_true",
        help="Skip fetching/cleaning observed inputs (faster, but can make LSTM stale).",
    )
    p.set_defaults(skip_fetch=False)
    p.add_argument("--retrain-lstm", action="store_true", help="Train new LSTM models after refresh")
    p.add_argument("--retrain-days-window", type=int, default=730, help="Training window (days)")
    p.add_argument("--val-days", type=int, default=30, help="Validation window (days)")

    # 0 means "auto-size to city budget" (see kalshi_trader.py).
    p.add_argument("--count", type=int, default=0)
    p.add_argument("--side", type=str, default="yes", choices=["yes", "no"])
    # With orderbook-aware pricing, treat yes-price as a *ceiling* (default: permissive).
    p.add_argument("--yes-price", type=int, default=99)
    p.add_argument("--no-price", type=int, default=99)

    p.add_argument("--predictions-latest", type=str, default="Data/predictions_latest.csv")
    p.add_argument("--predictions-history", type=str, default="Data/predictions_history.csv")
    p.add_argument("--performance-csv", type=str, default="Data/source_performance.csv", help="Source performance for MAE-weighted consensus")
    p.add_argument("--mae-window-days", type=int, default=7, help="Rolling window (days) for MAE")
    p.add_argument("--trades-history", type=str, default="Data/trades_history.csv")
    p.add_argument("--decisions-history", type=str, default="Data/decisions_history.csv")
    p.add_argument("--eval-history", type=str, default="Data/eval_history.csv")
    p.add_argument("--min-confidence", type=float, default=0.75)
    p.add_argument("--max-spread", type=float, default=3.0)

    # Orderbook/EV/sizing controls (conservative defaults; forwarded to kalshi_trader.py)
    p.add_argument("--orderbook-depth", type=int, default=25)
    p.add_argument("--sigma-floor", type=float, default=2.0)
    p.add_argument("--sigma-mult", type=float, default=1.0)
    p.add_argument("--min-ev-cents", type=float, default=3.0)
    p.add_argument("--max-yes-spread-cents", type=float, default=6.0)
    p.add_argument("--min-ask-depth", type=int, default=25)
    p.add_argument("--max-dollars-per-city", type=float, default=50.0)
    p.add_argument("--max-dollars-total", type=float, default=150.0)
    p.add_argument("--max-contracts-per-order", type=int, default=25)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    trade_dt = dt.date.today() if not args.trade_date else dt.datetime.strptime(args.trade_date, "%Y-%m-%d").date()
    fetch_dt = trade_dt - dt.timedelta(days=1)

    validate_config(trade_dt=trade_dt, env=args.env)

    os.makedirs("Data", exist_ok=True)

    # Always overwrite the "latest" file so it's canonical for this run.
    if os.path.exists(args.predictions_latest):
        os.remove(args.predictions_latest)

    # 1) Prediction step: run for today and tomorrow so dashboard always has next trade date (e.g. after 7 PM ET).
    for run_dt in (trade_dt, trade_dt + dt.timedelta(days=1)):
        pred_cmd = [
            sys.executable,
            "daily_prediction.py",
            "--trade-date",
            run_dt.strftime("%Y-%m-%d"),
            "--prediction-mode",
            args.prediction_mode,
            "--blend-forecast-weight",
            str(args.blend_forecast_weight),
            "--predictions-csv",
            args.predictions_latest,
        ]
        if bool(getattr(args, "skip_fetch", False)):
            pred_cmd.append("--skip-fetch")
        _run(pred_cmd)

    # 1b) Voting-model postprocess: compute weighted consensus/spread/confidence and overwrite tmax_predicted.
    _postprocess_voting(
        args.predictions_latest,
        weights_json="Data/weights.json",
        performance_csv=args.performance_csv,
        mae_window_days=args.mae_window_days,
        end_date=trade_dt - dt.timedelta(days=1),
    )

    # 2) Append to predictions history with run metadata
    _append_with_metadata(
        src_csv=args.predictions_latest,
        dst_csv=args.predictions_history,
        extra_fields={
            "run_ts": _now_iso(),
            "env": args.env,
            "prediction_mode": args.prediction_mode,
            "blend_forecast_weight": str(args.blend_forecast_weight),
            "refresh_history": str(not bool(getattr(args, "skip_fetch", False))),
            "retrain_lstm": str(bool(args.retrain_lstm)),
        },
    )

    # 3) Optional daily model retraining
    if args.retrain_lstm:
        train_cmd = [
            sys.executable,
            "train_models.py",
            "--as-of-date",
            (trade_dt - dt.timedelta(days=1)).strftime("%Y-%m-%d"),
            "--days-window",
            str(args.retrain_days_window),
            "--val-days",
            str(args.val_days),
        ]
        _run(train_cmd)

    # 4) Trading step (dry-run by default)
    trade_cmd = [
        sys.executable,
        "kalshi_trader.py",
        "--env",
        args.env,
        "--trade-date",
        trade_dt.strftime("%Y-%m-%d"),
        "--predictions-csv",
        args.predictions_latest,
        "--count",
        str(args.count),
        "--side",
        args.side,
        "--yes-price",
        str(args.yes_price),
        "--no-price",
        str(args.no_price),
        "--trades-log",
        args.trades_history,
        "--decisions-log",
        args.decisions_history,
        "--eval-log",
        args.eval_history,
        "--min-confidence",
        str(args.min_confidence),
        "--max-spread",
        str(args.max_spread),
        "--orderbook-depth",
        str(args.orderbook_depth),
        "--sigma-floor",
        str(args.sigma_floor),
        "--sigma-mult",
        str(args.sigma_mult),
        "--min-ev-cents",
        str(args.min_ev_cents),
        "--max-yes-spread-cents",
        str(args.max_yes_spread_cents),
        "--min-ask-depth",
        str(args.min_ask_depth),
        "--max-dollars-per-city",
        str(args.max_dollars_per_city),
        "--max-dollars-total",
        str(args.max_dollars_total),
        "--max-contracts-per-order",
        str(args.max_contracts_per_order),
    ]
    if args.send_orders:
        trade_cmd.append("--send-orders")
    _run(trade_cmd)

