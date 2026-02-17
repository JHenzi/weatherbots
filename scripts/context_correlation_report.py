import argparse
import csv
import datetime as dt
import json
import os
from collections import defaultdict
from typing import Any


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        s = str(x).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _mean(vals: list[float]) -> float | None:
    if not vals:
        return None
    return sum(vals) / float(len(vals))


def _load_rewards(path: str) -> list[dict[str, str]]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def build_report(reward_rows: list[dict[str, str]], *, min_samples: int = 3) -> dict[str, Any]:
    mae_by_sky_city_mode: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    mae_by_condition_mode: dict[tuple[str, str], list[float]] = defaultdict(list)
    delta_vs_forecast: dict[tuple[str, str], list[float]] = defaultdict(list)
    applied_vs_forecast_by_city: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"applied": [], "forecast": []})

    for row in reward_rows:
        city = (row.get("city") or "").strip().lower()
        if not city:
            continue
        sky = (row.get("sky_label") or "").strip().lower() or "unknown"
        cond_token = (row.get("condition_token") or "").strip().lower() or "unknown"
        cond_label = (row.get("condition_label") or "").strip() or cond_token
        cond_key = f"{cond_token}::{cond_label}"

        err_forecast = _safe_float(row.get("error_forecast"))
        err_blend = _safe_float(row.get("error_blend"))
        err_lstm = _safe_float(row.get("error_lstm"))
        err_map = {
            "forecast": err_forecast,
            "blend": err_blend,
            "lstm": err_lstm,
        }

        for mode, err in err_map.items():
            if err is None:
                continue
            mae_by_sky_city_mode[(city, sky, mode)].append(float(err))
            mae_by_condition_mode[(cond_key, mode)].append(float(err))

        if err_forecast is not None:
            if err_blend is not None:
                delta_vs_forecast[(cond_key, "blend")].append(float(err_blend - err_forecast))
            if err_lstm is not None:
                delta_vs_forecast[(cond_key, "lstm")].append(float(err_lstm - err_forecast))

        applied_action = (row.get("applied_action") or "").strip().lower()
        applied_err = err_map.get(applied_action)
        if applied_err is not None and err_forecast is not None:
            applied_vs_forecast_by_city[city]["applied"].append(float(applied_err))
            applied_vs_forecast_by_city[city]["forecast"].append(float(err_forecast))

    out_sky = []
    for (city, sky, mode), vals in sorted(mae_by_sky_city_mode.items()):
        if len(vals) < min_samples:
            continue
        out_sky.append(
            {
                "city": city,
                "sky_label": sky,
                "mode": mode,
                "n": len(vals),
                "mae": round(float(_mean(vals) or 0.0), 4),
            }
        )

    out_cond = []
    for (cond_key, mode), vals in sorted(mae_by_condition_mode.items()):
        if len(vals) < min_samples:
            continue
        cond_token, cond_label = cond_key.split("::", 1)
        out_cond.append(
            {
                "condition_token": cond_token,
                "condition_label": cond_label,
                "mode": mode,
                "n": len(vals),
                "mae": round(float(_mean(vals) or 0.0), 4),
            }
        )

    out_delta = []
    for (cond_key, mode), vals in sorted(delta_vs_forecast.items()):
        if len(vals) < min_samples:
            continue
        cond_token, cond_label = cond_key.split("::", 1)
        out_delta.append(
            {
                "condition_token": cond_token,
                "condition_label": cond_label,
                "mode": mode,
                "n": len(vals),
                "delta_mae_vs_forecast": round(float(_mean(vals) or 0.0), 4),
            }
        )

    out_perf = []
    all_applied: list[float] = []
    all_forecast: list[float] = []
    for city, bucket in sorted(applied_vs_forecast_by_city.items()):
        a = bucket["applied"]
        f = bucket["forecast"]
        if not a or not f:
            continue
        all_applied.extend(a)
        all_forecast.extend(f)
        out_perf.append(
            {
                "city": city,
                "n": min(len(a), len(f)),
                "mae_applied": round(float(_mean(a) or 0.0), 4),
                "mae_forecast_baseline": round(float(_mean(f) or 0.0), 4),
                "improvement_vs_forecast": round(float((_mean(f) or 0.0) - (_mean(a) or 0.0)), 4),
            }
        )

    overall_perf = {
        "n": min(len(all_applied), len(all_forecast)),
        "mae_applied": round(float(_mean(all_applied) or 0.0), 4) if all_applied else None,
        "mae_forecast_baseline": round(float(_mean(all_forecast) or 0.0), 4) if all_forecast else None,
        "improvement_vs_forecast": (
            round(float((_mean(all_forecast) or 0.0) - (_mean(all_applied) or 0.0)), 4)
            if all_applied and all_forecast
            else None
        ),
    }

    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rows_scanned": len(reward_rows),
        "min_samples": int(min_samples),
        "mae_by_sky_city_mode": out_sky,
        "mae_by_condition_mode": out_cond,
        "delta_vs_forecast_by_condition": out_delta,
        "bandit_performance_by_city": out_perf,
        "bandit_performance_overall": overall_perf,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build context-correlation report from bandit reward history.")
    p.add_argument("--bandit-rewards", type=str, default="Data/bandit_rewards_history.csv")
    p.add_argument("--min-samples", type=int, default=3)
    p.add_argument("--out-json", type=str, default="")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    rows = _load_rewards(args.bandit_rewards)
    report = build_report(rows, min_samples=max(1, int(args.min_samples)))
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            f.write(payload + "\n")
    print(payload)
