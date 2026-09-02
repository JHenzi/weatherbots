import argparse
import csv
import datetime as dt
import json
import math
import os
import random
import statistics
import time
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

load_dotenv()

from bandit.context import vote_provider_conditions
from bandit.modes import choose_mode_prediction, compute_candidate_mode_predictions
from bandit.policy import build_feature_vector, load_policy_state, save_policy_state

try:
    import db  # type: ignore  # local Postgres helpers
except Exception:  # pragma: no cover - defensive fallback when db.py missing
    db = None  # type: ignore[assignment]

try:
    import requests_cache  # type: ignore
except ModuleNotFoundError:
    requests_cache = None


# Providers included in intraday snapshots (09/15/21 + final 22:00).
# This is the "full" forecast set used for trading decisions.
SOURCES_ORDER = [
    "google-weather",
    "open-meteo",
    "openweathermap",
    "pirateweather",
    "tomorrow",
    "visual-crossing",
    "weather.gov",
    "weatherapi",
]

CITIES = ["ny", "il", "tx", "fl"]
LATLON = {
    "ny": (40.79736, -73.97785),
    "il": (41.78701, -87.77166),
    "tx": (30.14440, -97.66876),
    "fl": (25.77380, -80.19360),
}

PROVIDER_COLS: dict[str, str] = {
    "open-meteo": "tmax_open_meteo",
    "visual-crossing": "tmax_visual_crossing",
    "tomorrow": "tmax_tomorrow",
    "weatherapi": "tmax_weatherapi",
    "google-weather": "tmax_google_weather",
    "openweathermap": "tmax_openweathermap",
    "pirateweather": "tmax_pirateweather",
    "weather.gov": "tmax_weather_gov",
}


def _local_tz() -> dt.tzinfo:
    tzname = (os.getenv("TZ") or "").strip() or "America/New_York"
    try:
        return ZoneInfo(tzname)
    except Exception:
        return dt.datetime.now().astimezone().tzinfo or dt.timezone.utc


def _now_iso_local() -> str:
    return dt.datetime.now(tz=_local_tz()).isoformat()


def _safe_float(x) -> float | None:
    if x is None:
        return None
    try:
        s = str(x).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _safe_int(x, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return int(default)


def _safe_json_dumps(x) -> str:
    try:
        return json.dumps(x, sort_keys=True)
    except Exception:
        return "{}"


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return float(default)
    try:
        return float(str(raw).strip())
    except Exception:
        return float(default)


def _parse_provider_result(raw) -> tuple[float | None, dict[str, object]]:
    """
    Provider functions return either:
      - float | None (legacy)
      - (float | None, dict context)
    """
    if isinstance(raw, tuple) and len(raw) == 2:
        val = _safe_float(raw[0])
        ctx = raw[1] if isinstance(raw[1], dict) else {}
        return val, dict(ctx)
    return _safe_float(raw), {}


def _bandit_mode_default() -> str:
    mode = str(os.getenv("WT_BANDIT_MODE", "off")).strip().lower()
    if mode not in ("off", "shadow", "canary", "live"):
        return "off"
    return mode


def _bandit_seed_for(city: str, trade_date: str, run_ts: str) -> int:
    seed_s = f"{city}|{trade_date}|{run_ts}"
    return abs(hash(seed_s)) % (2**31 - 1)


def _parse_iso_dt(s: str) -> dt.datetime | None:
    ss = (s or "").strip()
    if not ss:
        return None
    try:
        return dt.datetime.fromisoformat(ss.replace("Z", "+00:00"))
    except Exception:
        return None


def _try_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


_CONDITION_BUCKET: dict[str, str] = {
    "clear": "clear",
    "partly_cloudy": "mixed",
    "cloudy_overcast": "mixed",
    "wind": "mixed",
    "other": "mixed",
    "rain": "precip",
    "storm": "precip",
    "fog": "precip",
    "snow": "snow",
}


def _condition_confidence_factor(
    condition_token: str,
    vote_entropy: float,
    mae_by_condition: dict[str, float] | None,
    base_mae: float | None,
) -> float:
    """
    Learned condition-aware confidence multiplier.

    Computes the ratio of skill for the current condition vs the city's average skill using
    the same _mae_to_skill curve. Clear-sky days with historically low MAE get a boost;
    stormy/snow days with high MAE get a penalty. vote_entropy adds an independent penalty
    when providers disagree on what the conditions even are.

    Returns a multiplier in [0.70, 1.15]. Falls back to 1.0 if no per-condition data.
    """
    if not mae_by_condition or base_mae is None or base_mae <= 0:
        return 1.0
    bucket = _CONDITION_BUCKET.get(condition_token, "mixed")
    cond_mae = mae_by_condition.get(bucket)
    if cond_mae is None:
        return 1.0
    base_skill = _mae_to_skill(base_mae)
    if base_skill <= 0:
        return 1.0
    cond_skill = _mae_to_skill(cond_mae)
    ratio = cond_skill / base_skill
    # vote_entropy in [0,1]: high entropy = providers disagree on conditions → up to 10% penalty
    entropy_penalty = 0.10 * float(vote_entropy)
    return max(0.70, min(1.15, ratio - entropy_penalty))


def _confidence_from_spread(spread_f: float) -> float:
    # Match run_daily.py behavior.
    if spread_f <= 1.5:
        return 1.0
    if spread_f >= 3.0:
        return 0.0
    return float((3.0 - float(spread_f)) / (3.0 - 1.5))


def _entropy_skill_from_weights(weights_used: dict[str, float]) -> float:
    """
    Distribution-shape confidence from provider weights.
    """
    if not weights_used:
        return 0.5

    ws = [max(0.0, float(v)) for v in weights_used.values()]
    s = sum(ws)
    if s <= 0:
        return 0.5

    probs = [w / s for w in ws if w > 0.0]
    if len(probs) <= 1:
        return 0.5

    H = -sum(p * math.log(p) for p in probs)
    H_max = math.log(len(probs))
    if H_max <= 0:
        return 0.5

    return float(max(0.0, min(1.0, H / H_max)))


def _mae_to_skill(mae_f: float) -> float:
    """
    Map expected MAE to [0,1] quality:
    - <=0.8F -> 1.0 (high skill)
    - >=4.0F -> 0.0 (low skill)
    """
    mae = float(mae_f)
    if mae <= 0.8:
        return 1.0
    if mae >= 4.0:
        return 0.0
    return float((4.0 - mae) / (4.0 - 0.8))


def _quality_skill_from_weights(
    weights_used: dict[str, float],
    mae_map: dict[str, float] | None,
    *,
    neutral: float = 0.7,
) -> float:
    """
    MAE-aware confidence that downweights poor sources proportional to their weight.
    """
    if not weights_used:
        return float(max(0.0, min(1.0, neutral)))
    if not mae_map:
        return float(max(0.0, min(1.0, neutral)))

    known: list[tuple[float, float]] = []
    for src, w in weights_used.items():
        if src not in mae_map:
            continue
        ww = max(0.0, float(w))
        if ww <= 0.0:
            continue
        mae_safe = max(0.01, float(mae_map[src]))
        known.append((ww, mae_safe))

    coverage = sum(w for w, _ in known)
    if coverage <= 1e-9:
        return float(max(0.0, min(1.0, neutral)))

    weighted_quality = sum((w / coverage) * _mae_to_skill(mae) for w, mae in known)
    blended = coverage * weighted_quality + (1.0 - coverage) * neutral
    return float(max(0.0, min(1.0, blended)))


def _skill_from_weights(
    weights_used: dict[str, float],
    mae_map: dict[str, float] | None = None,
    *,
    entropy_blend: float = 0.75,
) -> float:
    """
    Blend distribution shape with MAE-aware quality.
    This reduces the effect of low-weight poor sources without over-trusting single-source setups.
    """
    ent = _entropy_skill_from_weights(weights_used)
    qual = _quality_skill_from_weights(weights_used, mae_map, neutral=0.7)
    a = max(0.0, min(1.0, float(entropy_blend)))
    return float((a * ent) + ((1.0 - a) * qual))


def _load_weights(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _weights_for_city(weights_all: dict, city: str) -> dict[str, float]:
    """
    Supported shapes:
      - {"ny": {"weights": {"open-meteo": 0.2, ...}, ...}, ...}
      - {"ny": {"open-meteo": 0.2, ...}, ...}
    """
    node = weights_all.get(city) if isinstance(weights_all, dict) else None
    if isinstance(node, dict) and isinstance(node.get("weights"), dict):
        node = node.get("weights")
    if not isinstance(node, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in node.items():
        # LSTM is a model output, not a forecast provider.
        if str(k) == "lstm":
            continue
        if str(k) not in SOURCES_ORDER:
            continue
        fv = _safe_float(v)
        if fv is None:
            continue
        out[str(k)] = float(fv)
    s = sum(max(0.0, v) for v in out.values())
    if s <= 0:
        return {}
    return {k: float(v) / s for k, v in out.items()}


def _append_intraday_row(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path)
    fieldnames = [
        "timestamp",
        "city",
        "trade_date",
        "mean_forecast",
        "current_sigma",
        "tmax_open_meteo",
        "tmax_visual_crossing",
        "tmax_tomorrow",
        "tmax_weatherapi",
        "tmax_google_weather",
        "tmax_openweathermap",
        "tmax_pirateweather",
        "tmax_weather_gov",
        "sources_used",
        "weights_used",
        "outliers_rejected",
    ]
    if not write_header:
        try:
            with open(path, "r", newline="") as f:
                r = csv.reader(f)
                existing = next(r, [])
            existing = [str(x).strip() for x in existing if str(x).strip() != ""]
            if existing != fieldnames:
                _migrate_intraday_forecasts_schema(path, fieldnames)
        except Exception:
            # Best-effort: if migration fails, still append (worst case: row preserved but columns shifted).
            pass
    payload = {k: row.get(k, "") for k in fieldnames}
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(payload)
    if db is not None:
        db.insert_intraday_snapshot_row(payload)  # type: ignore[attr-defined]


def _append_context_feature_row(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path)
    fieldnames = [
        "run_ts",
        "decision_role",
        "bandit_mode",
        "city",
        "trade_date",
        "provider_count",
        "spread_f",
        "condition_token",
        "condition_label",
        "sky_label",
        "mean_cloud_cover",
        "vote_entropy",
        "raw_provider_labels_json",
        "token_weights_json",
    ]
    payload = {k: row.get(k, "") for k in fieldnames}
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(payload)
    if db is not None:
        db.insert_context_feature_row(payload)  # type: ignore[attr-defined]


def _append_bandit_decision_row(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path)
    fieldnames = [
        "run_ts",
        "decision_role",
        "bandit_mode",
        "city",
        "trade_date",
        "selected_action",
        "applied_action",
        "action_reason",
        "guardrail_reason",
        "mode_forecast_pred",
        "mode_blend_pred",
        "mode_lstm_pred",
        "feature_vector_json",
        "feature_map_json",
        "policy_scores_json",
        "condition_token",
        "condition_label",
        "sky_label",
        "mean_cloud_cover",
        "vote_entropy",
        "provider_count",
        "spread_f",
        "raw_provider_labels_json",
    ]
    payload = {k: row.get(k, "") for k in fieldnames}
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(payload)
    if db is not None:
        db.insert_bandit_decision_row(payload)  # type: ignore[attr-defined]


def _load_recent_intraday_history(
    path: str,
    *,
    city: str,
    trade_date: str,
    max_rows: int = 4,
) -> list[dict]:
    """
    Load up to max_rows most recent intraday snapshots for (city, trade_date).
    Used for lead-time tracking / per-provider volatility.
    """
    if db is not None and getattr(db, "_pg_read_enabled", lambda: False)():
        try:
            return db.get_recent_intraday_snapshots(
                city_code=city, trade_date=trade_date, limit=max_rows
            )
        except Exception as e:
            print(f"Postgres read failed ({e}), falling back to CSV for intraday history")
    if not path or not os.path.exists(path):
        return []
    rows: list[tuple[dt.datetime, dict]] = []
    try:
        with open(path, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                if not row:
                    continue
                if (row.get("city") or "").strip() != (city or "").strip():
                    continue
                if (row.get("trade_date") or "").strip() != (trade_date or "").strip():
                    continue
                ts = _parse_iso_dt(row.get("timestamp") or "")
                if ts is None:
                    continue
                rows.append((ts, row))
    except Exception:
        return []
    if not rows:
        return []
    rows.sort(key=lambda t: t[0])
    rows = rows[-max_rows:]
    return [row for _, row in rows]


def _compute_volatility_info(
    history_rows: list[dict],
    current_vals: dict[str, float],
) -> dict[str, dict[str, float]]:
    """
    For each provider, compute:
      - volatility: mean |delta| over the last up-to-3 deltas
      - last_delta: most recent delta
      - mean_level: mean forecast level across history+current

    Deltas are consecutive differences in the provider's forecast for the same
    (city, trade_date) across pulses.
    """
    info: dict[str, dict[str, float]] = {}
    for src, col in PROVIDER_COLS.items():
        cur = current_vals.get(src)
        if cur is None:
            continue
        series: list[float] = []
        for row in history_rows:
            v = _safe_float(row.get(col))
            if v is None:
                continue
            series.append(float(v))
        series.append(float(cur))
        if len(series) < 2:
            volatility = 0.0
            last_delta = 0.0
        else:
            deltas = [series[i + 1] - series[i] for i in range(len(series) - 1)]
            recent = deltas[-3:]
            volatility = sum(abs(d) for d in recent) / float(len(recent))
            last_delta = deltas[-1]
        mean_level = sum(series) / float(len(series)) if series else 0.0
        info[src] = {
            "volatility": float(volatility),
            "last_delta": float(last_delta),
            "mean_level": float(mean_level),
        }
    return info


def _apply_volatility_weighting(
    available: dict[str, float],
    base_weights: dict[str, float],
    history_rows: list[dict],
) -> tuple[dict[str, float], float]:
    """
    Consensus 2.0:
      - Start from base_weights (learned weights if available, else uniform).
      - For each provider, compute volatility over the last few pulses.
      - If volatility > 2°F OR >10% of its mean level, apply a 50% penalty.
      - Agreement bonus: if two volatile providers share a similar last_delta
        (same sign, similar magnitude), do NOT penalize them (treat as trend).

    Returns:
      - new_weights: renormalized dynamic weights for available providers.
      - stability_score: 0..1 summarizing how many high-weight providers are
        stable or in coherent trend (used for conviction_score).
    """
    if not available:
        return ({}, 0.5)

    vol_info = _compute_volatility_info(history_rows, available)

    # If we don't have learned weights, start from uniform over available.
    bw: dict[str, float]
    if base_weights:
        bw = {k: float(v) for k, v in base_weights.items() if k in available}
        total = sum(max(0.0, v) for v in bw.values())
        if total <= 0:
            n = len(available)
            bw = {k: 1.0 / float(n) for k in available.keys()}
        else:
            bw = {k: float(v) / float(total) for k, v in bw.items()}
    else:
        n = len(available)
        bw = {k: 1.0 / float(n) for k in available.keys()}

    # First pass: identify which providers are "volatile".
    THRESH_DEG = 2.0
    is_volatile: dict[str, bool] = {}
    for src in available.keys():
        meta = vol_info.get(src) or {}
        vol = float(meta.get("volatility", 0.0))
        mean_level = abs(float(meta.get("mean_level", 0.0)))
        high_deg = vol > THRESH_DEG
        high_pct = False
        if mean_level > 0:
            high_pct = vol > (0.10 * mean_level)
        is_volatile[src] = bool(high_deg or high_pct)

    # Second pass: agreement bonus for coherent high skew and penalties for outliers.
    factors: dict[str, float] = {k: 1.0 for k in available.keys()}
    for src in available.keys():
        if not is_volatile.get(src, False):
            continue
        meta_i = vol_info.get(src) or {}
        delta_i = float(meta_i.get("last_delta", 0.0))
        if delta_i == 0.0:
            # Flat but flagged volatile via threshold; still treat as noise.
            factors[src] = 0.5
            continue
        sign_i = 1.0 if delta_i > 0 else -1.0
        found_partner = False
        for other in available.keys():
            if other == src or not is_volatile.get(other, False):
                continue
            meta_j = vol_info.get(other) or {}
            delta_j = float(meta_j.get("last_delta", 0.0))
            if delta_j == 0.0:
                continue
            sign_j = 1.0 if delta_j > 0 else -1.0
            if sign_j != sign_i:
                continue
            if abs(delta_j - delta_i) <= 1.0:
                found_partner = True
                break
        if not found_partner:
            # Volatile and not supported by another similarly-moving provider.
            factors[src] = 0.5

    # Third pass: staleness penalty. If most providers are updating in a common
    # direction but one stays flat, treat that as "stale" and downweight it.
    deltas = [
        float((vol_info.get(src) or {}).get("last_delta", 0.0)) for src in available.keys()
    ]
    # Ignore tiny noise when inferring the pack's movement.
    significant = [d for d in deltas if abs(d) >= 0.1]
    if len(significant) >= 2:
        try:
            import statistics  # local import to avoid top-level dependency issues
        except Exception:
            statistics = None  # type: ignore[assignment]

        if statistics is not None:
            med = statistics.median(significant)
            trend_mag = abs(med)
            if trend_mag >= 0.5:
                # Pack is moving meaningfully; penalize sources that are near-flat.
                for src in available.keys():
                    meta = vol_info.get(src) or {}
                    d = float(meta.get("last_delta", 0.0))
                    if abs(d) < 0.1:
                        # Everyone else is updating in roughly the same direction,
                        # this one is effectively unchanged → likely stale.
                        factors[src] = min(float(factors.get(src, 1.0)), 0.5)

    # Combine base weights with volatility factors and renormalize.
    pre: dict[str, float] = {}
    for src in available.keys():
        w0 = float(bw.get(src, 0.0))
        if w0 <= 0.0:
            continue
        f = float(factors.get(src, 1.0))
        pre[src] = w0 * f
    s = sum(pre.values())
    if s <= 0:
        new_weights = dict(bw)
    else:
        new_weights = {k: float(v) / float(s) for k, v in pre.items()}

    # Stability score: share of weight on non-penalized providers (or coherent trends).
    if not new_weights:
        stability = 0.5
    else:
        stable_mass = sum(
            new_weights[src]
            for src in new_weights.keys()
            if float(factors.get(src, 1.0)) >= 1.0
        )
        stability = max(0.0, min(1.0, float(stable_mass)))
    return (new_weights, float(stability))


def _migrate_intraday_forecasts_schema(path: str, new_fieldnames: list[str]) -> None:
    """
    Data/intraday_forecasts.csv may have an older, smaller header. When we add new provider
    columns, appends can create shifted rows. This migration rewrites the file using the
    canonical header.
    """
    old_fieldnames = [
        "timestamp",
        "city",
        "trade_date",
        "mean_forecast",
        "current_sigma",
        "tmax_open_meteo",
        "tmax_visual_crossing",
        "tmax_tomorrow",
        "tmax_weatherapi",
        "sources_used",
        "weights_used",
    ]

    with open(path, "r", newline="") as f:
        r = csv.reader(f)
        _ = next(r, None)  # existing header (may be stale)
        rows = list(r)

    out_rows: list[dict[str, str]] = []
    for row in rows:
        if not row:
            continue
        if len(row) == len(old_fieldnames):
            d = dict(zip(old_fieldnames, row))
        elif len(row) == len(new_fieldnames):
            d = dict(zip(new_fieldnames, row))
        else:
            d = dict(zip(old_fieldnames, row[: len(old_fieldnames)]))
        out_rows.append(d)

    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=new_fieldnames)
        w.writeheader()
        for d in out_rows:
            w.writerow({k: d.get(k, "") for k in new_fieldnames})
    os.replace(tmp, path)


def _write_predictions_latest(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "date",
        "city",
        "tmax_predicted",
        "tmax_lstm",
        "tmax_forecast",
        "forecast_sources",
        "tmax_open_meteo",
        "tmax_visual_crossing",
        "tmax_tomorrow",
        "tmax_weatherapi",
        "tmax_google_weather",
        "tmax_openweathermap",
        "tmax_pirateweather",
        "tmax_weather_gov",
        "spread_f",
        "confidence_score",
        "sources_used",
        "weights_used",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            # Ensure conviction_score doesn't leak into the CSV yet if it's not in the header.
            # (We will add it to the history schema properly in a later step if needed).
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _append_predictions_history(path: str, latest_rows: list[dict], *, extra_fields: dict[str, str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path)
    
    # Use the same canonical header as run_daily.py to avoid misalignment.
    fieldnames = [
        "date",
        "city",
        "tmax_predicted",
        "tmax_lstm",
        "tmax_forecast",
        "forecast_sources",
        "tmax_open_meteo",
        "tmax_visual_crossing",
        "tmax_tomorrow",
        "tmax_weatherapi",
        "tmax_google_weather",
        "tmax_openweathermap",
        "tmax_pirateweather",
        "tmax_weather_gov",
        "spread_f",
        "confidence_score",
        "sources_used",
        "weights_used",
        "run_ts",
        "env",
        "prediction_mode",
        "blend_forecast_weight",
        "refresh_history",
        "retrain_lstm",
    ]
    
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for r in latest_rows:
            rr = {k: r.get(k, "") for k in fieldnames if k in r}
            rr.update(extra_fields)
            # Fill missing keys with empty strings to avoid DictWriter errors
            row_to_write = {k: rr.get(k, "") for k in fieldnames}
            w.writerow(row_to_write)
            if db is not None:
                db.insert_prediction_row(row_to_write)  # type: ignore[attr-defined]


def _openmeteo_code_to_text(code: int | None) -> str:
    if code is None:
        return ""
    mapping = {
        0: "clear",
        1: "mainly clear",
        2: "partly cloudy",
        3: "overcast",
        45: "fog",
        48: "depositing rime fog",
        51: "drizzle",
        53: "drizzle",
        55: "dense drizzle",
        56: "freezing drizzle",
        57: "freezing drizzle",
        61: "rain",
        63: "rain",
        65: "heavy rain",
        66: "freezing rain",
        67: "freezing rain",
        71: "snow",
        73: "snow",
        75: "heavy snow",
        77: "snow grains",
        80: "rain showers",
        81: "rain showers",
        82: "violent rain showers",
        85: "snow showers",
        86: "snow showers",
        95: "thunderstorm",
        96: "thunderstorm with hail",
        99: "thunderstorm with hail",
    }
    return mapping.get(int(code), f"weather code {int(code)}")


def forecast_tmax_open_meteo(*, city: str, trade_dt: dt.date) -> tuple[float | None, dict[str, object]]:
    lat, lon = LATLON[city]
    session = (
        requests_cache.CachedSession("Data/open_meteo_forecast_cache", expire_after=3600)
        if requests_cache is not None
        else requests.Session()
    )
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,weather_code",
        "temperature_unit": "fahrenheit",
        "timezone": "UTC",
        "start_date": trade_dt.strftime("%Y-%m-%d"),
        "end_date": trade_dt.strftime("%Y-%m-%d"),
    }
    r = session.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return (None, {})
    js = r.json()
    daily = js.get("daily") or {}
    temps = daily.get("temperature_2m_max") or []
    codes = daily.get("weather_code") or []
    if not temps:
        return (None, {})
    code = _safe_int(codes[0], default=-1) if codes else None
    ctx = {
        "condition_text": _openmeteo_code_to_text(code if code != -1 else None),
        "condition_icon": "",
        "cloud_cover": None,
    }
    return (_safe_float(temps[0]), ctx)


def forecast_tmax_visual_crossing(*, city: str, trade_dt: dt.date) -> tuple[float | None, dict[str, object]]:
    api_key = os.getenv("VISUAL_CROSSING_API_KEY")
    if not api_key:
        return (None, {})
    lat, lon = LATLON[city]
    start = trade_dt.strftime("%Y-%m-%d")
    end = trade_dt.strftime("%Y-%m-%d")
    url = (
        "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
        + f"{lat}%2C{lon}/{start}/{end}"
        + "?unitGroup=us&include=days&key="
        + api_key
        + "&contentType=json"
    )
    session = (
        requests_cache.CachedSession("Data/visualcrossing_cache", expire_after=3600)
        if requests_cache is not None
        else requests.Session()
    )
    r = session.get(url, timeout=30)
    if r.status_code != 200:
        return (None, {})
    js = r.json()
    days = js.get("days") or []
    if not days:
        return (None, {})
    day0 = days[0] or {}
    ctx = {
        "condition_text": str(day0.get("conditions") or "").strip(),
        "condition_icon": str(day0.get("icon") or "").strip(),
        "cloud_cover": _safe_float(day0.get("cloudcover")),
    }
    return (_safe_float(day0.get("tempmax")), ctx)


def forecast_tmax_tomorrow(*, city: str, trade_dt: dt.date, _state: dict) -> tuple[float | None, dict[str, object]]:
    api_key = os.getenv("TOMORROW")
    if not api_key:
        return (None, {})
    lat, lon = LATLON[city]
    url = "https://api.tomorrow.io/v4/weather/forecast"
    params = {
        "location": f"{lat},{lon}",
        "timesteps": "1d",
        "units": "imperial",
        "apikey": api_key,
    }
    session = (
        requests_cache.CachedSession("Data/tomorrow_cache", expire_after=3600)
        if requests_cache is not None
        else requests.Session()
    )

    last_ts = _state.get("last_req_ts")
    if last_ts is not None:
        elapsed = time.time() - float(last_ts)
        if elapsed < 0.40:
            time.sleep(0.40 - elapsed)

    r = session.get(url, params=params, timeout=30)
    if not getattr(r, "from_cache", False):
        _state["last_req_ts"] = time.time()
    if r.status_code != 200:
        return (None, {})
    js = r.json()
    tl = (js.get("timelines") or {}).get("daily") or []
    if not tl:
        return (None, {})
    target = trade_dt.strftime("%Y-%m-%d")
    for item in tl:
        t = str(item.get("time") or "")
        if t.startswith(target):
            vals = item.get("values") or {}
            return (
                _safe_float(vals.get("temperatureMax")),
                {
                    "condition_text": str(vals.get("weatherCodeDay") or vals.get("weatherCodeFullDay") or "").strip(),
                    "condition_icon": "",
                    "cloud_cover": _safe_float(vals.get("cloudCoverAvg") or vals.get("cloudCover")),
                },
            )
    t0 = str((tl[0] or {}).get("time") or "")
    if t0.startswith(target):
        vals = (tl[0] or {}).get("values") or {}
        return (
            _safe_float(vals.get("temperatureMax")),
            {
                "condition_text": str(vals.get("weatherCodeDay") or vals.get("weatherCodeFullDay") or "").strip(),
                "condition_icon": "",
                "cloud_cover": _safe_float(vals.get("cloudCoverAvg") or vals.get("cloudCover")),
            },
        )
    return (None, {})


def forecast_tmax_weatherapi(*, city: str, trade_dt: dt.date) -> tuple[float | None, dict[str, object]]:
    api_key = os.getenv("WEATHERAPI")
    if not api_key:
        return (None, {})
    lat, lon = LATLON[city]
    url = "https://api.weatherapi.com/v1/forecast.json"
    params = {
        "key": api_key,
        "q": f"{lat},{lon}",
        "days": 1,
        "dt": trade_dt.strftime("%Y-%m-%d"),
        "alerts": "no",
        "aqi": "no",
    }
    session = (
        requests_cache.CachedSession("Data/weatherapi_cache", expire_after=3600)
        if requests_cache is not None
        else requests.Session()
    )
    r = session.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return (None, {})
    js = r.json()
    fc = (js.get("forecast") or {}).get("forecastday") or []
    if not fc:
        return (None, {})
    day = (fc[0] or {}).get("day") or {}
    cond = day.get("condition") or {}
    return (
        _safe_float(day.get("maxtemp_f")),
        {
            "condition_text": str(cond.get("text") or "").strip(),
            "condition_icon": str(cond.get("icon") or "").strip(),
            "cloud_cover": None,
        },
    )


def forecast_tmax_google_weather(*, city: str, trade_dt: dt.date) -> tuple[float | None, dict[str, object]]:
    """
    Google Weather hourly forecast -> take max hourly temp on trade_dt (local to the location).

    Env var:
      - GOOGLE (preferred)
      - GOOGLE_WEATHER_API_KEY (fallback)
    """
    api_key = os.getenv("GOOGLE") or os.getenv("GOOGLE_WEATHER_API_KEY")
    if not api_key:
        return (None, {})

    lat, lon = LATLON[city]
    url = "https://weather.googleapis.com/v1/forecast/hours:lookup"
    params = {
        "key": api_key,
        "location.latitude": lat,
        "location.longitude": lon,
        "hours": 240,
    }
    session = (
        requests_cache.CachedSession("Data/google_weather_cache", expire_after=3600)
        if requests_cache is not None
        else requests.Session()
    )
    r = session.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return (None, {})
    js = r.json() or {}

    tz_id = ((js.get("timeZone") or {}) if isinstance(js.get("timeZone"), dict) else {}).get("id") or "UTC"
    try:
        tz = ZoneInfo(str(tz_id))
    except Exception:
        tz = ZoneInfo("UTC")

    hours = js.get("forecastHours") or []
    best = None
    best_cond: dict[str, object] = {}
    for h in hours:
        interval = (h or {}).get("interval") or {}
        st = interval.get("startTime") or interval.get("endTime")
        if not st:
            continue
        try:
            t = dt.datetime.fromisoformat(str(st).replace("Z", "+00:00"))
        except Exception:
            continue
        try:
            local_day = t.astimezone(tz).date()
        except Exception:
            local_day = t.date()
        if local_day != trade_dt:
            continue

        temp = (h or {}).get("temperature") or {}
        deg = temp.get("degrees")
        unit = str(temp.get("unit") or "").upper()
        fv = _safe_float(deg)
        if fv is None:
            continue
        if unit == "CELSIUS":
            fv = (fv * 9.0 / 5.0) + 32.0
        if best is None or fv >= best:
            best = fv
            wc = (h or {}).get("weatherCondition") or {}
            best_cond = {
                "condition_text": str(wc.get("description") or wc.get("type") or "").strip(),
                "condition_icon": str(wc.get("iconBaseUri") or "").strip(),
                "cloud_cover": _safe_float((h or {}).get("cloudCover")),
            }
    return (best, best_cond)


def forecast_tmax_openweathermap(*, city: str, trade_dt: dt.date) -> tuple[float | None, dict[str, object]]:
    if str(os.getenv("DISABLE_OPENWEATHERMAP", "")).strip().lower() in ("1", "true", "yes", "y"):
        return (None, {})
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    if not api_key:
        return (None, {})
    lat, lon = LATLON[city]
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "imperial"}
    session = (
        requests_cache.CachedSession("Data/openweathermap_cache", expire_after=3600)
        if requests_cache is not None
        else requests.Session()
    )
    r = session.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return (None, {})
    js = r.json()
    tz_offset = int(((js.get("city") or {}).get("timezone")) or 0)
    items = js.get("list") or []
    if not items:
        return (None, {})
    max_t = None
    best_ctx: dict[str, object] = {}
    for it in items:
        ts = it.get("dt")
        if ts is None:
            continue
        try:
            local_day = (dt.datetime.utcfromtimestamp(int(ts)) + dt.timedelta(seconds=tz_offset)).date()
        except Exception:
            continue
        if local_day != trade_dt:
            continue
        main = it.get("main") or {}
        v = _safe_float(main.get("temp_max", main.get("temp")))
        if v is None:
            continue
        if max_t is None or v >= max_t:
            max_t = v
            weather = (it.get("weather") or [{}])[0] or {}
            best_ctx = {
                "condition_text": str(weather.get("description") or weather.get("main") or "").strip(),
                "condition_icon": str(weather.get("icon") or "").strip(),
                "cloud_cover": _safe_float((it.get("clouds") or {}).get("all")),
            }
    return (max_t, best_ctx)


def forecast_tmax_pirateweather(*, city: str, trade_dt: dt.date) -> tuple[float | None, dict[str, object]]:
    api_key = os.getenv("PIRATE_WEATHER_API_KEY") or os.getenv("PIRATE_WEATER_API_KEY")
    if not api_key:
        return (None, {})
    lat, lon = LATLON[city]
    url = f"https://api.pirateweather.net/forecast/{api_key}/{lat},{lon}"
    params = {"units": "us", "exclude": "currently,minutely,hourly,alerts"}
    session = (
        requests_cache.CachedSession("Data/pirateweather_cache", expire_after=3600)
        if requests_cache is not None
        else requests.Session()
    )
    r = session.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return (None, {})
    js = r.json()
    tz_name = js.get("timezone") or "UTC"
    try:
        tz = ZoneInfo(str(tz_name))
    except Exception:
        tz = ZoneInfo("UTC")
    daily = (js.get("daily") or {}).get("data") or []
    for d in daily:
        ts = d.get("time")
        if ts is None:
            continue
        try:
            dd = dt.datetime.fromtimestamp(int(ts), tz=tz).date()
        except Exception:
            continue
        if dd != trade_dt:
            continue
        return (
            _safe_float(d.get("temperatureMax", d.get("temperatureHigh"))),
            {
                "condition_text": str(d.get("summary") or "").strip(),
                "condition_icon": str(d.get("icon") or "").strip(),
                "cloud_cover": _safe_float(d.get("cloudCover")),
            },
        )
    return (None, {})


def forecast_tmax_weather_gov(*, city: str, trade_dt: dt.date) -> tuple[float | None, dict[str, object]]:
    user_agent = os.getenv("NWS_USER_AGENT")
    if not user_agent:
        return (None, {})
    lat, lon = LATLON[city]
    points_url = f"https://api.weather.gov/points/{lat},{lon}"
    session = (
        requests_cache.CachedSession("Data/weather_gov_cache", expire_after=3600)
        if requests_cache is not None
        else requests.Session()
    )
    headers = {"User-Agent": user_agent, "Accept": "application/geo+json"}
    r = session.get(points_url, headers=headers, timeout=30)
    if r.status_code != 200:
        return (None, {})
    js = r.json()
    props = js.get("properties") or {}
    forecast_url = props.get("forecast")
    if not forecast_url:
        return (None, {})
    r2 = session.get(str(forecast_url), headers=headers, timeout=30)
    if r2.status_code != 200:
        return (None, {})
    js2 = r2.json()
    periods = (js2.get("properties") or {}).get("periods") or []
    for p in periods:
        st = p.get("startTime")
        if not st:
            continue
        try:
            dt_start = dt.datetime.fromisoformat(str(st))
        except Exception:
            continue
        if dt_start.date() != trade_dt:
            continue
        if p.get("isDaytime") is True:
            return (
                _safe_float(p.get("temperature")),
                {
                    "condition_text": str(p.get("shortForecast") or p.get("detailedForecast") or "").strip(),
                    "condition_icon": str(p.get("icon") or "").strip(),
                    "cloud_cover": None,
                },
            )
    # No daytime period for trade_dt. NWS drops the daytime period once it has passed, so
    # after ~19:00 local only "Tonight" (isDaytime=False) remains for today -- and its
    # temperature is the overnight LOW, not the daily max. Falling back to it made
    # weather.gov report ~82F for Miami on days that settled at 91-94F, a -15F bias on 99%
    # of graded rows. Returning None instead lets the ensemble drop the provider for this
    # run, which is correct: we have no daytime max forecast to offer.
    return (None, {})


def _parse_args():
    p = argparse.ArgumentParser(description="Fetch forecasts at specific intraday times.")
    p.add_argument("--trade-date", type=str, required=True, help="YYYY-MM-DD (event date the forecasts target)")
    p.add_argument("--out-csv", type=str, default="Data/intraday_forecasts.csv")
    p.add_argument("--weights-json", type=str, default="Data/weights.json")
    p.add_argument("--env", type=str, default=os.getenv("KALSHI_ENV", "demo"))
    p.add_argument(
        "--write-predictions",
        action="store_true",
        help="Also write Data/predictions_latest.csv and append to predictions_history (for the 22:00 trade run).",
    )
    p.add_argument("--predictions-latest", type=str, default="Data/predictions_latest.csv")
    p.add_argument("--predictions-history", type=str, default="Data/predictions_history.csv")
    p.add_argument(
        "--print",
        action="store_true",
        help="Print fetched forecasts to stdout (useful for docker exec / debugging).",
    )
    p.add_argument(
        "--print-format",
        type=str,
        default="table",
        choices=["table", "json"],
        help="Output format when --print is set (default: table).",
    )
    p.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write Data/*.csv files (still performs API calls; combine with --print).",
    )
    p.add_argument("--performance-csv", type=str, default="Data/source_performance.csv", help="Source performance for MAE-weighted consensus")
    p.add_argument("--mae-window-days", type=int, default=7, help="Rolling window (days) for MAE")
    p.add_argument("--decision-role", type=str, default="monitoring", choices=["trade", "monitoring"])
    p.add_argument(
        "--bandit-mode",
        type=str,
        default=_bandit_mode_default(),
        choices=["off", "shadow", "canary", "live"],
        help="Contextual bandit mode: off|shadow|canary|live (default from WT_BANDIT_MODE).",
    )
    p.add_argument("--bandit-state-path", type=str, default="Data/bandit_state.json")
    p.add_argument("--context-features-csv", type=str, default="Data/context_features_history.csv")
    p.add_argument("--bandit-decisions-csv", type=str, default="Data/bandit_decisions_history.csv")
    p.add_argument("--bandit-alpha", type=float, default=_env_float("WT_BANDIT_ALPHA", 0.7))
    p.add_argument("--bandit-epsilon-shadow", type=float, default=_env_float("WT_BANDIT_EPSILON_SHADOW", 0.15))
    p.add_argument("--bandit-epsilon-canary", type=float, default=_env_float("WT_BANDIT_EPSILON_CANARY", 0.05))
    p.add_argument("--bandit-lambda", type=float, default=_env_float("WT_BANDIT_LAMBDA", 1.0))
    p.add_argument("--bandit-canary-city", type=str, default=str(os.getenv("WT_BANDIT_CANARY_CITY", "ny")).strip().lower())
    p.add_argument("--bandit-max-spread", type=float, default=3.0, help="Guardrail spread threshold for canary apply.")
    p.add_argument("--bandit-min-confidence", type=float, default=0.35, help="Guardrail min confidence for canary apply.")
    p.add_argument(
        "--bandit-max-deviation-f",
        type=float,
        default=6.0,
        help="Guardrail max |selected - forecast| in F before fallback in canary mode.",
    )
    p.add_argument("--blend-forecast-weight", type=float, default=0.8)
    p.add_argument(
        "--max-source-divergence",
        dest="max_source_divergence_f",
        type=float,
        default=3.0,
        help=(
            "Widen sigma when any single source deviates more than this many °F from the "
            "weighted consensus mean. Captures lone-outlier signals (e.g., warm-front days "
            "where one provider is 4°F above the others). Default 3.0°F."
        ),
    )
    p.add_argument(
        "--outlier-rejection-f",
        dest="outlier_rejection_f",
        type=float,
        default=8.0,
        help=(
            "Exclude sources deviating more than this many °F from the weighted consensus "
            "before computing spread/sigma. Prevents corrupt or stale data (e.g. a weather.gov "
            "returning an overnight low as the day max) from artificially inflating spread and "
            "zeroing confidence. Requires ≥2 sources to remain after rejection. Default 8.0°F."
        ),
    )
    p.add_argument(
        "--outlier-max-fraction",
        dest="outlier_max_fraction",
        type=float,
        default=0.35,
        help=(
            "Maximum fraction of sources that may be rejected as outliers. If more sources "
            "than this fraction would be removed, the situation is treated as a genuine bimodal "
            "provider split (real uncertainty) and no rejection is applied. Default 0.35 "
            "(≤35%% rejected = at most 2 out of 8 sources). Prevents single-cluster "
            "survivorship from masking genuine disagreement."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    trade_dt = dt.datetime.strptime(args.trade_date, "%Y-%m-%d").date()
    weights_all = _load_weights(args.weights_json)
    tomorrow_state: dict[str, float | None] = {"last_req_ts": None}

    started = _now_iso_local()
    wrote = 0
    pred_rows: list[dict] = []
    intraday_rows: list[dict] = []

    policy, policy_state = load_policy_state(
        args.bandit_state_path,
        alpha=float(args.bandit_alpha),
        reg_lambda=float(args.bandit_lambda),
        epsilon=0.0,
    )
    if args.bandit_mode == "shadow":
        policy.set_epsilon(float(args.bandit_epsilon_shadow))
    elif args.bandit_mode in ("canary", "live"):
        policy.set_epsilon(float(args.bandit_epsilon_canary))
    else:
        policy.set_epsilon(0.0)
    if args.bandit_mode != "off" and not args.no_write:
        save_policy_state(args.bandit_state_path, policy, policy_state)

    rolling_mae: dict[str, dict[str, float]] = {}
    if getattr(args, "performance_csv", None) and os.path.exists(getattr(args, "performance_csv", "")):
        from prediction_mae import get_rolling_mae_per_city_source
        mae_end = trade_dt - dt.timedelta(days=1)
        rolling_mae = get_rolling_mae_per_city_source(
            args.performance_csv,
            window_days=getattr(args, "mae_window_days", 7),
            end_date=mae_end,
        )

    # Load per-city bias corrections from city_metadata.json.
    # bias_correction_f > 0 means we historically run cold (under-predict); add it to forecast.
    city_bias: dict[str, float] = {}
    city_bias_by_condition: dict[str, dict[str, float]] = {}
    city_mae_by_condition: dict[str, dict[str, float]] = {}
    city_historical_mae: dict[str, float] = {}
    city_metadata_path = getattr(args, "city_metadata_json", "Data/city_metadata.json")
    if city_metadata_path and os.path.exists(city_metadata_path):
        try:
            import json as _json
            with open(city_metadata_path) as _f:
                _meta = _json.load(_f) or {}
            for _city, _info in (_meta.get("cities") or {}).items():
                _city_key = str(_city).strip().lower()
                if not isinstance(_info, dict):
                    continue
                _bc = _info.get("bias_correction_f")
                if _bc is not None:
                    city_bias[_city_key] = float(_bc)
                _by_cond = _info.get("bias_correction_by_condition")
                if isinstance(_by_cond, dict) and _by_cond:
                    city_bias_by_condition[_city_key] = {str(k): float(v) for k, v in _by_cond.items()}
                _mae_cond = _info.get("mae_by_condition")
                if isinstance(_mae_cond, dict) and _mae_cond:
                    city_mae_by_condition[_city_key] = {str(k): float(v) for k, v in _mae_cond.items()}
                _base_mae = _info.get("historical_MAE")
                if _base_mae is not None:
                    city_historical_mae[_city_key] = float(_base_mae)
        except Exception:
            pass

    # Lead 0/1: fetch and store forecasts for both today (lead 0) and tomorrow (lead 1).
    lead_dates = [
        trade_dt,
        trade_dt + dt.timedelta(days=1),
    ]

    for target_dt in lead_dates:
        trade_date_str = target_dt.strftime("%Y-%m-%d")
        for city in CITIES:
            tmax_open_meteo, ctx_open_meteo = _parse_provider_result(
                _try_call(forecast_tmax_open_meteo, city=city, trade_dt=target_dt)
            )
            tmax_visual_crossing, ctx_visual_crossing = _parse_provider_result(
                _try_call(forecast_tmax_visual_crossing, city=city, trade_dt=target_dt)
            )
            tmax_tomorrow, ctx_tomorrow = _parse_provider_result(
                _try_call(forecast_tmax_tomorrow, city=city, trade_dt=target_dt, _state=tomorrow_state)
            )
            tmax_weatherapi, ctx_weatherapi = _parse_provider_result(
                _try_call(forecast_tmax_weatherapi, city=city, trade_dt=target_dt)
            )
            tmax_google_weather, ctx_google_weather = _parse_provider_result(
                _try_call(forecast_tmax_google_weather, city=city, trade_dt=target_dt)
            )
            tmax_openweathermap, ctx_openweathermap = _parse_provider_result(
                _try_call(forecast_tmax_openweathermap, city=city, trade_dt=target_dt)
            )
            tmax_pirateweather, ctx_pirateweather = _parse_provider_result(
                _try_call(forecast_tmax_pirateweather, city=city, trade_dt=target_dt)
            )
            tmax_weather_gov, ctx_weather_gov = _parse_provider_result(
                _try_call(forecast_tmax_weather_gov, city=city, trade_dt=target_dt)
            )

            vals = {
                "google-weather": tmax_google_weather,
                "open-meteo": tmax_open_meteo,
                "openweathermap": tmax_openweathermap,
                "pirateweather": tmax_pirateweather,
                "visual-crossing": tmax_visual_crossing,
                "tomorrow": tmax_tomorrow,
                "weatherapi": tmax_weatherapi,
                "weather.gov": tmax_weather_gov,
            }
            available = {k: float(v) for k, v in vals.items() if v is not None}
            provider_contexts = {
                "google-weather": ctx_google_weather,
                "open-meteo": ctx_open_meteo,
                "openweathermap": ctx_openweathermap,
                "pirateweather": ctx_pirateweather,
                "visual-crossing": ctx_visual_crossing,
                "tomorrow": ctx_tomorrow,
                "weatherapi": ctx_weatherapi,
                "weather.gov": ctx_weather_gov,
            }

            mae_map = rolling_mae.get(city, {}) if rolling_mae else {}
            weights_used = {}
            for src in available:
                mae = mae_map.get(src)
                if mae is not None:
                    mae_safe = max(float(mae), 0.01)
                    weights_used[src] = 1.0 / (mae_safe * mae_safe)
            if weights_used:
                s = sum(weights_used.values())
                weights_used = {k: v / s for k, v in weights_used.items()}
            else:
                w_city = _weights_for_city(weights_all, city)
                weights_used = {k: float(w_city[k]) for k in available if k in w_city}
                if not weights_used and available:
                    u = 1.0 / len(available)
                    weights_used = {k: u for k in available}
                elif weights_used:
                    s = sum(weights_used.values())
                    weights_used = {k: v / s for k, v in weights_used.items()} if s > 0 else {k: 1.0 / len(available) for k in available}

            history_rows = _load_recent_intraday_history(
                args.out_csv,
                city=city,
                trade_date=trade_date_str,
                max_rows=4,
            )
            weights_used, stability_score = _apply_volatility_weighting(
                available, weights_used, history_rows
            )

            mean_forecast = (
                sum(weights_used[k] * available[k] for k in weights_used.keys()) if weights_used else None
            )

            # Outlier rejection: before computing spread, remove sources whose value
            # deviates more than outlier_rejection_f from the weighted consensus mean.
            # This prevents stale/corrupt data (e.g. weather.gov returning 27°F in March
            # when all others say 54°F) from inflating sigma and zeroing confidence.
            # The consensus mean is already robust to outliers via weights; only sigma suffers.
            #
            # Safety valve: if more than outlier_max_fraction of sources would be rejected,
            # we treat the situation as a genuine bimodal split (providers truly disagree)
            # and leave sigma alone rather than silently suppressing the disagreement.
            # E.g. NY today: 5/8 sources rejected → 62% > 35% → no rejection, high sigma kept.
            # Single-outlier case: 1/8 → 12% ≤ 35% → rejection applies.
            # Guard: ≥2 sources must also remain after rejection.
            outliers_rejected: list[str] = []
            outlier_threshold = float(args.outlier_rejection_f)
            outlier_max_fraction = float(getattr(args, "outlier_max_fraction", 0.35))
            if mean_forecast is not None and len(available) > 2:
                _candidate_rejected = [
                    s for s, v in available.items()
                    if abs(v - mean_forecast) > outlier_threshold
                ]
                _fraction = len(_candidate_rejected) / len(available)
                if (
                    _candidate_rejected
                    and _fraction <= outlier_max_fraction
                    and len(available) - len(_candidate_rejected) >= 2
                ):
                    outliers_rejected = sorted(_candidate_rejected)
                    print(
                        f"[outlier_rejection] {city}: excluded {outliers_rejected} "
                        f"(>{outlier_threshold:.1f}°F from consensus {mean_forecast:.1f}°F)"
                    )
                elif _candidate_rejected and _fraction > outlier_max_fraction:
                    print(
                        f"[outlier_rejection] {city}: {len(_candidate_rejected)}/{len(available)} sources "
                        f"would be rejected ({_fraction:.0%} > {outlier_max_fraction:.0%} cap) — "
                        f"treating as genuine bimodal split, keeping high sigma"
                    )
            available_for_spread = (
                {s: v for s, v in available.items() if s not in outliers_rejected}
                if outliers_rejected else available
            )

            sources_with_mae = [s for s in available_for_spread if s in mae_map]
            if sources_with_mae:
                best_mae = min(mae_map[s] for s in sources_with_mae)
                reliable = [s for s in sources_with_mae if mae_map[s] <= 1.5 * best_mae]
                sigma = float(statistics.pstdev([available_for_spread[s] for s in reliable])) if len(reliable) >= 2 else 0.0
                mae_sorted = sorted(mae_map[s] for s in sources_with_mae)
                bonus = 0.1 if len(mae_sorted) >= 2 and mae_sorted[0] < 0.8 * mae_sorted[1] else 0.0
                # Sources with poor trailing MAE for this city don't get a say in spread:
                # sigma above already uses only the reliable set, and the divergence
                # guardrail below must match, or a known-bad provider (e.g. weatherapi in
                # NY at 6+°F MAE) re-inflates sigma despite near-zero consensus weight.
                spread_sources = (
                    {s: available_for_spread[s] for s in reliable}
                    if len(reliable) >= 2
                    else available_for_spread
                )
            else:
                sigma = float(statistics.pstdev(list(available_for_spread.values()))) if len(available_for_spread) > 1 else (0.0 if available_for_spread else None)
                bonus = 0.0
                spread_sources = available_for_spread

            # Max-source-divergence guardrail: if any reliable source still deviates more
            # than max_source_divergence_f, widen sigma. This catches genuine
            # warm/cold-front days where one provider is early on a real move.
            if mean_forecast is not None and spread_sources and sigma is not None:
                max_src_dev = max(abs(v - mean_forecast) for v in spread_sources.values())
                if max_src_dev > float(args.max_source_divergence_f):
                    sigma = max(sigma, max_src_dev / 2.0)

            context_vote = vote_provider_conditions(
                provider_contexts,
                provider_weights=weights_used if weights_used else None,
            )
            provider_count = len(available)
            condition_token = str(context_vote.get("condition_token") or "other").strip().lower()
            condition_label = str(context_vote.get("condition_label") or "").strip()
            sky_label = str(context_vote.get("sky_label") or "mixed").strip().lower()
            mean_cloud_cover = _safe_float(context_vote.get("mean_cloud_cover"))
            vote_entropy = _safe_float(context_vote.get("vote_entropy")) or 0.0

            spread_conf_raw = _confidence_from_spread(float(sigma)) if sigma is not None else 0.0
            spread_conf = min(0.9, max(0.0, float(spread_conf_raw)) + bonus)
            skill_conf = _skill_from_weights(weights_used, mae_map=mae_map)
            sigma_conf = spread_conf * (0.5 + 0.5 * skill_conf) if sigma is not None else None
            # Blend in realized accuracy: sigma_conf only measures provider agreement, so
            # it stays flat even as the city's trailing consensus MAE improves. Anchor half
            # of the score to actual settled-vs-predicted skill (historical_MAE from
            # city_metadata.json) so confidence rises when we are demonstrably accurate.
            _realized_mae = city_historical_mae.get(city)
            if sigma_conf is None:
                conf_final = None
            elif _realized_mae is not None and _realized_mae > 0:
                conf_final = 0.5 * sigma_conf + 0.5 * _mae_to_skill(_realized_mae)
            else:
                conf_final = sigma_conf

            # Apply condition-aware multiplier: learned MAE per condition bucket relative to city
            # average. Clear/sunny days get a boost; precip/storm days get a penalty.
            # vote_entropy independently penalizes provider disagreement on conditions.
            if conf_final is not None:
                cond_factor = _condition_confidence_factor(
                    condition_token,
                    vote_entropy,
                    city_mae_by_condition.get(city),
                    city_historical_mae.get(city),
                )
                conf_final = max(0.0, min(1.0, conf_final * cond_factor))

            # 4) Conviction score: blend confidence with stability of recent provider skews.
            conviction_score: float | None
            if conf_final is None:
                conviction_score = None
            else:
                # conf_final is in [0, ~0.9]; stability_score is in [0,1].
                # Rescale so that (high confidence & high stability) ≈ 1.0.
                raw = float(conf_final) * float(stability_score)
                conviction_score = max(0.0, min(1.0, raw / 0.9)) if raw > 0 else 0.0

            ts_now = _now_iso_local()
            candidate_modes = compute_candidate_mode_predictions(
                city=city,
                forecast_pred=mean_forecast,
                bias_correction_f=city_bias.get(city, 0.0),
                bias_correction_by_condition=city_bias_by_condition.get(city),
                condition_token=condition_token,
            )
            mode_forecast_pred = _safe_float(candidate_modes.get("mode_forecast_pred"))
            mode_blend_pred = _safe_float(candidate_modes.get("mode_blend_pred"))
            mode_lstm_pred = _safe_float(candidate_modes.get("mode_lstm_pred"))
            candidate_pred_map: dict[str, float | None] = {
                "forecast": mode_forecast_pred,
                "blend": mode_blend_pred,
                "lstm": mode_lstm_pred,
            }
            available_actions = [a for a, v in candidate_pred_map.items() if v is not None]

            selected_action = "forecast"
            applied_action = "forecast"
            action_reason = "bandit_off"
            guardrail_reason = ""
            policy_scores: dict[str, object] = {}
            feature_vector = []
            feature_map: dict[str, float] = {}

            if args.bandit_mode in ("shadow", "canary", "live") and available_actions:
                fvec, fmap = build_feature_vector(
                    city=city,
                    trade_date=target_dt,
                    spread_f=sigma,
                    provider_count=provider_count,
                    condition_token=condition_token,
                    sky_label=sky_label,
                    mean_cloud_cover=mean_cloud_cover,
                    vote_entropy=vote_entropy,
                )
                feature_vector = [round(float(v), 8) for v in fvec.tolist()]
                feature_map = {k: float(v) for k, v in fmap.items()}
                rng = random.Random(_bandit_seed_for(city, trade_date_str, ts_now))
                selected_action, score_info = policy.select_action(
                    fvec,
                    available_actions=available_actions,
                    rng=rng,
                )
                policy_scores = score_info
                action_reason = str(score_info.get("selected_via") or "linucb")
                selected_action, selected_pred, select_reason = choose_mode_prediction(
                    selected_action=selected_action,
                    candidates=candidate_pred_map,
                    fallback_action="forecast",
                )
                if select_reason != "selected":
                    guardrail_reason = select_reason

                if args.bandit_mode == "shadow":
                    applied_action, _, _ = choose_mode_prediction(
                        selected_action="forecast",
                        candidates=candidate_pred_map,
                        fallback_action=selected_action,
                    )
                    guardrail_reason = ";".join([x for x in [guardrail_reason, "shadow_no_apply"] if x])
                else:
                    # Scope gate: canary restricts to one city; live applies to all cities.
                    if args.bandit_mode == "live":
                        scope_ok = str(args.decision_role).strip().lower() == "trade"
                        scope_label = "live_monitoring_no_apply"
                    else:  # canary
                        scope_ok = bool(
                            city == str(args.bandit_canary_city).strip().lower()
                            and str(args.decision_role).strip().lower() == "trade"
                        )
                        scope_label = "canary_scope"
                    if not scope_ok:
                        applied_action, _, _ = choose_mode_prediction(
                            selected_action="forecast",
                            candidates=candidate_pred_map,
                            fallback_action=selected_action,
                        )
                        guardrail_reason = ";".join([x for x in [guardrail_reason, scope_label] if x])
                    else:
                        applied_action = selected_action
                        applied_pred = selected_pred
                        if sigma is not None and float(sigma) > float(args.bandit_max_spread):
                            applied_action, applied_pred, _ = choose_mode_prediction(
                                selected_action="forecast",
                                candidates=candidate_pred_map,
                                fallback_action=selected_action,
                            )
                            guardrail_reason = ";".join([x for x in [guardrail_reason, "spread_guardrail"] if x])
                        if conf_final is not None and float(conf_final) < float(args.bandit_min_confidence):
                            applied_action, applied_pred, _ = choose_mode_prediction(
                                selected_action="forecast",
                                candidates=candidate_pred_map,
                                fallback_action=selected_action,
                            )
                            guardrail_reason = ";".join([x for x in [guardrail_reason, "confidence_guardrail"] if x])
                        if (
                            applied_pred is not None
                            and mode_forecast_pred is not None
                            and abs(float(applied_pred) - float(mode_forecast_pred)) > float(args.bandit_max_deviation_f)
                        ):
                            applied_action, _, _ = choose_mode_prediction(
                                selected_action="forecast",
                                candidates=candidate_pred_map,
                                fallback_action=selected_action,
                            )
                            guardrail_reason = ";".join([x for x in [guardrail_reason, "deviation_guardrail"] if x])

            applied_action, applied_prediction, apply_reason = choose_mode_prediction(
                selected_action=applied_action,
                candidates=candidate_pred_map,
                fallback_action="forecast",
            )
            if apply_reason != "selected":
                guardrail_reason = ";".join([x for x in [guardrail_reason, apply_reason] if x])

            row = {
                "timestamp": ts_now,
                "city": city,
                "trade_date": trade_date_str,
                "mean_forecast": "" if applied_prediction is None else f"{applied_prediction:.4f}",
                "current_sigma": "" if sigma is None else f"{sigma:.4f}",
                "tmax_open_meteo": "" if tmax_open_meteo is None else f"{tmax_open_meteo:.4f}",
                "tmax_visual_crossing": "" if tmax_visual_crossing is None else f"{tmax_visual_crossing:.4f}",
                "tmax_tomorrow": "" if tmax_tomorrow is None else f"{tmax_tomorrow:.4f}",
                "tmax_weatherapi": "" if tmax_weatherapi is None else f"{tmax_weatherapi:.4f}",
                "tmax_google_weather": "" if tmax_google_weather is None else f"{tmax_google_weather:.4f}",
                "tmax_openweathermap": "" if tmax_openweathermap is None else f"{tmax_openweathermap:.4f}",
                "tmax_pirateweather": "" if tmax_pirateweather is None else f"{tmax_pirateweather:.4f}",
                "tmax_weather_gov": "" if tmax_weather_gov is None else f"{tmax_weather_gov:.4f}",
                "sources_used": ",".join([s for s in SOURCES_ORDER if s in available]),
                "weights_used": ",".join(
                    [f"{k}:{weights_used[k]:.4f}" for k in sorted(weights_used.keys())]
                )
                if weights_used
                else "",
                "outliers_rejected": ",".join(outliers_rejected),
            }
            intraday_rows.append(row)
            if not args.no_write:
                _append_intraday_row(args.out_csv, row)
                wrote += 1
                _append_context_feature_row(
                    args.context_features_csv,
                    {
                        "run_ts": ts_now,
                        "decision_role": args.decision_role,
                        "bandit_mode": args.bandit_mode,
                        "city": city,
                        "trade_date": trade_date_str,
                        "provider_count": str(provider_count),
                        "spread_f": "" if sigma is None else f"{sigma:.4f}",
                        "condition_token": condition_token,
                        "condition_label": condition_label,
                        "sky_label": sky_label,
                        "mean_cloud_cover": "" if mean_cloud_cover is None else f"{float(mean_cloud_cover):.4f}",
                        "vote_entropy": f"{float(vote_entropy):.6f}",
                        "raw_provider_labels_json": str(context_vote.get("raw_provider_labels_json") or "{}"),
                        "token_weights_json": str(context_vote.get("token_weights_json") or "{}"),
                    },
                )
                _append_bandit_decision_row(
                    args.bandit_decisions_csv,
                    {
                        "run_ts": ts_now,
                        "decision_role": args.decision_role,
                        "bandit_mode": args.bandit_mode,
                        "city": city,
                        "trade_date": trade_date_str,
                        "selected_action": selected_action,
                        "applied_action": applied_action,
                        "action_reason": action_reason,
                        "guardrail_reason": guardrail_reason,
                        "mode_forecast_pred": "" if mode_forecast_pred is None else f"{mode_forecast_pred:.4f}",
                        "mode_blend_pred": "" if mode_blend_pred is None else f"{mode_blend_pred:.4f}",
                        "mode_lstm_pred": "" if mode_lstm_pred is None else f"{mode_lstm_pred:.4f}",
                        "feature_vector_json": _safe_json_dumps(feature_vector),
                        "feature_map_json": _safe_json_dumps(feature_map),
                        "policy_scores_json": _safe_json_dumps(policy_scores),
                        "condition_token": condition_token,
                        "condition_label": condition_label,
                        "sky_label": sky_label,
                        "mean_cloud_cover": "" if mean_cloud_cover is None else f"{float(mean_cloud_cover):.4f}",
                        "vote_entropy": f"{float(vote_entropy):.6f}",
                        "provider_count": str(provider_count),
                        "spread_f": "" if sigma is None else f"{sigma:.4f}",
                        "raw_provider_labels_json": str(context_vote.get("raw_provider_labels_json") or "{}"),
                    },
                )

            # Write predictions for both today and tomorrow so the dashboard always has
            # the next trade date (e.g. after 7 PM ET we show tomorrow; file must have it).
            if args.write_predictions:
                conf = "" if conf_final is None else f"{float(conf_final):.4f}"
                conviction_str = (
                    "" if conviction_score is None else f"{float(conviction_score):.4f}"
                )
                pred_rows.append(
                    {
                        "date": trade_date_str,
                        "city": city,
                        "tmax_predicted": "" if applied_prediction is None else f"{applied_prediction:.4f}",
                        "tmax_lstm": "" if mode_lstm_pred is None else f"{mode_lstm_pred:.4f}",
                        "tmax_forecast": "" if mode_forecast_pred is None else f"{mode_forecast_pred:.4f}",
                        "spread_f": "" if sigma is None else f"{float(sigma):.4f}",
                        "confidence_score": conf,
                        "conviction_score": conviction_str,
                        "forecast_sources": ",".join([s for s in SOURCES_ORDER if s in available]),
                        "tmax_open_meteo": "" if tmax_open_meteo is None else f"{tmax_open_meteo:.4f}",
                        "tmax_visual_crossing": ""
                        if tmax_visual_crossing is None
                        else f"{tmax_visual_crossing:.4f}",
                        "tmax_tomorrow": "" if tmax_tomorrow is None else f"{tmax_tomorrow:.4f}",
                        "tmax_weatherapi": ""
                        if tmax_weatherapi is None
                        else f"{tmax_weatherapi:.4f}",
                        "tmax_google_weather": ""
                        if tmax_google_weather is None
                        else f"{tmax_google_weather:.4f}",
                        "tmax_openweathermap": ""
                        if tmax_openweathermap is None
                        else f"{tmax_openweathermap:.4f}",
                        "tmax_pirateweather": ""
                        if tmax_pirateweather is None
                        else f"{tmax_pirateweather:.4f}",
                        "tmax_weather_gov": ""
                        if tmax_weather_gov is None
                        else f"{tmax_weather_gov:.4f}",
                        "sources_used": ",".join([s for s in SOURCES_ORDER if s in available]),
                        "weights_used": ",".join(
                            [f"{k}:{weights_used[k]:.4f}" for k in sorted(weights_used.keys())]
                        )
                        if weights_used
                        else "",
                    }
                )

    if args.print:
        if args.print_format == "json":
            payload = {
                "started": started,
                "trade_date": trade_dt.isoformat(),
                "env": args.env,
                "intraday_rows": intraday_rows,
                "prediction_rows": pred_rows,
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            # Human-readable summary.
            print(f"[intraday_pulse.py] started={started} trade_date={trade_dt.isoformat()} env={args.env}")
            for r in intraday_rows:
                city = r.get("city")
                mean_f = r.get("mean_forecast")
                sig = r.get("current_sigma")
                print(f"\n--- {city} ---")
                print(f"mean_forecast={mean_f} sigma={sig}")
                print(f"sources_used={r.get('sources_used')}")
                print(f"weights_used={r.get('weights_used')}")
                # Provider values (compact).
                for k in (
                    "tmax_open_meteo",
                    "tmax_visual_crossing",
                    "tmax_tomorrow",
                    "tmax_weatherapi",
                    "tmax_google_weather",
                    "tmax_openweathermap",
                    "tmax_pirateweather",
                    "tmax_weather_gov",
                ):
                    v = r.get(k, "")
                    if str(v).strip() != "":
                        print(f"{k}={v}")

    if args.write_predictions:
        if not args.no_write:
            _write_predictions_latest(args.predictions_latest, pred_rows)
            _append_predictions_history(
                args.predictions_history,
                pred_rows,
                extra_fields={
                    "run_ts": _now_iso_local(),
                    "env": args.env,
                    "prediction_mode": "forecast",
                    "blend_forecast_weight": str(args.blend_forecast_weight),
                    "refresh_history": "False",
                    "retrain_lstm": "False",
                },
            )

    print(
        f"[intraday_pulse.py] done start={started} trade_date={trade_dt.isoformat()} "
        f"cities_written={wrote if not args.no_write else 0} out={args.out_csv}"
        + (f" predictions_latest={args.predictions_latest}" if (args.write_predictions and not args.no_write) else "")
        + (" (no-write)" if args.no_write else "")
    )
