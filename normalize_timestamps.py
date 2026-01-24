import argparse
import csv
import datetime as dt
import json
import os
import shutil
from pathlib import Path


DEFAULT_TZ = "America/New_York"


def _get_tz(tzname: str) -> dt.tzinfo:
    tzname = (tzname or "").strip() or DEFAULT_TZ
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(tzname)
    except Exception:
        return dt.datetime.now().astimezone().tzinfo or dt.timezone.utc


def _parse_iso(s: str) -> dt.datetime | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        # tolerate Z
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _should_convert(s: str) -> bool:
    ss = (s or "").strip()
    if not ss:
        return False
    # Only rewrite values that are explicitly UTC (Z or +00:00). Leave already-local offsets alone.
    return ss.endswith("Z") or ss.endswith("+00:00")


def _to_local_iso(s: str, tz: dt.tzinfo) -> str:
    t = _parse_iso(s)
    if t is None:
        return s
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    try:
        return t.astimezone(tz).isoformat()
    except Exception:
        return t.isoformat()


CSV_TIME_FIELDS = {
    "timestamp",
    "run_ts",
    "updated_at",
    "settled_at",
}

JSON_TIME_FIELDS = {
    "updated_at",
    "run_ts",
    "settled_at",
}


def normalize_csv(path: Path, tz: dt.tzinfo, *, in_place: bool, backup: bool) -> tuple[int, int]:
    # Special-case: predictions_history.csv can accumulate mixed schemas over time.
    # Normalize it to the current canonical schema before timestamp normalization.
    if path.as_posix().endswith("Data/predictions_history.csv"):
        return _normalize_predictions_history(path, tz, in_place=in_place, backup=backup)
    if path.as_posix().endswith("Data/hourly_forecasts.csv"):
        return _normalize_hourly_forecasts(path, tz, in_place=in_place, backup=backup)

    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        fieldnames = list(r.fieldnames or [])
        rows = list(r)

    changed = 0
    scanned = 0
    for row in rows:
        if not row:
            continue
        # Some CSVs can have malformed rows (extra columns); csv.DictReader stores them under None.
        # Drop those so we can safely reserialize.
        if None in row:
            row.pop(None, None)
        for k in list(row.keys()):
            if k not in CSV_TIME_FIELDS:
                continue
            v = (row.get(k) or "").strip()
            if not v:
                continue
            scanned += 1
            if not _should_convert(v):
                continue
            nv = _to_local_iso(v, tz)
            if nv != v:
                row[k] = nv
                changed += 1

    if not in_place:
        return scanned, changed

    if backup:
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(path, bak)

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    os.replace(tmp, path)
    return scanned, changed


def _normalize_hourly_forecasts(path: Path, tz: dt.tzinfo, *, in_place: bool, backup: bool) -> tuple[int, int]:
    """
    Normalize Data/hourly_forecasts.csv to a consistent header and convert UTC timestamps to local tz.
    (This file may contain mixed schemas if provider columns were added over time.)
    """
    old_hdr = [
        "timestamp",
        "city",
        "mean_forecast",
        "spread_at_time",
        "trade_date",
        "tmax_open_meteo",
        "tmax_visual_crossing",
        "tmax_tomorrow",
        "tmax_weatherapi",
        "tmax_openweathermap",
        "tmax_pirateweather",
        "tmax_weather_gov",
        "sources_used",
        "weights_used",
    ]
    new_hdr = [
        "timestamp",
        "city",
        "mean_forecast",
        "spread_at_time",
        "trade_date",
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
    ]

    with path.open("r", newline="") as f:
        r = csv.reader(f)
        _ = next(r, None)
        raw_rows = list(r)

    scanned = 0
    changed = 0
    out_rows: list[dict[str, str]] = []
    for row in raw_rows:
        if not row:
            continue
        if len(row) == len(old_hdr):
            d = dict(zip(old_hdr, row))
        elif len(row) == len(new_hdr):
            d = dict(zip(new_hdr, row))
        else:
            d = dict(zip(old_hdr, row[: len(old_hdr)]))

        ts = (d.get("timestamp") or "").strip()
        if ts:
            scanned += 1
            if _should_convert(ts):
                nts = _to_local_iso(ts, tz)
                if nts != ts:
                    d["timestamp"] = nts
                    changed += 1

        out_rows.append(d)

    if not in_place:
        return scanned, changed

    if backup:
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(path, bak)

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=new_hdr)
        w.writeheader()
        for d in out_rows:
            w.writerow({k: d.get(k, "") for k in new_hdr})
    os.replace(tmp, path)
    return scanned, changed


def _canonical_predictions_history_header() -> list[str]:
    """
    Build the canonical predictions_history header from the current predictions_latest header plus metadata fields.
    """
    latest = Path("Data/predictions_latest.csv")
    if latest.exists():
        with latest.open("r", newline="") as f:
            r = csv.reader(f)
            hdr = next(r, [])
            base = [h.strip() for h in hdr if h and h.strip()]
    else:
        base = []
    extras = ["run_ts", "env", "prediction_mode", "blend_forecast_weight", "refresh_history", "retrain_lstm"]
    out: list[str] = []
    for h in base + extras:
        if h not in out:
            out.append(h)
    # Fall back to the known older schema if predictions_latest is missing.
    if not out:
        out = [
            "date",
            "city",
            "tmax_predicted",
            "tmax_lstm",
            "tmax_forecast",
            "forecast_sources",
            "run_ts",
            "env",
            "prediction_mode",
            "blend_forecast_weight",
            "refresh_history",
            "retrain_lstm",
        ]
    return out


def _normalize_predictions_history(path: Path, tz: dt.tzinfo, *, in_place: bool, backup: bool) -> tuple[int, int]:
    """
    Normalize Data/predictions_history.csv to a consistent header and convert UTC run_ts to local tz.
    """
    # Known schemas we may encounter (based on how the project evolved):
    hdr_12 = [
        "date",
        "city",
        "tmax_predicted",
        "tmax_lstm",
        "tmax_forecast",
        "forecast_sources",
        "run_ts",
        "env",
        "prediction_mode",
        "blend_forecast_weight",
        "refresh_history",
        "retrain_lstm",
    ]
    hdr_20 = [
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

    canonical = _canonical_predictions_history_header()
    # If canonical includes more provider columns than hdr_20, we can pad missing fields by position.

    with path.open("r", newline="") as f:
        r = csv.reader(f)
        _ = next(r, None)  # ignore existing header (can be stale)
        raw_rows = list(r)

    scanned = 0
    changed = 0
    out_rows: list[dict[str, str]] = []

    for row in raw_rows:
        if not row:
            continue
        row = [c for c in row]  # copy
        if len(row) == len(hdr_12):
            d = dict(zip(hdr_12, row))
        elif len(row) == len(hdr_20):
            d = dict(zip(hdr_20, row))
        elif len(row) == len(canonical):
            d = dict(zip(canonical, row))
        else:
            # Best-effort: map first len(hdr_12) fields, drop the rest.
            d = dict(zip(hdr_12, row[: len(hdr_12)]))

        # Normalize run_ts if it is UTC.
        rt = (d.get("run_ts") or "").strip()
        if rt:
            scanned += 1
            if _should_convert(rt):
                nrt = _to_local_iso(rt, tz)
                if nrt != rt:
                    d["run_ts"] = nrt
                    changed += 1

        out_rows.append(d)

    if not in_place:
        return scanned, changed

    if backup:
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(path, bak)

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=canonical)
        w.writeheader()
        for d in out_rows:
            w.writerow({k: d.get(k, "") for k in canonical})
    os.replace(tmp, path)
    return scanned, changed


def normalize_json(path: Path, tz: dt.tzinfo, *, in_place: bool, backup: bool) -> tuple[int, int]:
    payload = json.loads(path.read_text())
    scanned = 0
    changed = 0

    def visit(obj):
        nonlocal scanned, changed
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                if k in JSON_TIME_FIELDS and isinstance(v, str) and v.strip():
                    scanned += 1
                    if _should_convert(v):
                        nv = _to_local_iso(v, tz)
                        if nv != v:
                            obj[k] = nv
                            changed += 1
                visit(v)
        elif isinstance(obj, list):
            for it in obj:
                visit(it)

    visit(payload)

    if not in_place:
        return scanned, changed

    if backup:
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(path, bak)

    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return scanned, changed


def _parse_args():
    p = argparse.ArgumentParser(description="Rewrite UTC ISO timestamps to local TZ (for readability).")
    p.add_argument("--tz", type=str, default=os.getenv("TZ", "") or DEFAULT_TZ)
    p.add_argument("--in-place", action="store_true", help="Rewrite files in place (default: dry-run).")
    p.add_argument("--no-backup", action="store_true", help="Do not create .bak backups when rewriting.")
    p.add_argument(
        "paths",
        nargs="*",
        default=[
            "Data/hourly_forecasts.csv",
            "Data/predictions_history.csv",
            "Data/trades_history.csv",
            "Data/decisions_history.csv",
            "Data/eval_history.csv",
            "Data/source_performance.csv",
            "Data/weights_history.csv",
            "Data/daily_metrics.csv",
            "Data/city_metadata.json",
        ],
        help="Files to normalize (defaults to common Data/* outputs).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    tz = _get_tz(args.tz)
    backup = not bool(args.no_backup)
    in_place = bool(args.in_place)

    print(f"normalize_timestamps: tz={getattr(tz,'key',str(tz))} in_place={in_place} backup={backup}")
    for p in args.paths:
        path = Path(p)
        if not path.exists():
            print(f"- {p}: missing")
            continue
        try:
            if path.suffix.lower() == ".csv":
                scanned, changed = normalize_csv(path, tz, in_place=in_place, backup=backup)
            elif path.suffix.lower() == ".json":
                scanned, changed = normalize_json(path, tz, in_place=in_place, backup=backup)
            else:
                print(f"- {p}: skip (unsupported extension)")
                continue
            print(f"- {p}: scanned={scanned} changed={changed}" + ("" if in_place else " (dry-run)"))
        except Exception as e:
            print(f"- {p}: ERROR {e}")

