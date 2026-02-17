import argparse
import csv
import datetime as dt
import json
import os
from typing import Any

from bandit.policy import build_feature_vector, load_policy_state, save_policy_state

try:
    import db  # type: ignore
except Exception:
    db = None  # type: ignore[assignment]


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


def _parse_iso(s: str) -> dt.datetime:
    ss = str(s or "").strip()
    if not ss:
        return dt.datetime.min.replace(tzinfo=dt.timezone.utc)
    try:
        t = dt.datetime.fromisoformat(ss.replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone.utc)
        return t
    except Exception:
        return dt.datetime.min.replace(tzinfo=dt.timezone.utc)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Update contextual bandit state from settled actuals.")
    p.add_argument("--trade-date", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument("--decisions-csv", type=str, default="Data/bandit_decisions_history.csv")
    p.add_argument("--performance-csv", type=str, default="Data/source_performance.csv")
    p.add_argument("--rewards-csv", type=str, default="Data/bandit_rewards_history.csv")
    p.add_argument("--state-path", type=str, default="Data/bandit_state.json")
    p.add_argument("--state-snapshots-csv", type=str, default="Data/bandit_state_snapshots.csv")
    p.add_argument("--max-error-f", type=float, default=10.0, help="Error cap for normalized reward")
    p.add_argument("--alpha", type=float, default=float(os.getenv("WT_BANDIT_ALPHA", "0.7")))
    p.add_argument("--reg-lambda", type=float, default=float(os.getenv("WT_BANDIT_LAMBDA", "1.0")))
    return p.parse_args()


def _load_latest_decisions(path: str, trade_date: str) -> dict[str, dict[str, Any]]:
    if not path or not os.path.exists(path):
        return {}
    best: dict[str, dict[str, Any]] = {}
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if (row.get("trade_date") or "").strip() != trade_date:
                continue
            city = (row.get("city") or "").strip().lower()
            if not city:
                continue
            role = (row.get("decision_role") or "").strip().lower()
            pri = 0 if role == "trade" else 1
            ts = _parse_iso(row.get("run_ts") or row.get("timestamp") or "")
            prev = best.get(city)
            if prev is None:
                best[city] = dict(row)
                best[city]["_pri"] = pri
                best[city]["_ts"] = ts
                continue
            prev_pri = int(prev.get("_pri", 99))
            prev_ts = prev.get("_ts") if isinstance(prev.get("_ts"), dt.datetime) else dt.datetime.min.replace(tzinfo=dt.timezone.utc)
            if pri < prev_pri or (pri == prev_pri and ts >= prev_ts):
                best[city] = dict(row)
                best[city]["_pri"] = pri
                best[city]["_ts"] = ts
    return best


def _load_actuals(path: str, trade_date: str) -> dict[str, float]:
    if not path or not os.path.exists(path):
        return {}
    out: dict[str, float] = {}
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if (row.get("date") or "").strip() != trade_date:
                continue
            if (row.get("source_name") or "").strip() != "consensus":
                continue
            city = (row.get("city") or "").strip().lower()
            if not city:
                continue
            actual = _safe_float(row.get("actual_tmax"))
            if actual is None:
                continue
            out[city] = actual
    return out


def _reward_from_error(err: float, max_error_f: float) -> float:
    cap = max(1e-6, float(max_error_f))
    clipped = min(abs(float(err)), cap)
    return max(0.0, min(1.0, 1.0 - (clipped / cap)))


def _parse_feature_vector_from_row(row: dict[str, Any], trade_date: dt.date):
    raw = (row.get("feature_vector_json") or "").strip()
    if raw:
        try:
            vec = json.loads(raw)
            if isinstance(vec, list) and vec:
                return [float(x) for x in vec]
        except Exception:
            pass

    spread = _safe_float(row.get("spread_f"))
    provider_count = int(_safe_float(row.get("provider_count")) or 0)
    token = (row.get("condition_token") or "other").strip().lower()
    sky = (row.get("sky_label") or "mixed").strip().lower()
    mean_cloud = _safe_float(row.get("mean_cloud_cover"))
    entropy = _safe_float(row.get("vote_entropy")) or 0.0
    vec, _ = build_feature_vector(
        city=(row.get("city") or "").strip().lower(),
        trade_date=trade_date,
        spread_f=spread,
        provider_count=provider_count,
        condition_token=token,
        sky_label=sky,
        mean_cloud_cover=mean_cloud,
        vote_entropy=float(entropy),
    )
    return vec.tolist()


def _append_rewards_rows(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path)
    fieldnames = [
        "run_ts",
        "trade_date",
        "city",
        "decision_role",
        "bandit_mode",
        "actual_tmax",
        "selected_action",
        "applied_action",
        "mode_forecast_pred",
        "mode_blend_pred",
        "mode_lstm_pred",
        "error_forecast",
        "error_blend",
        "error_lstm",
        "reward_forecast",
        "reward_blend",
        "reward_lstm",
        "reward_chosen",
        "condition_token",
        "condition_label",
        "sky_label",
        "mean_cloud_cover",
        "vote_entropy",
        "provider_count",
        "updated_actions_json",
        "feature_vector_json",
    ]
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for row in rows:
            payload = {k: row.get(k, "") for k in fieldnames}
            w.writerow(payload)
            if db is not None:
                db.insert_bandit_reward_row(payload)  # type: ignore[attr-defined]


def _append_state_snapshot(path: str, *, run_ts: str, trade_date: str, state_path: str, state: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path)
    row = {
        "run_ts": run_ts,
        "trade_date": trade_date,
        "state_path": state_path,
        "state_json": json.dumps(state, sort_keys=True),
    }
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run_ts", "trade_date", "state_path", "state_json"])
        if write_header:
            w.writeheader()
        w.writerow(row)
    if db is not None:
        db.insert_bandit_state_snapshot_row(row)  # type: ignore[attr-defined]


if __name__ == "__main__":
    args = _parse_args()
    trade_dt = dt.datetime.strptime(args.trade_date, "%Y-%m-%d").date()

    decisions = _load_latest_decisions(args.decisions_csv, args.trade_date)
    actuals = _load_actuals(args.performance_csv, args.trade_date)
    if not decisions:
        print(f"[bandit_update] no decisions for {args.trade_date}; skipping")
        raise SystemExit(0)
    if not actuals:
        print(f"[bandit_update] no settled actuals for {args.trade_date}; skipping")
        raise SystemExit(0)

    policy, state = load_policy_state(
        args.state_path,
        alpha=float(args.alpha),
        reg_lambda=float(args.reg_lambda),
        epsilon=0.0,
    )

    run_ts = dt.datetime.now(dt.timezone.utc).isoformat()
    rows_to_append: list[dict[str, Any]] = []
    updates = 0

    for city, row in sorted(decisions.items()):
        actual = actuals.get(city)
        if actual is None:
            continue

        mode_preds = {
            "forecast": _safe_float(row.get("mode_forecast_pred")),
            "blend": _safe_float(row.get("mode_blend_pred")),
            "lstm": _safe_float(row.get("mode_lstm_pred")),
        }
        rewards: dict[str, float] = {}
        errors: dict[str, float] = {}
        for action, pred in mode_preds.items():
            if pred is None:
                continue
            err = abs(float(pred) - float(actual))
            errors[action] = float(err)
            rewards[action] = _reward_from_error(err, args.max_error_f)

        if not rewards:
            continue

        feature_vec = _parse_feature_vector_from_row(row, trade_dt)
        x = feature_vec
        for action, reward in rewards.items():
            policy.update(action, x, reward)
            updates += 1

        selected_action = (row.get("selected_action") or row.get("bandit_selected_action") or "").strip().lower()
        applied_action = (row.get("applied_action") or row.get("bandit_applied_action") or selected_action).strip().lower()
        reward_chosen = rewards.get(applied_action)
        if reward_chosen is None:
            reward_chosen = rewards.get(selected_action)
        if reward_chosen is None:
            reward_chosen = rewards.get("forecast")

        out_row = {
            "run_ts": run_ts,
            "trade_date": args.trade_date,
            "city": city,
            "decision_role": row.get("decision_role", ""),
            "bandit_mode": row.get("bandit_mode", ""),
            "actual_tmax": f"{float(actual):.4f}",
            "selected_action": selected_action,
            "applied_action": applied_action,
            "mode_forecast_pred": "" if mode_preds.get("forecast") is None else f"{mode_preds['forecast']:.4f}",
            "mode_blend_pred": "" if mode_preds.get("blend") is None else f"{mode_preds['blend']:.4f}",
            "mode_lstm_pred": "" if mode_preds.get("lstm") is None else f"{mode_preds['lstm']:.4f}",
            "error_forecast": "" if errors.get("forecast") is None else f"{errors['forecast']:.4f}",
            "error_blend": "" if errors.get("blend") is None else f"{errors['blend']:.4f}",
            "error_lstm": "" if errors.get("lstm") is None else f"{errors['lstm']:.4f}",
            "reward_forecast": "" if rewards.get("forecast") is None else f"{rewards['forecast']:.6f}",
            "reward_blend": "" if rewards.get("blend") is None else f"{rewards['blend']:.6f}",
            "reward_lstm": "" if rewards.get("lstm") is None else f"{rewards['lstm']:.6f}",
            "reward_chosen": "" if reward_chosen is None else f"{float(reward_chosen):.6f}",
            "condition_token": row.get("condition_token", ""),
            "condition_label": row.get("condition_label", ""),
            "sky_label": row.get("sky_label", ""),
            "mean_cloud_cover": row.get("mean_cloud_cover", ""),
            "vote_entropy": row.get("vote_entropy", ""),
            "provider_count": row.get("provider_count", ""),
            "updated_actions_json": json.dumps(sorted(rewards.keys())),
            "feature_vector_json": json.dumps([round(float(v), 8) for v in x]),
        }
        rows_to_append.append(out_row)

    md = dict(state.get("metadata") or {})
    md["updates"] = int(md.get("updates") or 0) + updates
    md["last_trade_date"] = args.trade_date
    md["last_update_run_ts"] = run_ts
    state["metadata"] = md

    save_policy_state(args.state_path, policy, state)
    _append_rewards_rows(args.rewards_csv, rows_to_append)
    _append_state_snapshot(
        args.state_snapshots_csv,
        run_ts=run_ts,
        trade_date=args.trade_date,
        state_path=args.state_path,
        state=state,
    )

    print(
        f"[bandit_update] trade_date={args.trade_date} cities_updated={len(rows_to_append)} "
        f"policy_updates={updates} state={args.state_path}"
    )
