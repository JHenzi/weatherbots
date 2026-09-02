"""
Post-trade feedback loop: fit a probability calibrator and tune policy parameters
from settled outcomes.

This is the component the pipeline is currently missing. `settle_eval.py` already
backfills `settlement_tmax_f`, `bucket_hit` and `realized_pnl_*` into
`Data/eval_history.csv`, and `calibrate_sources.py` already closes the loop on *source
weights* -- but nothing reads realized outcomes back into the *trading policy*.
`min_confidence`, `min_yes_ask` and the spread caps are read from argparse and never
updated, so the decision engine cannot learn.

Two jobs:

  1. `fit_calibration` -- isotonic (pool-adjacent-violators) regression of realized
     bucket-hit rate on `model_prob_yes`. Measured over 271 settled rows the raw model
     probability is non-monotonic (the 0.0-0.1 bin hit 72%, the 0.4-0.5 bin hit 11%),
     so every downstream EV number is currently dominated by miscalibration rather than
     by genuine edge. The fitted map is persisted and loaded by `decision_policy`.

  2. `tune_params` -- a conservative, bounded coordinate search over PolicyParams that
     maximizes realized PnL on the settled history, with guardrails so a single good or
     bad week cannot move a parameter far. Writes a versioned JSON so every change is
     auditable and revertible.

Both are offline batch jobs: run after settlement, alongside `calibrate_sources.py`.
Pure stdlib.

CLI:
    python feedback_loop.py --fit-calibration
    python feedback_loop.py --tune --dry-run
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
from dataclasses import asdict
from typing import Iterable, Optional, Sequence

from decision_policy import PolicyParams, evaluate

DEFAULT_EVAL_CSV = "Data/eval_history.csv"
DEFAULT_CALIBRATION_JSON = "Data/probability_calibration.json"
DEFAULT_PARAMS_JSON = "Data/policy_params.json"
DEFAULT_TUNING_LOG = "Data/policy_tuning_history.csv"

MIN_ROWS_FOR_CALIBRATION = 60
MIN_ROWS_FOR_TUNING = 120


# --------------------------------------------------------------------------------------
# Settled-outcome loading
# --------------------------------------------------------------------------------------

def _f(row: dict, key: str) -> Optional[float]:
    raw = (row.get(key) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _truthy(value) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes", "y", "t")


def load_settled(
    eval_csv: str = DEFAULT_EVAL_CSV,
    *,
    since: Optional[dt.date] = None,
) -> list[dict]:
    """Rows from eval_history.csv that have a settled outcome."""
    if not eval_csv or not os.path.exists(eval_csv):
        return []
    out: list[dict] = []
    with open(eval_csv, "r", newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("bucket_hit") or "").strip() == "":
                continue
            td = (row.get("trade_date") or "").strip()
            if since is not None:
                try:
                    if dt.datetime.strptime(td, "%Y-%m-%d").date() < since:
                        continue
                except ValueError:
                    continue
            out.append(row)
    return out


# --------------------------------------------------------------------------------------
# 1. Probability calibration (isotonic regression via PAVA)
# --------------------------------------------------------------------------------------

class CalibrationMap:
    """
    Monotone piecewise-linear map from raw model probability to calibrated probability.

    Fitted by pool-adjacent-violators, which is the right tool here: it makes no
    parametric assumption (unlike Platt scaling) and enforces monotonicity, which is
    exactly the property the raw probabilities currently violate.
    """

    def __init__(self, xs: Sequence[float], ys: Sequence[float], *, n_obs: int = 0):
        self.xs = [float(x) for x in xs]
        self.ys = [float(y) for y in ys]
        self.n_obs = int(n_obs)

    def __call__(self, p: float) -> float:
        return self.predict(p)

    def predict(self, p: float) -> float:
        if not self.xs:
            return float(max(0.0, min(1.0, p)))
        x = float(max(0.0, min(1.0, p)))
        if x <= self.xs[0]:
            return self.ys[0]
        if x >= self.xs[-1]:
            return self.ys[-1]
        for i in range(1, len(self.xs)):
            if x <= self.xs[i]:
                x0, x1 = self.xs[i - 1], self.xs[i]
                y0, y1 = self.ys[i - 1], self.ys[i]
                if x1 <= x0:
                    return y1
                t = (x - x0) / (x1 - x0)
                return float(y0 + t * (y1 - y0))
        return self.ys[-1]

    def to_json_obj(self) -> dict:
        return {"xs": self.xs, "ys": self.ys, "n_obs": self.n_obs}

    @classmethod
    def from_json_obj(cls, obj: dict) -> "CalibrationMap":
        return cls(obj.get("xs") or [], obj.get("ys") or [], n_obs=int(obj.get("n_obs") or 0))

    @classmethod
    def identity(cls) -> "CalibrationMap":
        return cls([0.0, 1.0], [0.0, 1.0], n_obs=0)


def _pava(pairs: Sequence[tuple[float, float]]) -> tuple[list[float], list[float]]:
    """
    Pool-adjacent-violators. `pairs` is (x, y) sorted by x; y in {0,1} or a rate.

    Returns (xs, ys) of the fitted monotone step function, collapsed to block means.
    """
    xs = [p[0] for p in pairs]
    # Each block: [sum_y, count, x_right]
    blocks: list[list[float]] = []
    for x, y in pairs:
        blocks.append([float(y), 1.0, float(x)])
        while len(blocks) >= 2 and (blocks[-2][0] / blocks[-2][1]) > (blocks[-1][0] / blocks[-1][1]):
            b = blocks.pop()
            a = blocks.pop()
            blocks.append([a[0] + b[0], a[1] + b[1], b[2]])

    out_x: list[float] = []
    out_y: list[float] = []
    idx = 0
    for total, count, _x_right in blocks:
        mean_y = total / count
        n = int(count)
        # Anchor each block at the mean x of its members for a smoother interpolation.
        block_xs = xs[idx: idx + n]
        idx += n
        if not block_xs:
            continue
        out_x.append(sum(block_xs) / len(block_xs))
        out_y.append(mean_y)

    # Enforce strictly increasing x for interpolation.
    dedup_x: list[float] = []
    dedup_y: list[float] = []
    for x, y in zip(out_x, out_y):
        if dedup_x and x <= dedup_x[-1]:
            dedup_y[-1] = y
            continue
        dedup_x.append(x)
        dedup_y.append(y)
    return dedup_x, dedup_y


def fit_calibration(
    rows: Iterable[dict],
    *,
    prob_field: str = "model_prob_yes",
    smoothing: float = 0.02,
) -> CalibrationMap:
    """
    Fit an isotonic calibration map from settled rows.

    `smoothing` pulls fitted rates toward 0.5 slightly, which prevents a block of a few
    all-win or all-loss observations from mapping to exactly 1.0 or 0.0 and producing
    infinite apparent edge downstream.
    """
    pairs: list[tuple[float, float]] = []
    for row in rows:
        p = _f(row, prob_field)
        if p is None:
            continue
        pairs.append((max(0.0, min(1.0, p)), 1.0 if _truthy(row.get("bucket_hit")) else 0.0))

    if len(pairs) < MIN_ROWS_FOR_CALIBRATION:
        return CalibrationMap.identity()

    pairs.sort(key=lambda t: t[0])
    xs, ys = _pava(pairs)
    if not xs:
        return CalibrationMap.identity()

    s = max(0.0, min(0.5, float(smoothing)))
    ys = [float(min(1.0, max(0.0, (1.0 - s) * y + s * 0.5))) for y in ys]
    return CalibrationMap(xs, ys, n_obs=len(pairs))


def brier_score(rows: Iterable[dict], *, prob_field: str = "model_prob_yes",
                calibrator: Optional[CalibrationMap] = None) -> Optional[float]:
    """Mean squared error of predicted probability vs realized outcome. Lower is better."""
    total, n = 0.0, 0
    for row in rows:
        p = _f(row, prob_field)
        if p is None:
            continue
        if calibrator is not None:
            p = calibrator.predict(p)
        y = 1.0 if _truthy(row.get("bucket_hit")) else 0.0
        total += (p - y) ** 2
        n += 1
    return (total / n) if n else None


def save_calibration(cal: CalibrationMap, path: str = DEFAULT_CALIBRATION_JSON) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "fitted_at": dt.datetime.now().astimezone().isoformat(),
        "calibration": cal.to_json_obj(),
    }
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def load_calibration(path: str = DEFAULT_CALIBRATION_JSON) -> CalibrationMap:
    """Load the persisted calibrator; identity map if absent or unreadable."""
    if not path or not os.path.exists(path):
        return CalibrationMap.identity()
    try:
        with open(path, "r") as f:
            payload = json.load(f) or {}
        return CalibrationMap.from_json_obj(payload.get("calibration") or {})
    except Exception:
        return CalibrationMap.identity()


# --------------------------------------------------------------------------------------
# 2. Bounded policy-parameter tuning
# --------------------------------------------------------------------------------------

# (attribute, candidate values). Kept deliberately coarse and bounded: this is a nightly
# nudge, not a global optimizer, and an unbounded search on ~270 rows would overfit.
TUNING_GRID: dict[str, list[float]] = {
    "min_ev_cents": [1.0, 2.0, 3.0, 5.0, 8.0],
    "min_edge_prob": [0.02, 0.05, 0.08, 0.12],
    "min_confidence_floor": [0.20, 0.30, 0.40, 0.50],
    "sigma_penalty_per_degree": [0.0, 0.02, 0.04, 0.08],
    "kelly_fraction": [0.10, 0.25, 0.40],
}

# Max fractional move per run, so no parameter can jump on one week of noise.
MAX_RELATIVE_STEP = 0.35

# Minimum |t| on mean per-trade PnL before we allow any parameter to move. Measured on the
# current 271 settled rows the realized edge is +$0.32/trade with se $0.41 -> t = 0.79,
# i.e. indistinguishable from zero (bootstrap 95% CI on total PnL: -$110 to +$323). Tuning
# against a signal that weak fits noise and will happily "learn" a losing configuration, so
# the tuner refuses to act until the history can actually distinguish it from chance.
MIN_TSTAT_TO_TUNE = 1.5


def pnl_significance(rows: Sequence[dict]) -> dict:
    """
    Is realized PnL distinguishable from zero?

    Returns {n, total, mean, se, tstat}. Used as a gate on parameter tuning: with a
    t-stat below `MIN_TSTAT_TO_TUNE` any 'improvement' found by the search is noise.
    """
    pnl: list[float] = []
    for row in rows:
        c = _f(row, "count")
        ask = _f(row, "yes_ask")
        if not c or ask is None:
            continue
        won = _truthy(row.get("bucket_hit"))
        pnl.append(c * (1.0 if won else 0.0) - c * (ask / 100.0))

    n = len(pnl)
    if n < 2:
        return {"n": n, "total": 0.0, "mean": 0.0, "se": 0.0, "tstat": 0.0}
    total = sum(pnl)
    mean = total / n
    var = sum((x - mean) ** 2 for x in pnl) / (n - 1)
    se = math.sqrt(var / n) if var > 0 else 0.0
    tstat = (mean / se) if se > 0 else 0.0
    return {"n": n, "total": total, "mean": mean, "se": se, "tstat": tstat}


def replay_pnl(rows: Sequence[dict], params: PolicyParams,
               calibrator: Optional[CalibrationMap] = None,
               *, bankroll_dollars: float = 50.0) -> tuple[float, int]:
    """
    Replay settled history under a candidate parameter set.

    Returns (total_pnl_dollars, n_trades). This is a counterfactual on *sizing and
    gating only*: it reuses the bucket that was actually chosen and the price actually
    quoted, so it cannot model trades the bot never looked at. It is therefore a
    conservative estimate, useful for ranking parameter sets against each other rather
    than for forecasting absolute returns.
    """
    total, n = 0.0, 0
    for row in rows:
        p_model = _f(row, "model_prob_yes")
        ask = _f(row, "yes_ask")
        if p_model is None or ask is None:
            continue
        d = evaluate(
            model_prob_yes=p_model,
            market_prob_yes=_f(row, "market_prob_yes"),
            yes_ask_cents=ask,
            yes_bid_cents=_f(row, "yes_bid"),
            effective_confidence=_f(row, "confidence_score"),
            sigma_f=_f(row, "sigma_f"),
            hours_to_settlement=None,
            bankroll_dollars=bankroll_dollars,
            params=params,
            calibrator=(calibrator.predict if calibrator else None),
            ask_depth=int(_f(row, "ask_qty") or 0) or None,
        )
        if d.action != "trade" or d.contracts <= 0:
            continue
        won = _truthy(row.get("bucket_hit"))
        cost = d.contracts * (ask / 100.0)
        payout = d.contracts * 1.0 if won else 0.0
        total += (payout - cost)
        n += 1
    return (total, n)


def tune_params(
    rows: Sequence[dict],
    current: PolicyParams,
    calibrator: Optional[CalibrationMap] = None,
    *,
    bankroll_dollars: float = 50.0,
) -> tuple[PolicyParams, dict]:
    """
    One pass of bounded coordinate ascent on realized PnL.

    Each parameter is moved at most `MAX_RELATIVE_STEP` from its current value, and only
    if the candidate strictly improves replayed PnL. Returns (new_params, report).
    """
    if len(rows) < MIN_ROWS_FOR_TUNING:
        return (current, {"status": "insufficient_data", "n_rows": len(rows)})

    # Refuse to tune against a PnL series that cannot be distinguished from chance.
    sig = pnl_significance(rows)
    if abs(sig["tstat"]) < MIN_TSTAT_TO_TUNE:
        return (current, {
            "status": "insignificant_edge",
            "n_rows": len(rows),
            "tstat": round(sig["tstat"], 3),
            "required_tstat": MIN_TSTAT_TO_TUNE,
            "total_pnl": round(sig["total"], 2),
            "mean_pnl_per_trade": round(sig["mean"], 4),
            "note": ("Realized edge is not statistically distinguishable from zero. "
                     "Tuning here would fit noise; collect more settled trades first."),
        })

    best = PolicyParams(**asdict(current))
    base_pnl, base_n = replay_pnl(rows, best, calibrator, bankroll_dollars=bankroll_dollars)
    report: dict = {
        "status": "ok",
        "n_rows": len(rows),
        "baseline_pnl": round(base_pnl, 2),
        "baseline_trades": base_n,
        "changes": {},
    }

    for attr, candidates in TUNING_GRID.items():
        cur_val = float(getattr(best, attr))
        best_val, best_pnl = cur_val, replay_pnl(rows, best, calibrator,
                                                 bankroll_dollars=bankroll_dollars)[0]
        for cand in candidates:
            # Bound the step so one noisy window cannot move a parameter far.
            if cur_val > 0 and abs(cand - cur_val) / abs(cur_val) > MAX_RELATIVE_STEP:
                continue
            trial = PolicyParams(**asdict(best))
            setattr(trial, attr, float(cand))
            pnl, _ = replay_pnl(rows, trial, calibrator, bankroll_dollars=bankroll_dollars)
            if pnl > best_pnl + 1e-9:
                best_pnl, best_val = pnl, float(cand)
        if best_val != cur_val:
            setattr(best, attr, best_val)
            report["changes"][attr] = {"from": cur_val, "to": best_val,
                                       "pnl": round(best_pnl, 2)}

    final_pnl, final_n = replay_pnl(rows, best, calibrator, bankroll_dollars=bankroll_dollars)
    report["tuned_pnl"] = round(final_pnl, 2)
    report["tuned_trades"] = final_n
    return (best, report)


def save_params(params: PolicyParams, path: str = DEFAULT_PARAMS_JSON) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"updated_at": dt.datetime.now().astimezone().isoformat(),
               "params": asdict(params)}
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def load_params(path: str = DEFAULT_PARAMS_JSON) -> PolicyParams:
    """Load tuned params; fall back to code defaults if absent or malformed."""
    if not path or not os.path.exists(path):
        return PolicyParams()
    try:
        with open(path, "r") as f:
            payload = json.load(f) or {}
        raw = payload.get("params") or {}
        base = PolicyParams()
        for k, v in raw.items():
            if hasattr(base, k):
                setattr(base, k, type(getattr(base, k))(v))
        return base
    except Exception:
        return PolicyParams()


def append_tuning_log(report: dict, path: str = DEFAULT_TUNING_LOG) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run_ts", "status", "n_rows", "baseline_pnl",
                                          "tuned_pnl", "changes_json"])
        if write_header:
            w.writeheader()
        w.writerow({
            "run_ts": dt.datetime.now().astimezone().isoformat(),
            "status": report.get("status", ""),
            "n_rows": report.get("n_rows", ""),
            "baseline_pnl": report.get("baseline_pnl", ""),
            "tuned_pnl": report.get("tuned_pnl", ""),
            "changes_json": json.dumps(report.get("changes", {}), sort_keys=True),
        })


def main() -> None:
    p = argparse.ArgumentParser(description="Fit probability calibration and tune policy params from settled outcomes.")
    p.add_argument("--eval-csv", default=DEFAULT_EVAL_CSV)
    p.add_argument("--calibration-json", default=DEFAULT_CALIBRATION_JSON)
    p.add_argument("--params-json", default=DEFAULT_PARAMS_JSON)
    p.add_argument("--since", default="", help="YYYY-MM-DD; only use settled rows on/after this date")
    p.add_argument("--fit-calibration", action="store_true")
    p.add_argument("--tune", action="store_true")
    p.add_argument("--bankroll", type=float, default=50.0)
    p.add_argument("--dry-run", action="store_true", help="Report only; write nothing.")
    args = p.parse_args()

    since = None
    if args.since.strip():
        since = dt.datetime.strptime(args.since.strip(), "%Y-%m-%d").date()

    rows = load_settled(args.eval_csv, since=since)
    print(f"[feedback_loop] settled rows: {len(rows)}")
    if not rows:
        return

    if args.fit_calibration or not args.tune:
        cal = fit_calibration(rows)
        raw_brier = brier_score(rows)
        cal_brier = brier_score(rows, calibrator=cal)
        print(f"[feedback_loop] calibration fitted on n={cal.n_obs} knots={len(cal.xs)}")
        if raw_brier is not None and cal_brier is not None:
            print(f"[feedback_loop] Brier raw={raw_brier:.4f} calibrated={cal_brier:.4f} "
                  f"({100.0 * (raw_brier - cal_brier) / raw_brier:+.1f}%)")
        if not args.dry_run:
            save_calibration(cal, args.calibration_json)
            print(f"[feedback_loop] wrote {args.calibration_json}")

    if args.tune:
        cal = load_calibration(args.calibration_json)
        current = load_params(args.params_json)
        tuned, report = tune_params(rows, current, cal, bankroll_dollars=args.bankroll)
        print(json.dumps(report, indent=2, sort_keys=True))
        if not args.dry_run and report.get("status") == "ok":
            save_params(tuned, args.params_json)
            append_tuning_log(report)
            print(f"[feedback_loop] wrote {args.params_json}")


if __name__ == "__main__":
    main()
