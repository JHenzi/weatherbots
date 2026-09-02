"""
Adaptive ensemble weighting for per-city forecast consensus.

Replaces the flat-mean `1/MAE^2` scheme in `prediction_mae.get_rolling_mae_per_city_source`
+ `intraday_pulse` with an EWMA-based scorer that is robust to the failure modes actually
observed in `Data/source_performance.csv`:

  1. Decision-time alignment. Scores are computed from the provider snapshot taken near
     the hour the bot actually trades, not the last snapshot of the day. Grading the 23:00
     snapshot leaks the realized outcome into the label and inverts the provider ranking
     (measured: weather.gov 2.29F at 09:00 vs 15.26F at 23:00).
  2. Winsorization. A single stale/broken reading (observed max 81.6F error) no longer
     dominates a short window.
  3. Shrinkage toward a per-city prior. Providers with few observations are pulled toward
     the pooled MAE instead of earning extreme weight from a lucky sample.
  4. Availability penalty. A provider that only answers occasionally is discounted.
  5. Weight cap. No single provider can take more than `max_weight` of the ensemble, so a
     near-zero MAE cannot collapse the consensus onto one source.

Pure stdlib. No pandas/numpy dependency so this can be imported from any entrypoint.
"""

from __future__ import annotations

import csv
import datetime as dt
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

# Providers that are genuine external forecasts. `consensus` is our own output and `lstm`
# is a model, not a provider; including either creates a self-referential feedback loop.
PROVIDERS: tuple[str, ...] = (
    "google-weather",
    "open-meteo",
    "openweathermap",
    "pirateweather",
    "tomorrow",
    "visual-crossing",
    "weather.gov",
    "weatherapi",
)

EXCLUDED_SOURCES: frozenset[str] = frozenset({"consensus", "lstm"})

# Defaults chosen to match the observed error scale of this system (per-city consensus MAE
# is ~0.7-0.9F, individual providers ~2.3-3.7F at decision time).
DEFAULT_HALF_LIFE_DAYS = 7.0
DEFAULT_WINSOR_CAP_F = 8.0
DEFAULT_PRIOR_MAE_F = 3.0
DEFAULT_PRIOR_STRENGTH = 3.0
DEFAULT_MAX_WEIGHT = 0.40
DEFAULT_MIN_AVAILABILITY = 0.25
DEFAULT_MAE_FLOOR_F = 0.25


@dataclass
class ProviderStats:
    """Rolling, exponentially-weighted skill summary for one (city, provider)."""

    source: str
    city: str
    ewma_mae: float = 0.0
    ewma_sq: float = 0.0
    weight_mass: float = 0.0          # sum of decay weights (effective sample size)
    n_obs: int = 0
    n_days_in_window: int = 0
    last_date: Optional[dt.date] = None
    signed_bias: float = 0.0          # EWMA of (predicted - actual); >0 means runs warm

    @property
    def availability(self) -> float:
        if self.n_days_in_window <= 0:
            return 0.0
        return min(1.0, self.n_obs / float(self.n_days_in_window))

    @property
    def residual_std(self) -> float:
        """EWMA residual standard deviation, used for inverse-variance weighting."""
        var = max(0.0, self.ewma_sq - (self.ewma_mae * self.ewma_mae))
        return math.sqrt(var)

    def shrunk_mae(self, prior_mae: float, prior_strength: float) -> float:
        """
        Pull the EWMA MAE toward a prior when the effective sample is small.

        mae* = (m * ewma_mae + k * prior) / (m + k), where m is the accumulated decay mass.
        With m >> k this is ~ewma_mae; with m -> 0 it is the prior.
        """
        m = max(0.0, float(self.weight_mass))
        k = max(0.0, float(prior_strength))
        if (m + k) <= 0:
            return float(prior_mae)
        return ((m * self.ewma_mae) + (k * float(prior_mae))) / (m + k)


@dataclass
class WeightSet:
    """Normalized provider weights for one city plus the diagnostics behind them."""

    city: str
    weights: dict[str, float] = field(default_factory=dict)
    stats: dict[str, ProviderStats] = field(default_factory=dict)
    as_of: Optional[dt.date] = None
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS
    excluded: dict[str, str] = field(default_factory=dict)   # source -> reason

    def to_json_obj(self) -> dict:
        return {
            "as_of": self.as_of.strftime("%Y-%m-%d") if self.as_of else None,
            "half_life_days": self.half_life_days,
            "weights": dict(sorted(self.weights.items())),
            "diagnostics": {
                src: {
                    "ewma_mae": round(s.ewma_mae, 4),
                    "residual_std": round(s.residual_std, 4),
                    "signed_bias": round(s.signed_bias, 4),
                    "availability": round(s.availability, 4),
                    "n_obs": s.n_obs,
                    "effective_n": round(s.weight_mass, 3),
                }
                for src, s in sorted(self.stats.items())
            },
            "excluded": dict(sorted(self.excluded.items())),
        }


def _decay(age_days: float, half_life_days: float) -> float:
    """Exponential decay weight: 1.0 today, 0.5 one half-life ago."""
    hl = max(1e-6, float(half_life_days))
    return float(0.5 ** (max(0.0, float(age_days)) / hl))


def _winsorize(err: float, cap: float) -> float:
    return min(abs(float(err)), abs(float(cap)))


def accumulate_stats(
    observations: Iterable[tuple[dt.date, str, str, float, float]],
    *,
    as_of: dt.date,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    winsor_cap_f: float = DEFAULT_WINSOR_CAP_F,
    window_days: int = 30,
) -> dict[str, dict[str, ProviderStats]]:
    """
    Build EWMA stats from graded observations.

    `observations` yields (date, city, source, predicted_f, actual_f). Rows outside the
    window, or from excluded pseudo-sources, are ignored. Returns {city: {source: stats}}.
    """
    window_days = max(1, int(window_days))
    start = as_of - dt.timedelta(days=window_days - 1)

    out: dict[str, dict[str, ProviderStats]] = defaultdict(dict)
    days_seen: dict[str, set[dt.date]] = defaultdict(set)

    for date, city, source, predicted, actual in observations:
        if source in EXCLUDED_SOURCES:
            continue
        if date < start or date > as_of:
            continue
        city = str(city).strip().lower()
        source = str(source).strip()
        if not city or not source:
            continue

        days_seen[city].add(date)
        st = out[city].get(source)
        if st is None:
            st = ProviderStats(source=source, city=city)
            out[city][source] = st

        err = _winsorize(predicted - actual, winsor_cap_f)
        signed = max(-abs(winsor_cap_f), min(abs(winsor_cap_f), float(predicted) - float(actual)))
        w = _decay((as_of - date).days, half_life_days)

        # Streaming EWMA: running weighted mean of |err| and err^2.
        new_mass = st.weight_mass + w
        if new_mass > 0:
            st.ewma_mae = ((st.ewma_mae * st.weight_mass) + (err * w)) / new_mass
            st.ewma_sq = ((st.ewma_sq * st.weight_mass) + (err * err * w)) / new_mass
            st.signed_bias = ((st.signed_bias * st.weight_mass) + (signed * w)) / new_mass
        st.weight_mass = new_mass
        st.n_obs += 1
        if st.last_date is None or date > st.last_date:
            st.last_date = date

    for city, per_source in out.items():
        n_days = len(days_seen[city])
        for st in per_source.values():
            st.n_days_in_window = n_days
    return {c: dict(v) for c, v in out.items()}


def compute_weights(
    stats_by_source: Mapping[str, ProviderStats],
    *,
    city: str,
    as_of: Optional[dt.date] = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    prior_mae_f: Optional[float] = None,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    min_availability: float = DEFAULT_MIN_AVAILABILITY,
    mae_floor_f: float = DEFAULT_MAE_FLOOR_F,
    eligible: Optional[Sequence[str]] = None,
) -> WeightSet:
    """
    Turn per-provider EWMA stats into normalized ensemble weights.

    Score is inverse-variance on the shrunk MAE, scaled by availability, then capped so no
    single provider dominates. `eligible` restricts the output to providers that actually
    returned a value on this run.
    """
    ws = WeightSet(city=city, as_of=as_of, half_life_days=half_life_days)

    usable = {
        s: st
        for s, st in stats_by_source.items()
        if s not in EXCLUDED_SOURCES and st.weight_mass > 0
    }
    if eligible is not None:
        allow = set(eligible)
        for s in list(usable):
            if s not in allow:
                ws.excluded[s] = "not_available_this_run"
                usable.pop(s)

    if not usable:
        return ws

    # Pooled MAE across providers is the natural prior for a small sample.
    if prior_mae_f is None:
        vals = [st.ewma_mae for st in usable.values() if st.ewma_mae > 0]
        prior_mae_f = (sum(vals) / len(vals)) if vals else DEFAULT_PRIOR_MAE_F

    raw: dict[str, float] = {}
    for source, st in usable.items():
        if st.availability < min_availability:
            ws.excluded[source] = f"availability={st.availability:.2f}<{min_availability:.2f}"
            continue
        mae = max(float(mae_floor_f), st.shrunk_mae(prior_mae_f, prior_strength))
        score = 1.0 / (mae * mae)
        # Flaky providers are discounted linearly by how often they actually answer.
        score *= st.availability
        raw[source] = score
        ws.stats[source] = st

    if not raw:
        return ws

    ws.weights = _normalize_with_cap(raw, max_weight=max_weight)
    return ws


def _normalize_with_cap(raw: Mapping[str, float], *, max_weight: float) -> dict[str, float]:
    """
    Normalize to sum 1.0 with no entry above `max_weight`.

    Excess mass from capped entries is redistributed proportionally among the uncapped
    ones, repeating until stable. This is what stops a single lucky provider (MAE -> 0,
    weight -> 1/0.0001) from becoming the entire consensus.
    """
    items = {k: max(0.0, float(v)) for k, v in raw.items()}
    total = sum(items.values())
    if total <= 0:
        n = len(items)
        return {k: 1.0 / n for k in items} if n else {}

    cap = float(max_weight)
    n = len(items)
    if cap <= 0 or cap * n <= 1.0:
        # Cap is too tight to satisfy (e.g. 2 providers, cap 0.4) -> fall back to uniform.
        return {k: 1.0 / n for k in items}

    weights = {k: v / total for k, v in items.items()}
    for _ in range(64):
        over = {k: v for k, v in weights.items() if v > cap + 1e-12}
        if not over:
            break
        under = {k: v for k, v in weights.items() if v <= cap + 1e-12}
        if not under:
            break
        spill = sum(v - cap for v in over.values())
        under_total = sum(under.values())
        for k in over:
            weights[k] = cap
        if under_total <= 0:
            share = spill / len(under)
            for k in under:
                weights[k] += share
        else:
            for k in under:
                weights[k] += spill * (weights[k] / under_total)
    s = sum(weights.values())
    return {k: v / s for k, v in weights.items()} if s > 0 else weights


# --------------------------------------------------------------------------------------
# Decision-time-aligned observation loading
# --------------------------------------------------------------------------------------

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


def load_observations_at_decision_hour(
    intraday_csv: str,
    actuals: Mapping[tuple[str, str], float],
    *,
    decision_hour: int = 9,
    max_hour_distance: int = 4,
) -> list[tuple[dt.date, str, str, float, float]]:
    """
    Yield graded observations using the provider snapshot nearest `decision_hour` local.

    This is the correction for the look-ahead bias in the current pipeline: scoring the
    last snapshot of the day rewards providers for converging on an already-realized
    temperature rather than for forecasting it at the moment the bot commits capital.

    `actuals` maps (city, 'YYYY-MM-DD') -> settled max F.
    Only same-day snapshots (timestamp date == trade_date) are considered.
    """
    if not intraday_csv or not os.path.exists(intraday_csv):
        return []

    # (city, trade_date) -> (hour_distance, timestamp, row)
    best: dict[tuple[str, str], tuple[int, str, dict]] = {}
    with open(intraday_csv, "r", newline="") as f:
        for row in csv.DictReader(f):
            city = (row.get("city") or "").strip().lower()
            trade_date = (row.get("trade_date") or "").strip()
            ts = (row.get("timestamp") or "").strip()
            if not city or not trade_date or len(ts) < 13:
                continue
            if ts[:10] != trade_date:
                continue
            try:
                hour = int(ts[11:13])
            except ValueError:
                continue
            dist = abs(hour - int(decision_hour))
            if dist > int(max_hour_distance):
                continue
            key = (city, trade_date)
            prev = best.get(key)
            # Tie-break on the earlier timestamp so the choice is deterministic.
            if prev is None or (dist, ts) < (prev[0], prev[1]):
                best[key] = (dist, ts, row)

    out: list[tuple[dt.date, str, str, float, float]] = []
    for (city, trade_date), (_dist, _ts, row) in best.items():
        actual = actuals.get((city, trade_date))
        if actual is None:
            continue
        try:
            date = dt.datetime.strptime(trade_date, "%Y-%m-%d").date()
        except ValueError:
            continue
        for source, col in PROVIDER_COLS.items():
            val = (row.get(col) or "").strip()
            if not val:
                continue
            try:
                pred = float(val)
            except ValueError:
                continue
            if pred == 0.0 or pred < -50.0 or pred > 150.0:
                continue
            out.append((date, city, source, pred, float(actual)))
    return out


def load_actuals(performance_csv: str) -> dict[tuple[str, str], float]:
    """Map (city, date) -> settled actual max F, read from the graded performance log."""
    actuals: dict[tuple[str, str], float] = {}
    if not performance_csv or not os.path.exists(performance_csv):
        return actuals
    with open(performance_csv, "r", newline="") as f:
        for row in csv.DictReader(f):
            city = (row.get("city") or "").strip().lower()
            date = (row.get("date") or "").strip()
            raw = (row.get("actual_tmax") or "").strip()
            if not city or not date or not raw:
                continue
            try:
                actuals[(city, date)] = float(raw)
            except ValueError:
                continue
    return actuals


def build_city_weights(
    *,
    intraday_csv: str = "Data/intraday_forecasts.csv",
    performance_csv: str = "Data/source_performance.csv",
    as_of: Optional[dt.date] = None,
    decision_hour: int = 9,
    window_days: int = 30,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    winsor_cap_f: float = DEFAULT_WINSOR_CAP_F,
    **weight_kwargs,
) -> dict[str, WeightSet]:
    """
    End-to-end: read decision-hour snapshots, grade them, return {city: WeightSet}.

    Drop-in replacement for `get_rolling_mae_per_city_source` at the point where weights
    are formed. Returns full WeightSet objects so callers can log the diagnostics.
    """
    if as_of is None:
        as_of = dt.date.today() - dt.timedelta(days=1)
    actuals = load_actuals(performance_csv)
    obs = load_observations_at_decision_hour(
        intraday_csv, actuals, decision_hour=decision_hour
    )
    stats = accumulate_stats(
        obs,
        as_of=as_of,
        half_life_days=half_life_days,
        winsor_cap_f=winsor_cap_f,
        window_days=window_days,
    )
    return {
        city: compute_weights(
            per_source,
            city=city,
            as_of=as_of,
            half_life_days=half_life_days,
            **weight_kwargs,
        )
        for city, per_source in stats.items()
    }


def weighted_consensus(
    values: Mapping[str, float],
    weights: Mapping[str, float],
    *,
    bias_correction_f: float = 0.0,
) -> Optional[float]:
    """Weighted mean over the providers present in both maps, plus optional bias shift."""
    usable = {s: float(v) for s, v in values.items() if s in weights and v is not None}
    if not usable:
        return None
    total_w = sum(max(0.0, float(weights[s])) for s in usable)
    if total_w <= 0:
        return sum(usable.values()) / len(usable) + float(bias_correction_f)
    mean = sum(float(weights[s]) * v for s, v in usable.items()) / total_w
    return float(mean + float(bias_correction_f))


def effective_sample_size(weights: Mapping[str, float]) -> float:
    """
    Kish effective N: 1 / sum(w^2). Near 1.0 means the consensus is really one provider.

    Use this as a trade gate -- an "8-source ensemble" with ESS 1.3 is not diversified,
    and its spread-based sigma understates true uncertainty.
    """
    ws = [max(0.0, float(v)) for v in weights.values()]
    s = sum(ws)
    if s <= 0:
        return 0.0
    probs = [w / s for w in ws]
    denom = sum(p * p for p in probs)
    return float(1.0 / denom) if denom > 0 else 0.0
