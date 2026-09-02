"""
Dynamic entry policy: EV-first gating, adaptive confidence floor, and edge-scaled sizing.

Replaces the fixed `effective_confidence < --min-confidence` veto in `kalshi_trader.py`.

Three structural changes over the current logic:

  1. Order of operations. Today the confidence gate runs *before* any market lookup, so a
     0.60-confidence signal against a 40c mispricing is discarded before its edge is ever
     computed, while a 0.80-confidence signal against a 2c mispricing passes. Here the
     market is priced first and confidence modulates *size*, not permission. A hard floor
     remains, but it sits far below the current 0.75 and is only a junk filter.

  2. The floor is a function, not a constant. Required confidence rises with the bid/ask
     spread (execution cost), with model sigma (our own uncertainty), and falls as
     settlement approaches (less time for the forecast to drift).

  3. Probabilities are recalibrated before use. Measured on 271 settled rows, the raw
     `model_prob_yes` is non-monotonic against realized hit rate (the 0.0-0.1 bin hits 72%
     of the time; the 0.4-0.5 bin hits 11%), while `market_prob_yes` is close to monotonic.
     Feeding the raw model probability into an EV calculation therefore produces edge
     estimates that are mostly miscalibration. `CalibrationMap` (see feedback_loop.py)
     corrects this before any EV is taken.

Pure stdlib.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional


# --------------------------------------------------------------------------------------
# Tunables. Every field here is intended to be machine-updated by feedback_loop.py rather
# than hand-edited, which is why they live in a dataclass rather than argparse defaults.
# --------------------------------------------------------------------------------------

@dataclass
class PolicyParams:
    # Hard floor: below this, the signal is noise regardless of price. Deliberately far
    # below the current 0.75 constant because confidence now scales size instead of vetoing.
    min_confidence_floor: float = 0.30

    # Confidence floor scaling.
    spread_penalty_per_cent: float = 0.010   # +0.010 required conf per cent of bid/ask spread
    sigma_penalty_per_degree: float = 0.040  # +0.040 required conf per degree F of model sigma
    max_dynamic_floor: float = 0.72          # never demand more than this

    # Time decay: as settlement nears, the forecast can drift less, so we relax the floor.
    # multiplier = 1.0 at open, -> (1 - late_relaxation) at settlement.
    late_relaxation: float = 0.25

    # EV gating.
    min_ev_cents: float = 3.0                # required expected value per contract
    min_edge_prob: float = 0.05              # required calibrated probability edge
    min_yes_ask: int = 2                     # avoid 1c lottery tickets / stale books
    max_yes_ask: int = 90                    # avoid paying up for near-certainties

    # Sizing.
    kelly_fraction: float = 0.25             # fraction of full Kelly
    max_position_dollars: float = 10.0
    min_position_dollars: float = 1.0

    # Diversification gate: an "8-source ensemble" whose Kish ESS is ~1 is one source.
    min_effective_sources: float = 2.0

    # Liquidity.
    max_fraction_of_depth: float = 0.50      # never take more than half the displayed ask depth


@dataclass
class Decision:
    action: str                              # "trade" | "skip"
    reason: str = ""
    contracts: int = 0
    dollars: float = 0.0
    calibrated_prob: Optional[float] = None
    edge_prob: Optional[float] = None
    ev_cents: Optional[float] = None
    required_confidence: Optional[float] = None
    effective_confidence: Optional[float] = None
    diagnostics: dict = field(default_factory=dict)

    def as_log_row(self) -> dict:
        return {
            "decision": self.action,
            "reason": self.reason,
            "contracts": self.contracts,
            "dollars": f"{self.dollars:.2f}",
            "calibrated_prob": "" if self.calibrated_prob is None else f"{self.calibrated_prob:.6f}",
            "edge_prob": "" if self.edge_prob is None else f"{self.edge_prob:.6f}",
            "ev_cents": "" if self.ev_cents is None else f"{self.ev_cents:.4f}",
            "required_confidence": "" if self.required_confidence is None else f"{self.required_confidence:.4f}",
            "effective_confidence": "" if self.effective_confidence is None else f"{self.effective_confidence:.4f}",
        }


def required_confidence(
    *,
    yes_spread_cents: float,
    sigma_f: Optional[float],
    hours_to_settlement: Optional[float],
    params: PolicyParams,
) -> float:
    """
    Continuous minimum-confidence surface.

    Rises with execution cost (bid/ask spread) and model uncertainty (sigma); falls as
    settlement approaches. Clamped to [min_confidence_floor, max_dynamic_floor].
    """
    floor = float(params.min_confidence_floor)
    req = floor
    req += float(params.spread_penalty_per_cent) * max(0.0, float(yes_spread_cents))
    if sigma_f is not None:
        req += float(params.sigma_penalty_per_degree) * max(0.0, float(sigma_f))

    if hours_to_settlement is not None:
        # Normalize over a 24h trading day; 1.0 at open, 0.0 at settlement.
        frac_remaining = max(0.0, min(1.0, float(hours_to_settlement) / 24.0))
        relax = float(params.late_relaxation) * (1.0 - frac_remaining)
        req *= (1.0 - relax)

    return float(max(floor, min(float(params.max_dynamic_floor), req)))


def kelly_contracts(
    *,
    calibrated_prob: float,
    yes_ask_cents: float,
    bankroll_dollars: float,
    params: PolicyParams,
    ask_depth: Optional[int] = None,
) -> tuple[int, float]:
    """
    Fractional-Kelly sizing for a binary contract.

    A Kalshi YES contract bought at `a` cents pays 100c on a win. Net odds b = (100-a)/a.
    Full Kelly f* = (p*b - (1-p)) / b. We take `kelly_fraction` of that and clamp to the
    per-position cap and to a fraction of displayed depth.

    Returns (contracts, dollars).
    """
    a = float(yes_ask_cents)
    if a <= 0 or a >= 100:
        return (0, 0.0)
    p = max(0.0, min(1.0, float(calibrated_prob)))
    b = (100.0 - a) / a
    if b <= 0:
        return (0, 0.0)

    f_star = ((p * b) - (1.0 - p)) / b
    if f_star <= 0:
        return (0, 0.0)

    stake = float(bankroll_dollars) * f_star * float(params.kelly_fraction)
    stake = min(stake, float(params.max_position_dollars))
    if stake < float(params.min_position_dollars):
        return (0, 0.0)

    cost_per_contract = a / 100.0
    contracts = int(stake / cost_per_contract)
    if ask_depth is not None and ask_depth > 0:
        contracts = min(contracts, int(ask_depth * float(params.max_fraction_of_depth)))
    if contracts <= 0:
        return (0, 0.0)
    return (contracts, contracts * cost_per_contract)


def evaluate(
    *,
    model_prob_yes: float,
    market_prob_yes: Optional[float],
    yes_ask_cents: float,
    yes_bid_cents: Optional[float],
    effective_confidence: Optional[float],
    sigma_f: Optional[float],
    hours_to_settlement: Optional[float],
    bankroll_dollars: float,
    params: PolicyParams,
    calibrator: Optional[Callable[[float], float]] = None,
    ask_depth: Optional[int] = None,
    effective_sources: Optional[float] = None,
) -> Decision:
    """
    Decide whether and how much to trade one bucket.

    Gate order is deliberate: cheap structural checks, then price, then EV, then the
    dynamic confidence floor. Confidence is the last gate rather than the first, so a
    large mispricing is never discarded before its edge is computed.
    """
    diag: dict = {}

    # --- Structural gates (no market data needed) -------------------------------------
    if effective_sources is not None and effective_sources < params.min_effective_sources:
        return Decision(
            action="skip",
            reason=f"undiversified_ensemble;ess={effective_sources:.2f}<{params.min_effective_sources}",
            diagnostics=diag,
        )

    # --- Recalibrate the model probability before it touches any EV math ---------------
    raw_p = max(0.0, min(1.0, float(model_prob_yes)))
    p = float(calibrator(raw_p)) if calibrator is not None else raw_p
    p = max(0.0, min(1.0, p))
    diag["raw_model_prob"] = raw_p
    diag["calibrated_prob"] = p

    # --- Price gates -------------------------------------------------------------------
    ask = float(yes_ask_cents)
    if ask < float(params.min_yes_ask):
        return Decision(
            action="skip",
            reason=f"ask_below_floor;yes_ask={ask:.0f}<min={params.min_yes_ask}",
            calibrated_prob=p,
            diagnostics=diag,
        )
    if ask > float(params.max_yes_ask):
        return Decision(
            action="skip",
            reason=f"ask_above_cap;yes_ask={ask:.0f}>max={params.max_yes_ask}",
            calibrated_prob=p,
            diagnostics=diag,
        )

    spread_cents = 0.0
    if yes_bid_cents is not None:
        spread_cents = max(0.0, ask - float(yes_bid_cents))
    diag["yes_spread_cents"] = spread_cents

    # --- Edge and EV, computed on the calibrated probability ---------------------------
    implied = float(market_prob_yes) if market_prob_yes is not None else (ask / 100.0)
    edge = p - implied
    # EV net of the spread we cross to get filled.
    ev_cents = (100.0 * p) - ask
    diag["implied_prob"] = implied

    if edge < float(params.min_edge_prob):
        return Decision(
            action="skip",
            reason=f"edge_too_small;edge={edge:.3f}<{params.min_edge_prob}",
            calibrated_prob=p, edge_prob=edge, ev_cents=ev_cents, diagnostics=diag,
        )
    if ev_cents < float(params.min_ev_cents):
        return Decision(
            action="skip",
            reason=f"ev_too_small;ev={ev_cents:.2f}c<{params.min_ev_cents}c",
            calibrated_prob=p, edge_prob=edge, ev_cents=ev_cents, diagnostics=diag,
        )

    # --- Dynamic confidence floor (last gate, not first) --------------------------------
    req = required_confidence(
        yes_spread_cents=spread_cents,
        sigma_f=sigma_f,
        hours_to_settlement=hours_to_settlement,
        params=params,
    )
    eff = None if effective_confidence is None else max(0.0, min(1.0, float(effective_confidence)))
    if eff is not None and eff < req:
        return Decision(
            action="skip",
            reason=f"confidence_below_dynamic_floor;eff={eff:.3f}<req={req:.3f}",
            calibrated_prob=p, edge_prob=edge, ev_cents=ev_cents,
            required_confidence=req, effective_confidence=eff, diagnostics=diag,
        )

    # --- Size: scale the Kelly stake by how far confidence clears the floor -------------
    conf_scale = 1.0
    if eff is not None:
        headroom = (eff - req) / max(1e-6, 1.0 - req)
        conf_scale = max(0.25, min(1.0, 0.5 + 0.5 * headroom))
    diag["confidence_scale"] = conf_scale

    contracts, dollars = kelly_contracts(
        calibrated_prob=p,
        yes_ask_cents=ask,
        bankroll_dollars=float(bankroll_dollars) * conf_scale,
        params=params,
        ask_depth=ask_depth,
    )
    if contracts <= 0:
        return Decision(
            action="skip",
            reason="size_rounds_to_zero",
            calibrated_prob=p, edge_prob=edge, ev_cents=ev_cents,
            required_confidence=req, effective_confidence=eff, diagnostics=diag,
        )

    return Decision(
        action="trade",
        reason="",
        contracts=contracts,
        dollars=dollars,
        calibrated_prob=p,
        edge_prob=edge,
        ev_cents=ev_cents,
        required_confidence=req,
        effective_confidence=eff,
        diagnostics=diag,
    )


def bucket_probability(
    *,
    mu_f: float,
    sigma_f: float,
    bucket_lo: Optional[float],
    bucket_hi: Optional[float],
    sigma_floor_f: float = 0.75,
) -> float:
    """
    P(bucket_lo <= T < bucket_hi) under a Normal(mu, sigma) forecast.

    `sigma_floor_f` guards against the degenerate sigma=0 case seen in the current logs
    (spread_f=0.0000 when the reliable-provider set collapses to one member), which would
    otherwise produce probabilities of exactly 0 or 1 and unbounded apparent edge.
    """
    s = max(float(sigma_floor_f), float(sigma_f))
    lo_p = 0.0 if bucket_lo is None else _norm_cdf(float(bucket_lo), mu_f, s)
    hi_p = 1.0 if bucket_hi is None else _norm_cdf(float(bucket_hi), mu_f, s)
    return float(max(0.0, min(1.0, hi_p - lo_p)))


def _norm_cdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    return 0.5 * (1.0 + math.erf((float(x) - float(mu)) / (float(sigma) * math.sqrt(2.0))))
