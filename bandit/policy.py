import datetime as dt
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .context import CANONICAL_TOKENS

ACTIONS = ["forecast", "blend"]
CITY_ORDER = ["ny", "il", "tx", "fl"]
SKY_ORDER = ["sunny", "mixed", "cloudy"]
TOKEN_ORDER = list(CANONICAL_TOKENS)

FEATURE_NAMES = [
    "city_ny",
    "city_il",
    "city_tx",
    "city_fl",
    "doy_sin",
    "doy_cos",
    "spread_scaled",
    "provider_count_scaled",
    "mean_cloud_cover_scaled",
    "vote_entropy",
] + [f"token_{t}" for t in TOKEN_ORDER] + [f"sky_{s}" for s in SKY_ORDER]


@dataclass
class ActionScore:
    action: str
    score: float
    exploit: float
    uncertainty: float


class LinUCBPolicy:
    def __init__(
        self,
        *,
        actions: Sequence[str] = ACTIONS,
        feature_dim: int = len(FEATURE_NAMES),
        alpha: float = 0.7,
        reg_lambda: float = 1.0,
        epsilon: float = 0.0,
    ) -> None:
        self.actions = [str(a) for a in actions]
        self.feature_dim = int(feature_dim)
        self.alpha = float(alpha)
        self.reg_lambda = max(1e-6, float(reg_lambda))
        self.epsilon = max(0.0, min(1.0, float(epsilon)))
        self.A: dict[str, np.ndarray] = {
            a: np.eye(self.feature_dim, dtype=np.float64) * self.reg_lambda for a in self.actions
        }
        self.b: dict[str, np.ndarray] = {
            a: np.zeros(self.feature_dim, dtype=np.float64) for a in self.actions
        }

    def set_epsilon(self, epsilon: float) -> None:
        self.epsilon = max(0.0, min(1.0, float(epsilon)))

    def select_action(
        self,
        x: np.ndarray,
        *,
        available_actions: Sequence[str] | None = None,
        rng: random.Random | None = None,
    ) -> tuple[str, dict[str, Any]]:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.feature_dim:
            raise ValueError(f"Feature vector dimension mismatch: got {x.shape[0]} expected {self.feature_dim}")

        if available_actions:
            pool = [a for a in available_actions if a in self.A]
        else:
            pool = list(self.actions)
        if not pool:
            raise ValueError("No available actions for LinUCB selection")

        rng = rng or random.Random()
        if self.epsilon > 0 and rng.random() < self.epsilon:
            picked = pool[int(rng.random() * len(pool)) % len(pool)]
            return picked, {"selected_via": "epsilon", "scores": []}

        scores: list[ActionScore] = []
        for action in pool:
            A = self.A[action]
            b = self.b[action]
            theta = np.linalg.solve(A, b)
            exploit = float(np.dot(theta, x))
            # x^T A^-1 x via solve (more stable than explicit inverse)
            ax = np.linalg.solve(A, x)
            unc = float(self.alpha * math.sqrt(max(0.0, float(np.dot(x, ax)))))
            scores.append(ActionScore(action=action, score=exploit + unc, exploit=exploit, uncertainty=unc))

        # Deterministic tie-break by action name.
        scores.sort(key=lambda s: (-s.score, s.action))
        picked = scores[0].action
        return picked, {
            "selected_via": "linucb",
            "scores": [
                {
                    "action": s.action,
                    "score": round(float(s.score), 8),
                    "exploit": round(float(s.exploit), 8),
                    "uncertainty": round(float(s.uncertainty), 8),
                }
                for s in scores
            ],
        }

    def update(self, action: str, x: np.ndarray, reward: float) -> None:
        if action not in self.A:
            return
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.feature_dim:
            return
        r = float(reward)
        self.A[action] = self.A[action] + np.outer(x, x)
        self.b[action] = self.b[action] + (r * x)

    def to_state(self) -> dict[str, Any]:
        return {
            "actions": list(self.actions),
            "feature_dim": self.feature_dim,
            "alpha": self.alpha,
            "reg_lambda": self.reg_lambda,
            "epsilon": self.epsilon,
            "A": {a: self.A[a].tolist() for a in self.actions},
            "b": {a: self.b[a].tolist() for a in self.actions},
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "LinUCBPolicy":
        actions = state.get("actions") or ACTIONS
        feature_dim = int(state.get("feature_dim") or len(FEATURE_NAMES))
        policy = cls(
            actions=actions,
            feature_dim=feature_dim,
            alpha=float(state.get("alpha") or 0.7),
            reg_lambda=float(state.get("reg_lambda") or 1.0),
            epsilon=float(state.get("epsilon") or 0.0),
        )
        A_raw = state.get("A") or {}
        b_raw = state.get("b") or {}
        for action in policy.actions:
            try:
                A = np.asarray(A_raw.get(action), dtype=np.float64)
                b = np.asarray(b_raw.get(action), dtype=np.float64)
                if A.shape == (feature_dim, feature_dim):
                    policy.A[action] = A
                if b.shape == (feature_dim,):
                    policy.b[action] = b
            except Exception:
                continue
        return policy


def build_feature_vector(
    *,
    city: str,
    trade_date: dt.date,
    spread_f: float | None,
    provider_count: int,
    condition_token: str,
    sky_label: str,
    mean_cloud_cover: float | None,
    vote_entropy: float,
) -> tuple[np.ndarray, dict[str, float]]:
    city = str(city or "").strip().lower()
    token = str(condition_token or "other").strip().lower()
    if token not in TOKEN_ORDER:
        token = "other"
    sky = str(sky_label or "mixed").strip().lower()
    if sky not in SKY_ORDER:
        sky = "mixed"

    doy = int(trade_date.timetuple().tm_yday)
    doy_sin = math.sin((2.0 * math.pi * doy) / 366.0)
    doy_cos = math.cos((2.0 * math.pi * doy) / 366.0)

    spread_scaled = 0.0 if spread_f is None else max(0.0, min(10.0, float(spread_f))) / 10.0
    provider_count_scaled = max(0.0, min(8.0, float(provider_count))) / 8.0
    mean_cloud_cover_scaled = (
        0.5
        if mean_cloud_cover is None
        else max(0.0, min(100.0, float(mean_cloud_cover))) / 100.0
    )
    entropy = max(0.0, min(1.0, float(vote_entropy)))

    feats: dict[str, float] = {name: 0.0 for name in FEATURE_NAMES}
    feats[f"city_{city}"] = 1.0 if city in CITY_ORDER else 0.0
    feats["doy_sin"] = float(doy_sin)
    feats["doy_cos"] = float(doy_cos)
    feats["spread_scaled"] = float(spread_scaled)
    feats["provider_count_scaled"] = float(provider_count_scaled)
    feats["mean_cloud_cover_scaled"] = float(mean_cloud_cover_scaled)
    feats["vote_entropy"] = float(entropy)
    feats[f"token_{token}"] = 1.0
    feats[f"sky_{sky}"] = 1.0

    vec = np.asarray([feats[n] for n in FEATURE_NAMES], dtype=np.float64)
    return vec, feats


def load_policy_state(
    path: str,
    *,
    alpha: float,
    reg_lambda: float,
    epsilon: float,
) -> tuple[LinUCBPolicy, dict[str, Any]]:
    if path and os.path.exists(path):
        try:
            with open(path, "r") as f:
                payload = json.load(f) or {}
            policy = LinUCBPolicy.from_state(payload.get("policy") or payload)
            policy.alpha = float(alpha)
            policy.reg_lambda = max(1e-6, float(reg_lambda))
            policy.set_epsilon(float(epsilon))
            state = payload if "policy" in payload else {"policy": policy.to_state()}
            if "metadata" not in state:
                state["metadata"] = {}
            return policy, state
        except Exception:
            pass

    policy = LinUCBPolicy(alpha=alpha, reg_lambda=reg_lambda, epsilon=epsilon)
    state = {
        "version": 1,
        "feature_names": list(FEATURE_NAMES),
        "policy": policy.to_state(),
        "metadata": {
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "updates": 0,
        },
    }
    return policy, state


def save_policy_state(path: str, policy: LinUCBPolicy, state: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = dict(state)
    out["feature_names"] = list(FEATURE_NAMES)
    out["policy"] = policy.to_state()
    md = dict(out.get("metadata") or {})
    md["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    out["metadata"] = md

    # Use a per-writer temp file to avoid collisions when concurrent jobs save state.
    # os.replace keeps the final write atomic.
    parent = os.path.dirname(path) or "."
    base = os.path.basename(path)
    fd, tmp = tempfile.mkstemp(prefix=f".{base}.", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(out, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
