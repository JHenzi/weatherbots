import json
import math
from typing import Any, Mapping

CANONICAL_TOKENS = [
    "clear",
    "partly_cloudy",
    "cloudy_overcast",
    "rain",
    "snow",
    "storm",
    "fog",
    "wind",
    "other",
]

TOKEN_TO_CLOUD_PROXY = {
    "clear": 10.0,
    "partly_cloudy": 50.0,
    "cloudy_overcast": 85.0,
    "rain": 85.0,
    "snow": 90.0,
    "storm": 95.0,
    "fog": 90.0,
    "wind": 55.0,
    "other": 50.0,
}


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


def normalize_condition_text(text: str | None, icon: str | None = None) -> str:
    merged = " ".join([str(text or ""), str(icon or "")]).strip().lower()
    if not merged:
        return "other"

    if any(k in merged for k in ("thunder", "storm", "squall", "tornado", "hail")):
        return "storm"
    if any(k in merged for k in ("snow", "sleet", "ice", "freezing", "blizzard", "flurr")):
        return "snow"
    if any(k in merged for k in ("rain", "drizzle", "shower", "downpour")):
        return "rain"
    if any(k in merged for k in ("fog", "mist", "haze", "smoke")):
        return "fog"

    if "overcast" in merged or "cloudy" in merged:
        if any(k in merged for k in ("partly", "mostly clear", "few clouds", "broken clouds")):
            return "partly_cloudy"
        return "cloudy_overcast"

    if any(k in merged for k in ("clear", "sun", "fair")):
        if any(k in merged for k in ("partly", "mostly cloudy")):
            return "partly_cloudy"
        return "clear"

    if any(k in merged for k in ("wind", "breez", "gust")):
        return "wind"

    return "other"


def sky_label_from_cloud_cover(cloud_cover: float | None, token: str | None = None) -> str:
    if cloud_cover is not None:
        cc = max(0.0, min(100.0, float(cloud_cover)))
        if cc <= 35.0:
            return "sunny"
        if cc >= 65.0:
            return "cloudy"
        return "mixed"

    if token == "clear":
        return "sunny"
    if token in ("cloudy_overcast", "rain", "snow", "storm", "fog"):
        return "cloudy"
    return "mixed"


def _normalized_entropy(weighted_votes: Mapping[str, float]) -> float:
    total = sum(max(0.0, float(v)) for v in weighted_votes.values())
    if total <= 0:
        return 0.0
    probs = [max(0.0, float(v)) / total for v in weighted_votes.values() if float(v) > 0.0]
    if len(probs) <= 1:
        return 0.0
    h = -sum(p * math.log(p) for p in probs)
    h_max = math.log(len(probs))
    if h_max <= 0:
        return 0.0
    return max(0.0, min(1.0, h / h_max))


def vote_provider_conditions(
    provider_payloads: Mapping[str, Mapping[str, Any]],
    provider_weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """
    Weighted vote over provider condition payloads.

    Each provider payload can include:
      - condition_text
      - condition_icon
      - cloud_cover
    """
    normalized_rows: list[dict[str, Any]] = []
    for provider in sorted(provider_payloads.keys()):
        payload = provider_payloads.get(provider) or {}
        text = str(payload.get("condition_text") or "").strip()
        icon = str(payload.get("condition_icon") or "").strip()
        cloud_cover = _safe_float(payload.get("cloud_cover"))
        token = normalize_condition_text(text, icon)
        normalized_rows.append(
            {
                "provider": provider,
                "condition_text": text,
                "condition_icon": icon,
                "cloud_cover": cloud_cover,
                "condition_token": token,
            }
        )

    if not normalized_rows:
        return {
            "condition_token": "other",
            "condition_label": "",
            "sky_label": "mixed",
            "mean_cloud_cover": None,
            "vote_entropy": 0.0,
            "provider_count": 0,
            "raw_provider_labels_json": "{}",
            "token_weights_json": "{}",
        }

    weight_map: dict[str, float] = {}
    if provider_weights:
        for src in provider_payloads.keys():
            try:
                weight_map[src] = max(0.0, float(provider_weights.get(src, 0.0)))
            except Exception:
                weight_map[src] = 0.0

    total_weight = sum(weight_map.values())
    if total_weight <= 0:
        u = 1.0 / float(len(normalized_rows))
        for row in normalized_rows:
            row["vote_weight"] = u
    else:
        for row in normalized_rows:
            row["vote_weight"] = weight_map.get(row["provider"], 0.0) / total_weight

    token_weights: dict[str, float] = {t: 0.0 for t in CANONICAL_TOKENS}
    for row in normalized_rows:
        token = str(row.get("condition_token") or "other")
        if token not in token_weights:
            token = "other"
        token_weights[token] = token_weights.get(token, 0.0) + float(row.get("vote_weight") or 0.0)

    winner = sorted(token_weights.items(), key=lambda kv: (-float(kv[1]), kv[0]))[0][0]

    winner_rows = [r for r in normalized_rows if r.get("condition_token") == winner]
    if winner_rows:
        winner_rows.sort(key=lambda r: (-float(r.get("vote_weight") or 0.0), str(r.get("provider") or "")))
        winner_label = str(winner_rows[0].get("condition_text") or winner).strip()
    else:
        winner_label = winner

    cloud_points: list[tuple[float, float]] = []
    for row in normalized_rows:
        w = float(row.get("vote_weight") or 0.0)
        if w <= 0:
            continue
        cc = _safe_float(row.get("cloud_cover"))
        if cc is None:
            cc = TOKEN_TO_CLOUD_PROXY.get(str(row.get("condition_token") or "other"), 50.0)
        cloud_points.append((w, cc))

    mean_cloud_cover: float | None
    if cloud_points:
        denom = sum(w for w, _ in cloud_points)
        mean_cloud_cover = sum(w * cc for w, cc in cloud_points) / denom if denom > 0 else None
    else:
        mean_cloud_cover = None

    sky_label = sky_label_from_cloud_cover(mean_cloud_cover, winner)
    entropy = _normalized_entropy(token_weights)

    raw_provider_labels = {}
    for row in normalized_rows:
        raw_provider_labels[str(row.get("provider"))] = {
            "condition_text": row.get("condition_text"),
            "condition_icon": row.get("condition_icon"),
            "cloud_cover": row.get("cloud_cover"),
            "condition_token": row.get("condition_token"),
            "vote_weight": round(float(row.get("vote_weight") or 0.0), 6),
        }

    token_weights_clean = {
        k: round(float(v), 6)
        for k, v in token_weights.items()
        if float(v) > 0.0
    }

    return {
        "condition_token": winner,
        "condition_label": winner_label,
        "sky_label": sky_label,
        "mean_cloud_cover": None if mean_cloud_cover is None else float(mean_cloud_cover),
        "vote_entropy": float(entropy),
        "provider_count": len(normalized_rows),
        "raw_provider_labels_json": json.dumps(raw_provider_labels, sort_keys=True),
        "token_weights_json": json.dumps(token_weights_clean, sort_keys=True),
    }
