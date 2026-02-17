from typing import Any

_LSTM_CACHE: dict[str, float | None] = {}


def get_lstm_prediction(city: str) -> float | None:
    key = str(city or "").strip().lower()
    if not key:
        return None
    if key in _LSTM_CACHE:
        return _LSTM_CACHE[key]

    try:
        from daily_prediction import getPrediction  # lazy import to avoid startup overhead

        v = getPrediction(key)
        out = None if v is None else float(v)
    except Exception:
        out = None

    _LSTM_CACHE[key] = out
    return out


def compute_candidate_mode_predictions(
    *,
    city: str,
    forecast_pred: float | None,
    blend_forecast_weight: float = 0.8,
    known_lstm_pred: float | None = None,
) -> dict[str, Any]:
    forecast_val = None if forecast_pred is None else float(forecast_pred)
    lstm_val = known_lstm_pred
    if lstm_val is None:
        lstm_val = get_lstm_prediction(city)

    blend_val = None
    if forecast_val is not None and lstm_val is not None:
        w = max(0.0, min(1.0, float(blend_forecast_weight)))
        blend_val = (w * forecast_val) + ((1.0 - w) * float(lstm_val))

    candidates = {
        "forecast": forecast_val,
        "blend": blend_val,
        "lstm": None if lstm_val is None else float(lstm_val),
    }
    available = [k for k, v in candidates.items() if v is not None]
    return {
        "mode_forecast_pred": forecast_val,
        "mode_blend_pred": blend_val,
        "mode_lstm_pred": None if lstm_val is None else float(lstm_val),
        "available_actions": available,
    }


def choose_mode_prediction(
    *,
    selected_action: str,
    candidates: dict[str, float | None],
    fallback_action: str = "forecast",
) -> tuple[str, float | None, str]:
    action = str(selected_action or "").strip().lower()
    fallback = str(fallback_action or "forecast").strip().lower()

    if action in candidates and candidates.get(action) is not None:
        return action, float(candidates[action]), "selected"

    if fallback in candidates and candidates.get(fallback) is not None:
        return fallback, float(candidates[fallback]), "fallback_unavailable"

    for a in ("forecast", "blend", "lstm"):
        if a in candidates and candidates.get(a) is not None:
            return a, float(candidates[a]), "fallback_first_available"

    return "none", None, "no_candidate_available"
