from typing import Any


def compute_candidate_mode_predictions(
    *,
    city: str,
    forecast_pred: float | None,
    bias_correction_f: float = 0.0,
) -> dict[str, Any]:
    """
    Compute candidate predictions for each bandit action.

    Actions:
      - forecast: raw weighted-ensemble mean (no correction)
      - blend:    forecast + bias_correction_f (calibrated to remove systematic cold bias)

    LSTM has been retired — it was 20-35°F off due to stale training data.
    """
    forecast_val = None if forecast_pred is None else float(forecast_pred)

    blend_val = None
    if forecast_val is not None:
        corr = float(bias_correction_f) if bias_correction_f else 0.0
        blend_val = forecast_val + corr

    available = []
    if forecast_val is not None:
        available.append("forecast")
    if blend_val is not None:
        available.append("blend")

    return {
        "mode_forecast_pred": forecast_val,
        "mode_blend_pred": blend_val,
        "mode_lstm_pred": None,
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
