# Improvement Roadmap

This document outlines suggested future enhancements for the Kalshi Weather Trading Bot, categorized by their potential impact and implementation difficulty.

## 1. Short-Term Enhancements (High Impact, Low Difficulty)

### Exponentially Decayed Weighting
- **Goal**: Make the bot adapt faster to seasonal changes or recent model performance.
- **Action**: Modify `calibrate_sources.py` to use an exponentially weighted moving average (EWMA) for MAE instead of a simple rolling average.
- **Formula**: $MAE_{new} = \alpha \cdot Error_{today} + (1 - \alpha) \cdot MAE_{old}$

### Robust Outlier Handling (Huber Loss)
- **Goal**: Prevent a single "weird" weather day from skewing model weights.
- **Action**: Implement Huber loss or cap the maximum error used in the calibration step (e.g., cap absolute error at 10°F).

### Improved Subtitle Parsing
- **Goal**: Increase the robustness of market mapping.
- **Action**: Move from regex-based parsing to a more structured approach using Kalshi's `floor_strike` and `cap_strike` fields if available in the API response.

## 2. Medium-Term Enhancements (Medium Impact, Medium Difficulty)

### Seasonal Specialization
- **Goal**: Account for the fact that some models are better in winter vs. summer.
- **Action**: Maintain separate `weights.json` tables for different "seasons" (e.g., Winter, Spring, Summer, Fall) or temperature regimes.

### Probabilistic Scoring (Brier Score)
- **Goal**: Evaluate how well the bot's *probabilities* match reality, not just its *point forecasts*.
- **Action**: Calculate the Brier Score for each trade: $BS = (P_{\text{predicted}} - Outcome)^2$, where Outcome is 1 if the bucket hit and 0 otherwise. Use this to refine the $\sigma$ calculation.

### Fractional Kelly Criterion
- **Goal**: Optimize position sizing for long-term capital growth.
- **Action**: Use the Kelly formula to determine the optimal bet size based on the model's edge and the probability of loss, likely using a conservative "Half-Kelly" or "Quarter-Kelly" multiplier.

## 3. Long-Term Enhancements (High Impact, High Difficulty)

### Contextual Bandits for Model Selection
- **Goal**: Dynamically choose the best prediction mode (LSTM vs. Forecast vs. Blend) based on the current context.
- **Action**: Implement a simple reinforcement learning agent (e.g., Epsilon-Greedy or Thompson Sampling) that uses features like city, month, and model spread to select the daily strategy.

### Multi-Bucket "Sweep" Strategy
- **Goal**: Hedge the prediction by buying multiple adjacent buckets.
- **Action**: Instead of picking the single highest-EV bucket, allocate the city budget across all buckets where $EV > 0$. This creates a "safety net" and increases the probability of some profit.

### Alternative Distributions
- **Goal**: Move beyond the Normal distribution assumption.
- **Action**: Explore the use of Skew-Normal or Student's T distributions, or use kernel density estimation (KDE) over the ensemble of individual model predictions to capture non-Normal behavior.
