# Mathematical Foundations

This document details the mathematical models and formulas used by the trading bot for forecasting, uncertainty quantification, and trade selection.

## 1. Weighted Consensus Model

The bot uses an ensemble of weather sources. To favor more accurate sources, it implements a weighted voting system.

### Weight Formula
The weight $w_i$ for a source $i$ is calculated based on its Mean Absolute Error ($MAE$) over a rolling $N$-day window:

$$w_i = \frac{1/MAE_i^2}{\sum_{j=1}^{M} 1/MAE_j^2}$$

Where:
- $MAE_i$: The average absolute difference between the source's prediction and the NWS actual high.
- $M$: The number of available sources for that city.

Using the square of the MAE ($1/MAE^2$) ensures that highly accurate sources receive significantly more influence than mediocre ones.

### Final Prediction ($\mu$)
The consensus mean $\mu$ is the weighted average of the individual predictions $x_i$:

$$\mu = \sum_{i=1}^{M} w_i \cdot x_i$$

## 1b. Implementation notes (current code)

This section documents the “in-place” implementation as of the current Docker build, so the math matches what’s actually executed.

### What gets weighted

- **Calibration weights (`Data/weights.json`)** are computed by `calibrate_sources.py` over sources in:
  - `consensus`, `open-meteo`, `visual-crossing`, `tomorrow`, `weatherapi`, `google-weather`,
    `openweathermap`, `pirateweather`, `weather.gov`, `lstm`
- **Intraday forecast aggregation (`intraday_pulse.py`)** uses **forecast providers only** (excludes `lstm` and ignores `consensus`), and only over providers that returned a value on that run.

### Rolling window (inclusive)

Calibration runs for an `as_of` event date (typically “yesterday” in cron). The MAE window is inclusive:

\[
[as\_of-(N-1),\ as\_of]
\]

This ensures that once a day’s NWS truth is fetched, that day immediately influences weights for subsequent intraday/trade runs.

## 2. Uncertainty Modeling ($\sigma$)

Quantifying uncertainty is critical for calculating the probability of hitting a specific Kalshi temperature bucket.

### Sigma Calculation
The system determines the standard deviation $\sigma$ for the prediction distribution as follows:

$$\sigma = \max(\sigma_{\text{spread}}, MAE_{\text{historical}})$$

Where:
- $\sigma_{\text{spread}}$: The standard deviation of the predictions from the different forecast providers (inter-model disagreement).
- $MAE_{\text{historical}}$: The historical Mean Absolute Error for that specific city, retrieved from `city_metadata.json`.

This approach ensures that if the models disagree, the distribution widens. Even if they perfectly agree, the distribution remains at least as wide as the historical average error.

### Implementation notes: two “sigmas”

The system uses two related but distinct spread notions:

1) **Intraday snapshot spread** (`intraday_pulse.py`)

- \(\sigma_{\text{snapshot}} = \text{pstdev}(\{x_i\})\) across available provider forecasts for that run.
- This is written to:
  - `Data/intraday_forecasts.csv` as `current_sigma`
  - `Data/predictions_latest.csv` as `spread_f` (when `--write-predictions` is used)

2) **Trading distribution sigma** (`kalshi_trader.py`)

- \(\sigma = \max(spread\_f,\ historical\_MAE)\)
- where `historical_MAE` is derived from `Data/source_performance.csv` (typically using `source_name=consensus` errors) and written to `Data/city_metadata.json`.

## 3. Probability Distribution

The bot assumes that the realized temperature $T$ follows a Normal distribution centered at the consensus mean:

$$T \sim N(\mu, \sigma^2)$$

### Bucket Probability
For a Kalshi bucket defined by the range $[L, H]$, the probability $P$ is calculated using the Cumulative Distribution Function (CDF) of the Normal distribution:

$$P(L \le T \le H) = \Phi\left(\frac{H - \mu}{\sigma}\right) - \Phi\left(\frac{L - \mu}{\sigma}\right)$$

Where $\Phi$ is the standard Normal CDF.

- For "Above $X$" buckets: $P(T \ge X) = 1 - \Phi\left(\frac{X - \mu}{\sigma}\right)$
- For "Below $X$" buckets: $P(T \le X) = \Phi\left(\frac{X - \mu}{\sigma}\right)$

## 4. Confidence, Expected Value (EV) and Trade Selection

### Prediction: inverse-MAE weighting (meritocratic)

When rolling MAE data is available (e.g. from `source_performance.csv` over the last 7 days):

- **Weights:** \(w_i = 1 / \text{MAE}_i^2\) (with a small floor on MAE to avoid division by zero). Only sources with MAE in the window participate; others are excluded from the prediction.
- **Consensus / mean forecast:** \(\mu = \sum_i (w_i / \sum_j w_j) \cdot T_i\), i.e. the weighted average of provider forecasts using these weights.
- If no MAE data exists for a city, the system falls back to weights from `weights.json` or equal weights.

### Confidence score: smart spread (meritocratic)

The system derives a **confidence score** that uses only **reliable** sources for spread, so bad sources cannot lower confidence:

- **Reliable vs unreliable:** Using rolling MAE per (city, source):
  - \(\text{best\_MAE} = \min_i \text{MAE}_i\) over sources that have both a forecast and MAE.
  - **Reliable sources** = sources with \(\text{MAE}_i \le 1.5 \times \text{best\_MAE}\).
  - All others are **unreliable**; their disagreement is ignored for spread.

- **Spread component** (agreement among reliable providers only):
  - \(\sigma_{\text{spread}}\) = population standard deviation of forecasts from **reliable sources only** (LSTM excluded from spread when applicable).
  - If there are 0 or 1 reliable sources, \(\sigma_{\text{spread}} = 0\) (high agreement).
  - Map \(\sigma_{\text{spread}}\) to raw confidence:
    - If \(\sigma_{\text{spread}} \le 1.5\): \(c_{\text{spread}} = 1.0\)
    - If \(\sigma_{\text{spread}} \ge 3.0\): \(c_{\text{spread}} = 0.0\)
    - Otherwise: \(c_{\text{spread}} = (3.0 - \sigma_{\text{spread}}) / (3.0 - 1.5)\)
  - Cap at 0.9; optionally add **+0.1** when the best source’s MAE is &lt; 0.8 × runner-up MAE (i.e. &gt;20% better), then cap again at 0.9.

- **Skill component** (ensemble robustness from the weights used for prediction):
  - The same weights \(w_i\) (inverse-MAE or fallback) are interpreted as a probability distribution; Shannon entropy and normalized skill score \(c_{\text{skill}}\) are computed as before (entropy / \(H_{\max}\), clipped to \([0, 1]\); default 0.5 when \(N \le 1\)).

- **Final confidence score**:
  \[
  c_{\text{final}} = c_{\text{spread, capped}} \times \left(0.5 + 0.5 \cdot c_{\text{skill}}\right)
  \]
  So: only disagreement among **reliable** sources lowers confidence; unreliable sources do not increase \(\sigma\).

### Expected Value Calculation

The Expected Value ($EV$) of a "YES" contract is calculated in cents:

$$EV_{\text{cents}} = (100 \cdot P_{\text{model}}) - Price_{\text{market}}$$

Where:
- $P_{\text{model}}$: The probability calculated in Step 3.
- $Price_{\text{market}}$: The current "YES" ask price on Kalshi (in cents, 1-99).

### Trade Decision

A trade is executed only if:
1. $EV_{\text{cents}} \ge \text{Min EV Threshold}$ (default 3 cents).
2. $\text{Confidence Score} \ge \text{Min Confidence}$ (now the blended spread×skill score).
3. $\sigma_{\text{spread}} \le \text{Max Spread}$.

## 5. Position Sizing and Allocation

### City Budget Allocation
The total daily budget is split across cities based on historical performance:

$$Score_{city} = \frac{1}{1 + MAE_{city}^2} \cdot (0.5 + HitRate_{city})$$

Budgets are then normalized such that the sum equals the total daily cap, ensuring more capital is allocated to cities where the bot has higher historical accuracy.

### Contract Count
The number of contracts to buy is:

$$Count = \min\left(Count_{\text{desired}}, \frac{Budget_{city}}{Price_{\text{market}}}, MaxContractsPerOrder\right)$$
