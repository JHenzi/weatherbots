By shifting from a simple average to a **Weighted Voting System**, you move from a "reactive" bot to a "learning" system that dynamically adjusts its confidence based on reality.

### 1\. The Proximity-to-Actual Score (The "Truth" Calibration)

Instead of just hoping your sources are right, you will grade them. The most common metric for this in weather forecasting is the **Brier Score** or **Mean Absolute Error (MAE)**.

- **Logic:** Every night at 1:30 AM (when the NWS CLI report drops), your bot compares each source's prediction to the **Actual High**.
    
- **The Weighting Formula:**
    
    Weightsource​\=MAEsource2​1​
    
    _(Using the square ensures that small errors are rewarded significantly more than large errors)._
    

---

### 2\. The Logic Flow for your Weighted Voting System

#### **Step 1: Prediction & Voting**

Your sources (Open-Meteo, Tomorrow, etc.) provide their "votes." You apply the weights you’ve calculated from the past 30 days.

| Source | Weight (Based on Skill) | Predicted High | Weighted Vote |
| --- | --- | --- | --- |
| **Visual Crossing** | 0.40 | 41.2°F | 16.48 |
| **Open-Meteo** | 0.25 | 39.5°F | 9.87 |
| **Tomorrow.io** | 0.20 | 43.0°F | 8.60 |
| **LSTM (Historical)** | 0.15 | 40.1°F | 6.01 |
| **TOTAL CONSENSUS** | **1.00** | **—** | **40.96°F** |

Export to Sheets

#### **Step 2: Confidence Guardrail**

The bot computes a **confidence score** that blends:

- **Spread-based agreement**: how tightly clustered the providers are (low standard deviation ⇒ higher confidence).
- **Learned provider skill**: how diversified and reliable the contributing sources are, based on the MAE-driven weights in `Data/weights.json`.

In the current code:

- A raw spread confidence is computed from the inter-model spread:
  - \(\sigma_{\text{spread}} \le 1.5°F \Rightarrow c_{\text{spread}} = 1.0\)
  - \(\sigma_{\text{spread}} \ge 3.0°F \Rightarrow c_{\text{spread}} = 0.0\)
  - Linearly interpolated between 1.5°F and 3.0°F, then capped at 0.9.
- A skill score is computed from the entropy of the learned weights (high entropy ⇒ many good sources contributing ⇒ higher skill).
- The final confidence is:
  - \(c_{\text{final}} = c_{\text{spread, capped}} \times (0.5 + 0.5 \cdot c_{\text{skill}})\).

Trades are only considered when:

- \(c_{\text{final}} \ge \text{Min Confidence}\) (currently 0.75), and
- \(\sigma_{\text{spread}} \le \text{Max Spread}\) (currently 3.0°F).

#### **Step 3: Market Execution (The "Fair Price" Calculation)**

If the **Weighted Consensus** is 41°F, the bot calculates the probability of hitting a Kalshi bucket (e.g., "Above 40°F").

- If the Market Price is **$0.45 (45%)** but your Consensus Probability is **$0.70 (70%)**, you have a **25¢ Edge**.
    
- **Action:** Execute "Buy" because the market rate is significantly below your "Fair Price."
    

---

### 3\. Implementation Plan for `daily_prediction.py`

To make this work, you need to add a "Calibration" step.

> **Instruction for Cursor:** "Modify the system to:
> 
> 1. Create a `Data/source_performance.csv` that logs `[date, city, source_name, predicted_tmax, actual_tmax, absolute_error]`.
>     
> 2. Implement a function `calculate_weights()` that looks at the last 14 days of error and generates a `weights.json`.
>     
> 3. Update the `run_daily.py` logic to use these weights instead of a simple average.
>     
> 4. If the weighted average places a city in a specific Kalshi bucket, but the 'Confidence Score' is below 0.5, **do not log the trade**."
>