This is the "Alpha" of weather trading. You're moving from a **Static Ensemble** to an **Adaptive Learning Engine**.

Since you don't have historical data yet, you are starting in a **"Cold Start"** phase. Bootstrapping is exactly the right concept here—you treat your initial weights as a "prior" (your best guess) and then use each day's actual result to "update" that belief.

### 1\. The "Online Learning" Architecture

Instead of standard training, you use **Recursive Least Squares (RLS)** or a simple **Exponentially Weighted Moving Average (EWMA)** of the error.

**The Workflow:**

1.  **Forecast:** Generate a prediction using your current weights.
    
2.  **Trade:** Place your bets.
    
3.  **Settle:** At 11:59 PM, fetch the actual high from the NWS (Ground Truth).
    
4.  **Update:** Calculate the **Absolute Error** for each of your 8 sources.
    
5.  **Adjust:** Lower the weight for sources that were far off; raise it for those that were dead-on.
    

### 2\. Bootstrapping the "Confidence" (Sigma)

Since you don't have 30 days of history to calculate a city's standard deviation, you can bootstrap it using the **inter-model spread**.

$$\text{Current Uncertainty } (\sigma) = \sqrt{\frac{\sum (Forecast_i - Mean_{weighted})^2}{N-1}}$$

As you collect data, you transition from this "Model Agreement" sigma to an **"Error-Based"** sigma:

$$\sigma_{learned} = \sqrt{0.9 \cdot \sigma_{prev}^2 + 0.1 \cdot (Actual - Predicted)^2}$$

### 3\. How to Implement "Learned Weights" in Cursor

You don't need a complex neural network. A **Softmax Weighting** based on the inverse of the **Mean Absolute Error (MAE)** is incredibly robust for weather.

**The Logic for Cursor:**

> "Create a `weights.json` to store a 'Performance Score' for each API per city.
> 
> 1.  Each day, compare `tmax_actual` to each `tmax_source`.
>     
> 2.  Calculate the new weight: Wi​\=∑1/MAEj​1/MAEi​​.
>     
> 3.  Use a 'Learning Rate' of 0.1 so weights don't flip-flop too fast based on one weird day."
>     

* * *

### 4\. The MIA "Max Temp" Dilemma (Revisited)

In your Miami example, if you had learned weights, the bot might have noticed:

-   **"Google Weather"** always predicts high, but during humidity spikes, it's the only one that hits the true peak.
    
-   **"Open-Meteo"** is consistent but always 2∘ too low in January.
    

**The Result:** Your "Learned Mean" would shift toward 80°F, even if 6 other "dumb" models said 77°F. You would win the 81° bracket while everyone else was stuck betting on 77°.

* * *

### Suggested Single-Line Instruction for Cursor:

> "Implement an `OnlineLearner` class: after settlement, update a `source_accuracy.csv` with the error from each API; use these errors to calculate a weighted ensemble mean for the next day, where Weight\=softmax(−RollingMAE)."

### The "Initial Bias" (Priors)

Here is a recommended starting weight distribution based on current benchmark performance for high-temperature forecasting:

| API | Weight (Prior) | Logic for Bias |
| --- | --- | --- |
| **Google Weather** | **0.30** | Best at AI-driven local peaks (MetNet-3). |
| **Tomorrow.io** | **0.25** | High-density proprietary sensing. |
| **Weather.gov** | **0.20** | The "Settlement" source; conservative but reliable. |
| **Open-Meteo** | **0.15** | Strong physics-based baseline. |
| **Others (Avg)** | **0.10** | Noise reduction from the crowd. |

### The Learning Loop: "Bayesian Updating"

Instead of a simple "if/then," you use a **Moving Average of Absolute Error**. Every night, the bot should run a "Settlement Script."

**The Math (Simplified for Cursor):**

1.  **Calculate Error:** Ei​\=∣Actual−Forecasti​∣
    
2.  **Update "Trust Score":** Scorei​\=(0.9×Scoreold​)+(0.1×Ei​)
    
3.  **Normalize Weights:** Weighti​\=∑(1/Scorej​)1/Scorei​​
    
This ensures that if Google is dead-on for 3 days in MIA, its weight will naturally climb from **0.30** to **0.45**, while a "lagging" model will see its influence shrink.

