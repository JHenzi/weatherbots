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

## 2. Uncertainty Modeling ($\sigma$)

Quantifying uncertainty is critical for calculating the probability of hitting a specific Kalshi temperature bucket.

### Sigma Calculation
The system determines the standard deviation $\sigma$ for the prediction distribution as follows:

$$\sigma = \max(\sigma_{\text{spread}}, MAE_{\text{historical}})$$

Where:
- $\sigma_{\text{spread}}$: The standard deviation of the predictions from the different forecast providers (inter-model disagreement).
- $MAE_{\text{historical}}$: The historical Mean Absolute Error for that specific city, retrieved from `city_metadata.json`.

This approach ensures that if the models disagree, the distribution widens. Even if they perfectly agree, the distribution remains at least as wide as the historical average error.

## 3. Probability Distribution

The bot assumes that the realized temperature $T$ follows a Normal distribution centered at the consensus mean:

$$T \sim N(\mu, \sigma^2)$$

### Bucket Probability
For a Kalshi bucket defined by the range $[L, H]$, the probability $P$ is calculated using the Cumulative Distribution Function (CDF) of the Normal distribution:

$$P(L \le T \le H) = \Phi\left(\frac{H - \mu}{\sigma}\right) - \Phi\left(\frac{L - \mu}{\sigma}\right)$$

Where $\Phi$ is the standard Normal CDF.

- For "Above $X$" buckets: $P(T \ge X) = 1 - \Phi\left(\frac{X - \mu}{\sigma}\right)$
- For "Below $X$" buckets: $P(T \le X) = \Phi\left(\frac{X - \mu}{\sigma}\right)$

## 4. Expected Value (EV) and Trade Selection

### Expected Value Calculation
The Expected Value ($EV$) of a "YES" contract is calculated in cents:

$$EV_{\text{cents}} = (100 \cdot P_{\text{model}}) - Price_{\text{market}}$$

Where:
- $P_{\text{model}}$: The probability calculated in Step 3.
- $Price_{\text{market}}$: The current "YES" ask price on Kalshi (in cents, 1-99).

### Trade Decision
A trade is executed only if:
1. $EV_{\text{cents}} \ge \text{Min EV Threshold}$ (default 3 cents).
2. $\text{Confidence Score} \ge \text{Min Confidence}$ (derived from $\sigma_{\text{spread}}$).
3. $\sigma_{\text{spread}} \le \text{Max Spread}$.

## 5. Position Sizing and Allocation

### City Budget Allocation
The total daily budget is split across cities based on historical performance:

$$Score_{city} = \frac{1}{1 + MAE_{city}^2} \cdot (0.5 + HitRate_{city})$$

Budgets are then normalized such that the sum equals the total daily cap, ensuring more capital is allocated to cities where the bot has higher historical accuracy.

### Contract Count
The number of contracts to buy is:

$$Count = \min\left(Count_{\text{desired}}, \frac{Budget_{city}}{Price_{\text{market}}}, MaxContractsPerOrder\right)$$
