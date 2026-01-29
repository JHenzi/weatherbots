# Real-Time Weather Observation & Hedging Plan

The goal of this system is **"near-automated" trading**. While the bot executes the technical data gathering and calculations, it serves as a high-conviction co-pilot that presents the mathematical "why" to the human trader, allowing for rapid-fire hedge approvals or fully autonomous execution based on pre-set risk thresholds.

## 1. Data Fetching Strategy

We will use the Synoptic Data API (formerly MesoWest) which powers the Weather.gov Time Series Viewer.

### Target Stations
- **KMIA**: Miami International Airport (Miami, FL)
- **KMDW**: Chicago Midway International Airport (Chicago, IL)
- **KNYC**: Central Park (New York, NY)
- **KAUS**: Austin-Bergstrom International Airport (Austin, TX)

### API Configuration
- **Base URL**: `https://api.synopticdata.com/v2/stations/timeseries`
- **Token**: Managed via `scripts/refresh_synoptic_token.py` (scraped from weather.gov).
- **Frequency**: Every 2 minutes.
- **Parameters**: `STID={site}&token={token}&recent=60&units=english`

### Token Management
Since the token used by Weather.gov can change, we use `scripts/refresh_synoptic_token.py` to:
1. Fetch the latest `mesoToken` from `https://www.weather.gov/source/wrh/apiKey.js`.
2. Update the `SYNOPTIC_TOKEN` variable in the `.env` file.
3. This script should be run if an API call returns a 401/403 error.

## 2. Trend Analysis Logic

A new service `scripts/observation_monitor.py` will handle fetching and analysis, acting as a "Real-Time Bridge" to the existing **Weighted Consensus Model**.

### Bridging Static Forecasts to Real-Time
The system currently uses a static $\mu_{consensus}$ from the daily forecast (comprised of 10+ sources and LSTM models). The monitor will calculate a **Dynamic Mean ($\mu_{live}$)**:
$$\mu_{live}(t) = \max(T_{curr}, T_{high\_seen}, \text{Forecast\_Projection})$$
- **Early Day**: Follows the diurnal warming slope.
- **Approaching Peak**: Uses linear regression to project if the day will "undershoot" or "overshoot" the original $\mu_{consensus}$.

### Trend Calculations
We calculate the slope ($m$) of temperature over time using Simple Linear Regression:
$$m = \frac{n\sum(xy) - \sum x \sum y}{n\sum(x^2) - (\sum x)^2}$$
- **Short-term (`trend_10m`)**: Captures sudden cloud cover or wind shifts.
- **Mid-term (`trend_30m`)**: Standard operational trend.
- **Long-term (`trend_1h`)**: The baseline diurnal curve for the day.

### Acceleration & Quality
- **Acceleration ($a$)**: $\frac{\Delta m}{\Delta t}$. If $a > 0$ while warming, the heat is intensifying (e.g., clear skies at noon).
- **Confidence ($R^2$)**: Measures how "noisy" the data is. High $R^2$ (>0.9) means the trend is very reliable; low $R^2$ (<0.5) suggests erratic winds or sensor jitter.

## 3. Position Evaluation & Bayesian Updating

The monitor correlates real-time observations with open Kalshi trades from `trades_history.csv` using a **Bayesian Convergence Model**.

### Uncertainty Convergence ($\sigma_{live}$)
The static uncertainty $\sigma$ from `mathematical_foundations.md` is based on historical MAE. In real-time, this uncertainty must **shrink** as we approach the peak time $t_{peak}$:
$$\sigma_{live}(t) = \sigma_{static} \cdot \sqrt{\frac{t_{peak} - t}{t_{peak} - t_{start}}}$$
- At $t_{start}$ (morning), $\sigma_{live} \approx \sigma_{static}$.
- At $t_{peak}$ (afternoon), $\sigma_{live} \approx 0$ (because the high is likely already recorded).

### Projected Peak Temperature ($T_{proj}$) & Timing
We estimate the max temperature for the day by combining current observations with the historical diurnal curve and solar timing:
$$T_{proj} = T_{curr} + (m_{1h} \times t_{remaining})$$
- **Peak Window Tracking**: Monitor the "Approach to Daily High". Most cities peak between 2:00 PM and 4:30 PM local time.
- **Solar Noon Alignment**: Use the current time relative to solar noon to adjust $t_{remaining}$ (warming usually slows as it approaches the peak).
- **Status Flags**: 
    - `APPROACHING_HIGH`: Current time is within 2 hours of the expected peak.
    - `PAST_HIGH`: Current time is 30+ minutes past the expected peak (useful for confirming the "High" has been set).

### Real-Time Notifications
The system will push alerts via multiple channels when specific thresholds are met:
- **Thresholds**:
    - **Trend Shift**: Sudden reversal (e.g., warming to cooling) during the peak window.
    - **Bucket Breach**: Real-time temp enters or exits a bet's bucket.
    - **Hedge Trigger**: $P_{fail}$ exceeds 80%.
- **Channels**:
    - **Webhooks**: Discord/Slack for remote monitoring.
    - **Desktop**: macOS `osascript` (System Notifications) for immediate local awareness.
    - **Dashboard Toasts**: Live popups in the Web UI.

### Risk: "Probability of Violation" ($P_{fail}$)
Using the updated $N(\mu_{live}, \sigma_{live}^2)$, we recalculate the probability of landing in the target bucket $[B_{low}, B_{high}]$:
$$P_{success} = \Phi\left(\frac{B_{high} - \mu_{live}}{\sigma_{live}}\right) - \Phi\left(\frac{B_{low} - \mu_{live}}{\sigma_{live}}\right)$$
$$P_{fail} = 1 - P_{success}$$
A position is "At Risk" if $P_{fail}$ increases by more than 25% from the time of entry.

## 4. Hedging & "NO" Share Strategy

### The Hedge Condition: Mathematical Justification
We buy "NO" shares if the cost of the hedge is less than the expected value of protecting the capital. The system will explain this to the human as:
> "We should hedge because the current trend has an 85% probability of breaking the bucket. Protecting the $100 YES position with a $15 NO hedge yields a Net Expected Return of +$65 vs. a $0 total loss."

**Hedge Expected Value ($EV_{hedge}$):**
$$EV_{hedge} = (P_{fail} \times \text{Payout}_{no}) - \text{Cost}_{no}$$
- **Hedge ROI**: Calculated as $\frac{EV_{hedge}}{Cost_{no}}$.
- **Human Decision Support**: Every hedge recommendation will be accompanied by a clear breakdown:
    - **Current Risk**: "High ($P_{fail}=85\%$) - Temperature is 0.2°F from edge and warming at 1.1°F/hr."
    - **Hedge Cost**: "$15.00 to protect $100.00."
    - **Outcome**: "If the bucket breaks (Likely), you salvage $85.00. If the bucket holds (Unlikely), you lose the $15.00 hedge but keep the $100.00 win."

### Arbitrage Scenarios
1. **The "Lagging Market"**: If the station reports a 2°F jump in 10 minutes (sudden heat), but Kalshi "YES" prices are still high, we aggressively buy "NO" shares before the rest of the market notices the update.
2. **The "Bucket Break"**: If $T_{curr}$ is already at $B_{high}$ and the trend is still $+1.5^\circ F/hr$, the "YES" bet is 99% likely to fail. We buy "NO" shares at any price below 95c to salvage some capital.

## 5. Implementation Steps

1. **Phase 1: Monitor Script**
   - Create `scripts/observation_monitor.py` to fetch and log observations. (DONE)
   - Implement auto-refresh: If the API returns a 401/403, the script will automatically invoke `scripts/refresh_synoptic_token.py` and retry. (DONE - now uses official NWS API for stability).
   - Record data to `Data/observations_latest.json` and append to `Data/observations_history.csv`. (DONE)

2. **Phase 2: Web API & Dashboard**
   - Create `scripts/web_dashboard_api.py` using FastAPI. (DONE)
   - Create `scripts/dashboard_web.html` with Tailwind and Chart.js. (DONE)

3. **Phase 3: Trading Engine Updates**
   - Modify `kalshi_trader.py` to remove the "YES-only" restriction. (DONE)
   - Implement a `--hedge` mode that uses the monitor's data to decide whether to buy "NO" shares. (PENDING - logic is in web advisor, but CLI mode is next).
   - **Web Trading API**: Expose functions to allow the web dashboard to trigger manual Buy/Sell orders via a REST API. (DONE)

4. **Phase 4: Automation**
   - Add the monitor to `docker-compose.yml` as a sidecar service.
   - Update crontab to run the hedge check every 5-10 minutes during active trading hours.
