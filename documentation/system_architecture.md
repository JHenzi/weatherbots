# System Architecture: Kalshi Weather Trading Bot

This document provides a high-level overview of the Kalshi Weather Trading Bot's architecture, data flow, and components.

## Overview

The bot is designed to autonomously trade daily high-temperature markets on Kalshi. It achieves this by gathering forecasts from multiple weather providers, generating a consensus prediction, mapping this prediction to Kalshi market buckets, and executing trades based on expected value (EV) and confidence guardrails.

## Core Components

### 1. Data Ingestion & Cleaning (`daily_prediction.py`)
- **Sources**: Fetches historical and forecast data from multiple APIs:
    - Open-Meteo (Archive & Forecast)
    - Visual Crossing (Timeline)
    - Meteostat
    - NOAA NCEI (Daily Summaries)
    - Tomorrow.io
    - WeatherAPI.com
    - Google Weather
    - OpenWeatherMap
    - Pirate Weather
    - National Weather Service (weather.gov)
- **Cleaning**: Normalizes units (Celsius to Fahrenheit), handles missing values via forward/backward filling, and calculates derived features like `tmax_avg`.
- **LSTM Model**: Uses a pre-trained LSTM (TensorFlow/Keras) to provide an autoregressive prediction based on the last 10 days of observed weather.

### 2. Consensus & Weighting (`run_daily.py` & `calibrate_sources.py`)
- **Weighted Voting**: Combines predictions from all available sources using learned weights.
- **Weight Calculation**: Weights are inversely proportional to the square of the Mean Absolute Error ($w_i = 1/MAE_i^2$) over a rolling window (default 14 days).
- **Confidence Scoring**: Computes a confidence score based on the spread (standard deviation) between different forecast providers.

### 3. Market Mapping & Trading (`kalshi_trader.py`)
- **Market Selection**: Identifies the correct Kalshi event and markets (buckets) for each city (NYC, Chicago, Austin, Miami).
- **Probability Modeling**: Models the predicted temperature as a Normal distribution $N(\mu, \sigma)$, where $\mu$ is the consensus mean and $\sigma$ is derived from the forecast spread and historical residuals.
- **Orderbook Integration**: Fetches real-time orderbook data to determine the best YES ask price and available liquidity.
- **Trade Execution**: Calculates the Expected Value (EV). If the EV exceeds a threshold and confidence guardrails are met, it places a limit order.

### 4. Calibration & Evaluation (`calibrate_sources.py`, `settle_eval.py`, `daily_metrics.py`)
- **Nightly Calibration**: Once the actual high temperature is available from the NWS Climate report (CLI), the bot grades each source's performance and updates the `weights.json` file.
- **Settlement**: Backfills realized outcomes into `eval_history.csv` to track PnL and model accuracy.
- **Metrics**: Computes daily MAE, RMSE, bucket hit rates, and rolling performance metrics.

## Data Flow

1.  **Input**: Schedule (Cron) triggers `run_daily.py`.
2.  **Forecast**: `daily_prediction.py` gathers current forecasts and LSTM predictions.
3.  **Consensus**: `run_daily.py` applies learned weights to produce a final $\mu$ and $\sigma$.
4.  **Market Fetch**: `kalshi_trader.py` pulls Kalshi event data and orderbooks.
5.  **Decision**: The bot calculates $P(\text{Bucket})$ and $EV$.
6.  **Action**: Limit orders are sent to Kalshi (if in live mode).
7.  **Feedback**: Nightly scripts fetch ground truth and update model weights for the next day.

## Technologies Used
- **Language**: Python 3
- **Data Analysis**: Pandas, NumPy, Scikit-learn
- **Machine Learning**: TensorFlow / Keras (LSTM)
- **APIs**: Kalshi Trade API v2, Various Weather APIs
- **Infrastructure**: Docker, Cron, `.env` for secret management
