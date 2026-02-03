# LSTM model details

The LSTM setup is **inspired by and aligned with** the [LSTM-Automated-Trading-System](https://github.com/pranavgoyanka/LSTM-Automated-Trading-System) repo (Kalshi Weather Prediction Common Task, BU CS542 Spring 2024). LSTMs are well-suited to weather forecasting because they capture sequential patterns and seasonality.

**Training** is in `train_models.py` (and historically `data_lstm.ipynb` in the upstream repo):

- **One model per city**, saved as `Data/model_<city>.keras` (e.g. `model_fl.keras`, `model_il.keras`, `model_ny.keras`, `model_tx.keras`) and optionally versioned under `Data/models/<YYYYMMDD>/`.
- **Input window**: `time_steps = 10` days (same as the original repo).
- **Features used**: `day_of_year`, `tmax`, `tmin`, and optionally `prec`, `humi` (from the merged multi-source data).
- **Optimizer**: Adam (same as upstream).
- **Epochs**: configurable (e.g. 30 in this repo; the original used 80).
- **Preprocessing**: `StandardScaler` fit on the full feature matrix; inference uses the same feature set. The model predicts *scaled* `tmax`; a dummy feature vector is inverse-transformed to recover °F.

**Cleaned data for LSTM:** After cleaning and feature engineering, per-city data is stored as `Data/prediction_data_cleaned_<city>.pkl`. Regenerate with `daily_prediction.py --prediction-mode lstm` for a given `--trade-date`.
