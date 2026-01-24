import argparse
import csv
import datetime as dt
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.optimizers import Adam
except ModuleNotFoundError:
    tf = None


CITIES = ["ny", "il", "tx", "fl"]
TIME_STEPS = 10
FEATURES = ["day_of_year", "tmax", "tmin", "prec", "humi"]


def _local_tz() -> dt.tzinfo:
    tzname = (os.getenv("TZ") or "").strip() or "America/New_York"
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(tzname)
    except Exception:
        return dt.datetime.now().astimezone().tzinfo or dt.timezone.utc


def _parse_args():
    p = argparse.ArgumentParser(description="Train per-city LSTM models on latest cleaned data.")
    p.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help="YYYY-MM-DD. Train using data up to this date (inclusive). Default: max date in data.",
    )
    p.add_argument("--days-window", type=int, default=730, help="Use only last N days for training.")
    p.add_argument("--val-days", type=int, default=30, help="Use last N days as validation (chronological).")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--model-units", type=int, default=70)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--min-improvement", type=float, default=0.0, help="Required MAE improvement vs previous model (°F).")
    return p.parse_args()


def create_dataset(X: np.ndarray, y: np.ndarray, time_steps: int) -> tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i : (i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


def inv_tmax_from_scaled(scaler: StandardScaler, tmax_scaled: np.ndarray) -> np.ndarray:
    # tmax is column index 1 in FEATURES ordering.
    dummy = np.zeros((len(tmax_scaled), len(FEATURES)))
    dummy[:, 1] = tmax_scaled.reshape(-1)
    inv = scaler.inverse_transform(dummy)
    return inv[:, 1]


def eval_model_mae_f(model: tf.keras.Model, scaler: StandardScaler, X: np.ndarray, y_scaled: np.ndarray) -> float:
    pred_scaled = model.predict(X, verbose=0).reshape(-1)
    y_f = inv_tmax_from_scaled(scaler, y_scaled.reshape(-1))
    pred_f = inv_tmax_from_scaled(scaler, pred_scaled)
    return float(np.mean(np.abs(pred_f - y_f)))


def train_city(
    *,
    city: str,
    as_of_date: dt.date | None,
    days_window: int,
    val_days: int,
    epochs: int,
    batch_size: int,
    units: int,
    lr: float,
    min_improvement: float,
    run_dir: Path,
) -> None:
    src_path = Path(f"Data/prediction_data_cleaned_{city}.pkl")
    if not src_path.exists():
        raise FileNotFoundError(f"Missing {src_path}")

    df = pd.read_pickle(src_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Build feature frame identical to inference.
    df = df.rename(
        columns={
            "day": "day_of_year",
            "tmax_avg": "tmax",
            "tmin_avg": "tmin",
            "prec_om": "prec",
            "humi_vc": "humi",
        }
    )
    df = df[["date"] + FEATURES].copy()
    df = df.ffill().bfill()

    if as_of_date is None:
        as_of_date = df["date"].max().date()

    df = df[df["date"].dt.date <= as_of_date]

    # Apply trailing training window.
    if days_window and days_window > 0:
        start = as_of_date - dt.timedelta(days=days_window - 1)
        df = df[df["date"].dt.date >= start]

    if len(df) < TIME_STEPS + val_days + 5:
        raise RuntimeError(f"Not enough data for {city}: {len(df)} rows")

    X_raw = df[FEATURES].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    y_scaled = X_scaled[:, 1]  # scaled tmax

    X, y = create_dataset(X_scaled, y_scaled, TIME_STEPS)

    # Chronological split by validation days (approx).
    # Each label y[i] corresponds to original row index (i+TIME_STEPS).
    val_cut_idx = max(1, len(df) - val_days)
    # Convert original-row cut into sequence index cut.
    seq_cut = max(1, val_cut_idx - TIME_STEPS)

    X_train, y_train = X[:seq_cut], y[:seq_cut]
    X_val, y_val = X[seq_cut:], y[seq_cut:]

    model = Sequential()
    model.add(LSTM(units, activation="relu", input_shape=(TIME_STEPS, X.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer=Adam(lr), loss="mean_squared_error")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    new_mae = eval_model_mae_f(model, scaler, X_val, y_val)

    # Compare to previous model if present
    current_path = Path(f"Data/model_{city}.keras")
    prev_mae = None
    if current_path.exists():
        try:
            prev_model = tf.keras.models.load_model(current_path)
            prev_mae = eval_model_mae_f(prev_model, scaler, X_val, y_val)
        except Exception:
            prev_mae = None

    accept = True
    if prev_mae is not None:
        # Require improvement by at least min_improvement, otherwise keep old model.
        accept = (prev_mae - new_mae) >= float(min_improvement)

    # Save versioned model always (so we can inspect), but only promote if accepted.
    out_path = run_dir / f"model_{city}.keras"
    model.save(out_path.as_posix())

    if accept:
        model.save(current_path.as_posix())

    # Log metrics
    metrics_path = Path("Data/model_metrics.csv")
    write_header = not metrics_path.exists()
    with open(metrics_path, "a", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "run_ts",
                "as_of_date",
                "city",
                "days_window",
                "val_days",
                "epochs",
                "batch_size",
                "units",
                "lr",
                "prev_mae_f",
                "new_mae_f",
                "accepted",
                "versioned_model_path",
                "current_model_path",
            ],
        )
        if write_header:
            w.writeheader()
        w.writerow(
            {
                "run_ts": dt.datetime.now(tz=_local_tz()).isoformat(),
                "as_of_date": as_of_date.strftime("%Y-%m-%d"),
                "city": city,
                "days_window": days_window,
                "val_days": val_days,
                "epochs": epochs,
                "batch_size": batch_size,
                "units": units,
                "lr": lr,
                "prev_mae_f": "" if prev_mae is None else f"{prev_mae:.4f}",
                "new_mae_f": f"{new_mae:.4f}",
                "accepted": str(bool(accept)),
                "versioned_model_path": out_path.as_posix(),
                "current_model_path": current_path.as_posix(),
            }
        )

    print(
        f"{city}: new_mae={new_mae:.2f}F"
        + ("" if prev_mae is None else f" prev_mae={prev_mae:.2f}F")
        + (" ACCEPTED" if accept else " REJECTED")
    )


if __name__ == "__main__":
    args = _parse_args()
    if tf is None:
        raise RuntimeError("TensorFlow is required for training. Install it in your .venv.")

    as_of = dt.datetime.strptime(args.as_of_date, "%Y-%m-%d").date() if args.as_of_date else None
    run_ts = dt.datetime.now(tz=_local_tz())
    run_dir = Path("Data/models") / run_ts.strftime("%Y%m%d")
    run_dir.mkdir(parents=True, exist_ok=True)

    for city in CITIES:
        train_city(
            city=city,
            as_of_date=as_of,
            days_window=args.days_window,
            val_days=args.val_days,
            epochs=args.epochs,
            batch_size=args.batch_size,
            units=args.model_units,
            lr=args.lr,
            min_improvement=args.min_improvement,
            run_dir=run_dir,
        )

