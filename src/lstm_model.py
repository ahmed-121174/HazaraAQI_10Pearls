"""
LSTM deep learning model for Hazara Division AQI.
Uses sequential neural network for time series forecasting.
"""

import os
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Suppress verbose TF logs
tf.get_logger().setLevel('ERROR')

SEQUENCE_LEN = 24   # Use past 24 hours to predict next hour
BATCH_SIZE   = 32
MAX_EPOCHS   = 80
PATIENCE     = 10


def build_sequences(data_array, seq_len=SEQUENCE_LEN):
    """Convert time series array into (X, y) sequence pairs."""
    X, y = [], []
    for i in range(len(data_array) - seq_len):
        X.append(data_array[i : i + seq_len])
        y.append(data_array[i + seq_len])
    return np.array(X), np.array(y)


def build_lstm_model(input_shape):
    """Create LSTM model structure."""
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    return model


def train_lstm(df, models_dir, district="Abbottabad"):
    """
    Train LSTM model on AQI time series.
    """
    print(f"\n  [LSTM] Preparing sequences from {district} data...")

    # Use AQI column for LSTM
    df_dist = df[df['district'] == district].sort_values('date').copy()
    df_dist['date'] = pd.to_datetime(df_dist['date'])
    if df_dist['date'].dt.tz is not None:
        df_dist['date'] = df_dist['date'].dt.tz_localize(None)
    df_dist = df_dist[df_dist['date'] < pd.Timestamp.now()]

    aqi_values = df_dist['us_aqi'].dropna().values.reshape(-1, 1)

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    aqi_scaled = scaler.fit_transform(aqi_values)

    # Create training sequences
    X, y = build_sequences(aqi_scaled, SEQUENCE_LEN)
    print(f"  [LSTM] Sequences: {X.shape} → target: {y.shape}")

    # Split train and test sets
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Reshape for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test  = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    print(f"  [LSTM] Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  [LSTM] Building LSTM model...")

    model = build_lstm_model(input_shape=(SEQUENCE_LEN, 1))

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0),
    ]

    print(f"  [LSTM] Training (max {MAX_EPOCHS} epochs, early stopping)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=0
    )

    actual_epochs = len(history.history['loss'])
    print(f"  [LSTM] Stopped at epoch {actual_epochs}")

    # Get predictions and metrics
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    print(f"  [LSTM] Test RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

    # Save model and scaler
    lstm_path   = os.path.join(models_dir, "lstm_model.keras")
    scaler_path = os.path.join(models_dir, "lstm_scaler.pkl")
    model.save(lstm_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"  [LSTM] Model saved: {lstm_path}")
    print(f"  [LSTM] Scaler saved: {scaler_path}")

    return model, scaler, {"name": "LSTM (Deep Learning)", "rmse": rmse, "mae": mae, "r2": r2}


def lstm_predict_next_72(models_dir, recent_aqi_series):
    """
    Load saved LSTM and generate 72-hour forecast.
    """
    lstm_path   = os.path.join(models_dir, "lstm_model.keras")
    scaler_path = os.path.join(models_dir, "lstm_scaler.pkl")

    if not os.path.exists(lstm_path):
        return None

    model = load_model(lstm_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    history = list(recent_aqi_series[-SEQUENCE_LEN:])
    predictions = []

    for _ in range(72):
        seq = np.array(history[-SEQUENCE_LEN:]).reshape(1, SEQUENCE_LEN, 1)
        seq_scaled = scaler.transform(seq.reshape(-1, 1)).reshape(1, SEQUENCE_LEN, 1)
        pred_scaled = model.predict(seq_scaled, verbose=0)[0][0]
        pred = float(scaler.inverse_transform([[pred_scaled]])[0][0])
        pred = np.clip(pred, 0, 500)
        history.append(pred)
        predictions.append(round(pred, 1))

    return predictions


if __name__ == "__main__":
    print("  LSTM Model - Standalone Training Test")

    import sys
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(project_root, "data", "cleaned_hazara_aqi.csv")
    models_dir = os.path.join(project_root, "models")

    df = pd.read_csv(data_path, parse_dates=["date"])
    _, _, metrics = train_lstm(df, models_dir)
    print(f"\nFinal metrics: {metrics}")
