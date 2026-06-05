"""
Modeling Module — Hazara Division AQI
======================================
Trains multiple ML models (scikit-learn + XGBoost) and selects
the best one based on RMSE. Also provides 72-hour iterative
forecast generation with dampening.
"""

import sys
import os
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
import xgboost as xgb
import pickle

from src.preprocessing import preprocess_data

DEFAULT_DISTRICT = "Abbottabad"


def load_training_data(district=None):
    """
    Load training data from local CSV files.
    """
    district = district or DEFAULT_DISTRICT

    # Load from local CSV
    data_dir = os.path.join(project_root, "data")
    csv_path = os.path.join(data_dir, "cleaned_hazara_aqi.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=['date'])
        if district and 'district' in df.columns:
            df = df[df['district'] == district]
        print(f"Loaded {len(df)} records from local CSV for {district}")
        return df

    print("No data found. Run run_pipeline.py first.")
    return pd.DataFrame()


def train_and_evaluate(district=None):
    """
    Trains multiple models and selects the best one based on RMSE.
    """
    district = district or DEFAULT_DISTRICT

    print("=" * 50)
    print("  Hazara Division AQI Model Training")
    print("=" * 50)

    df = load_training_data(district)
    if df.empty:
        return

    print(f"Total data fetched: {len(df)} records.")

    # Filter out future data
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        if df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_localize(None)

    df = df[df['date'] < datetime.now()]
    print(f"Training on {len(df)} historical records.")

    if 'district' in df.columns:
        df = df.drop(columns=['district'])

    # Preprocess
    df = preprocess_data(df, is_training=True)

    # Target: next hour's AQI
    df['target'] = df['us_aqi'].shift(-1)
    df = df.dropna()

    features = [c for c in df.columns if 'lag' in c or 'rolling' in c or 'change' in c
                or c in ['hour', 'day_of_week', 'month']]
    print(f"Features used: {features}")

    X = df[features]
    y = df['target']

    # Chronological split (80/20)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    models = {
        "LinearRegression": LinearRegression(),
        "RidgeRegression": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, max_depth=12, n_jobs=-1, random_state=42
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        ),
        "XGBoost": xgb.XGBRegressor(
            objective='reg:squarederror', n_estimators=150,
            learning_rate=0.1, max_depth=6, random_state=42, verbosity=0
        ),
    }

    best_model = None
    best_score = float('inf')
    best_name = ""

    print(f"\n{'Model':<25} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
    print("-" * 61)

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            status = "<-- BEST" if rmse < best_score else ""
            print(f"{name:<25} {rmse:<12.4f} {mae:<12.4f} {r2:<12.4f} {status}")

            if rmse < best_score:
                best_score = rmse
                best_model = model
                best_name = name
        except Exception as e:
            print(f"Failed to train {name}: {e}")

    if best_model is None:
        print("\nError: No models were successfully trained.")
        sys.exit(1)

    print(f"\n🏆 Best model: {best_name} with RMSE: {best_score:.4f}")

    # Save to models/ folder
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    with open(os.path.join(models_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)

    with open(os.path.join(models_dir, 'features.pkl'), 'wb') as f:
        pickle.dump(features, f)

    print(f"Model and features saved to {models_dir}")
    return best_model, features


def predict_next_72_hours(model, features, recent_data):
    """
    Generates 72-hour (3-day) ahead forecast using iterative 1-hour predictions.
    Uses dampening to prevent wild oscillations.
    """
    predictions = []
    history_df = recent_data.tail(100).copy()
    max_physical_aqi = 500

    for i in range(72):
        df_processed = preprocess_data(history_df.copy(), is_training=False)
        input_row = df_processed.iloc[-1:]

        for feat in features:
            if feat not in input_row.columns:
                input_row[feat] = 0

        X_input = input_row[features]
        raw_pred = model.predict(X_input)[0]
        prev_aqi = history_df['us_aqi'].iloc[-1]

        # Dampening: limit prediction to ±15% of previous value
        dampening = 0.15
        lower_bound = prev_aqi * (1 - dampening)
        upper_bound = prev_aqi * (1 + dampening)

        pred_aqi = np.clip(raw_pred, lower_bound, upper_bound)
        pred_aqi = np.clip(pred_aqi, 0, max_physical_aqi)

        last_date = history_df['date'].iloc[-1]
        next_date = last_date + pd.Timedelta(hours=1)

        new_row = pd.DataFrame({'date': [next_date], 'us_aqi': [pred_aqi]})
        last_known = history_df.iloc[-1].to_dict()
        for col, val in last_known.items():
            if col not in ['date', 'us_aqi', 'target', 'district'] and col not in new_row.columns \
               and 'lag' not in col and 'rolling' not in col:
                new_row[col] = val

        history_df = pd.concat([history_df, new_row], ignore_index=True)
        predictions.append({'date': next_date, 'predicted_aqi': float(pred_aqi)})

    return pd.DataFrame(predictions)


if __name__ == "__main__":
    train_and_evaluate()
