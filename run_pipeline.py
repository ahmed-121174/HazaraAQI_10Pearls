"""
=================================================================
  Hazara Division AQI — Complete Pipeline
  Step 1: Fetch data from Open-Meteo API (6 months backfill)
  Step 2: Clean and preprocess data
  Step 3: Engineer features
  Step 4: Train multiple models and select the best
  Step 5: Save model artifacts to models/
  Step 6: Save cleaned dataset to data/
=================================================================
"""
import os
import sys
import pickle
import warnings
warnings.filterwarnings('ignore')

import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np
from retry_requests import retry
from datetime import datetime, timedelta

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
import xgboost as xgb

# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Hazara Division District Coordinates
HAZARA_DISTRICTS = {
    "Abbottabad":      {"lat": 34.1463, "lon": 73.2117},
    "Mansehra":        {"lat": 34.3302, "lon": 73.1968},
    "Haripur":         {"lat": 33.9942, "lon": 72.9333},
    "Battagram":       {"lat": 34.6837, "lon": 73.0261},
    "Upper Kohistan":  {"lat": 35.2097, "lon": 73.3456},
    "Torghar":         {"lat": 34.6300, "lon": 72.9400},
}

POLLUTANT_COLS = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
                  "sulphur_dioxide", "ozone", "us_aqi"]

# ============================================================
# STEP 1: DATA COLLECTION
# ============================================================
def fetch_district_data(district_name, latitude, longitude):
    """Fetch 6 months historical + 4 day forecast for one district."""
    cache_session = requests_cache.CachedSession(
        os.path.join(DATA_DIR, '.cache'), expire_after=3600
    )
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    end_date = datetime.now().date() + timedelta(days=4)
    start_date = datetime.now().date() - timedelta(days=6 * 30)

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": POLLUTANT_COLS
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    hourly = response.Hourly()

    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )
    }

    for i, col in enumerate(POLLUTANT_COLS):
        hourly_data[col] = hourly.Variables(i).ValuesAsNumpy()

    df = pd.DataFrame(data=hourly_data)
    df['district'] = district_name
    return df


def collect_all_data():
    """Fetch data for all Hazara Division districts."""
    print("=" * 60)
    print("  STEP 1: DATA COLLECTION")
    print("=" * 60)

    all_data = []
    for name, coords in HAZARA_DISTRICTS.items():
        try:
            print(f"  Fetching {name} ({coords['lat']}, {coords['lon']})...", end=" ")
            df = fetch_district_data(name, coords['lat'], coords['lon'])
            all_data.append(df)
            print(f"OK -> {len(df)} records")
        except Exception as e:
            print(f"FAILED: {e}")

    combined = pd.concat(all_data, ignore_index=True)
    
    # Save raw data
    raw_path = os.path.join(DATA_DIR, "raw_hazara_aqi.csv")
    combined.to_csv(raw_path, index=False)
    print(f"\n  Raw data saved: {raw_path}")
    print(f"  Total records: {len(combined)}")
    print(f"  Districts: {combined['district'].nunique()}")
    print(f"  Date range: {combined['date'].min()} to {combined['date'].max()}")

    return combined


# ============================================================
# STEP 2: DATA CLEANING
# ============================================================
def clean_data(df):
    """Clean the raw dataset."""
    print("\n" + "=" * 60)
    print("  STEP 2: DATA CLEANING")
    print("=" * 60)

    initial_len = len(df)
    print(f"  Initial records: {initial_len}")

    # 2a. Check missing values
    missing = df[POLLUTANT_COLS].isnull().sum()
    print(f"\n  Missing values before cleaning:")
    for col, count in missing.items():
        if count > 0:
            print(f"    {col}: {count} ({count/len(df)*100:.1f}%)")
    if missing.sum() == 0:
        print("    None!")

    # 2b. Forward-fill then backward-fill (appropriate for time series)
    df = df.sort_values(['district', 'date']).reset_index(drop=True)
    for col in POLLUTANT_COLS:
        df[col] = df.groupby('district')[col].transform(
            lambda x: x.ffill().bfill()
        )

    # 2c. Handle remaining NaNs (drop rows where AQI is still null)
    before_drop = len(df)
    df = df.dropna(subset=['us_aqi'])
    dropped = before_drop - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with persistent NaN in us_aqi")

    # 2d. Outlier clipping (1st and 99th percentile per pollutant)
    print(f"\n  Outlier clipping (1st-99th percentile):")
    for col in POLLUTANT_COLS:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        clipped = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower, upper)
        if clipped > 0:
            print(f"    {col}: {clipped} values clipped to [{lower:.2f}, {upper:.2f}]")

    # 2e. Remove negative AQI values (invalid)
    neg = (df['us_aqi'] < 0).sum()
    if neg > 0:
        df = df[df['us_aqi'] >= 0]
        print(f"  Removed {neg} negative AQI values")

    # Save cleaned data
    clean_path = os.path.join(DATA_DIR, "cleaned_hazara_aqi.csv")
    df.to_csv(clean_path, index=False)
    print(f"\n  Cleaned data saved: {clean_path}")
    print(f"  Final records: {len(df)} (removed {initial_len - len(df)})")

    return df


# ============================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================
def engineer_features(df):
    """Create time-series features for AQI prediction."""
    print("\n" + "=" * 60)
    print("  STEP 3: FEATURE ENGINEERING")
    print("=" * 60)

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # 3a. Time-based features
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    print("  Added: hour, day_of_week, month")

    # 3b. Rolling averages (per district to avoid cross-contamination)
    for col in ['pm10', 'pm2_5', 'us_aqi']:
        df[f'{col}_rolling_24h'] = df.groupby('district')[col].transform(
            lambda x: x.rolling(window=24, min_periods=1).mean()
        )
        df[f'{col}_rolling_6h'] = df.groupby('district')[col].transform(
            lambda x: x.rolling(window=6, min_periods=1).mean()
        )
    print("  Added: rolling 6h and 24h averages for pm10, pm2_5, us_aqi")

    # 3c. Lag features
    df['lag_1h_aqi'] = df.groupby('district')['us_aqi'].shift(1)
    df['lag_2h_aqi'] = df.groupby('district')['us_aqi'].shift(2)
    df['lag_24h_aqi'] = df.groupby('district')['us_aqi'].shift(24)
    print("  Added: lag_1h_aqi, lag_2h_aqi, lag_24h_aqi")

    # 3d. AQI change rate (derived feature per instructions)
    df['aqi_change_rate'] = df.groupby('district')['us_aqi'].diff()
    print("  Added: aqi_change_rate (AQI difference from previous hour)")

    # 3e. Target variable: NEXT hour's AQI
    df['target'] = df.groupby('district')['us_aqi'].shift(-1)
    print("  Added: target (next hour AQI)")

    # Drop NaN rows from lag/shift operations
    before = len(df)
    df = df.dropna()
    print(f"  Dropped {before - len(df)} NaN rows from lag/shift operations")

    # Save engineered data
    feat_path = os.path.join(DATA_DIR, "features_hazara_aqi.csv")
    df.to_csv(feat_path, index=False)
    print(f"  Feature-engineered data saved: {feat_path}")
    print(f"  Final training-ready records: {len(df)}")

    # List all features
    feature_cols = [c for c in df.columns if c not in ['date', 'district', 'target', 'us_aqi'] 
                    and c not in POLLUTANT_COLS]
    print(f"\n  Features ({len(feature_cols)}): {feature_cols}")

    return df, feature_cols


# ============================================================
# STEP 4: MODEL TRAINING (Statistical + Ensemble + Boosting)
# ============================================================
def train_models(df, feature_cols):
    """Train multiple ML models and select the best one."""
    print("\n" + "=" * 60)
    print("  STEP 4a: ML MODEL TRAINING")
    print("=" * 60)

    df_train = df.copy()
    df_train['date'] = pd.to_datetime(df_train['date']).dt.tz_localize(None)
    df_train = df_train[df_train['date'] < datetime.now()]
    
    df_abbottabad = df_train[df_train['district'] == 'Abbottabad'].copy()
    print(f"  Training on Abbottabad data: {len(df_abbottabad)} records")

    X = df_abbottabad[feature_cols]
    y = df_abbottabad['target']

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"  Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"\n  {'Model':<30} {'RMSE':<10} {'MAE':<10} {'R²':<10} {'Status'}")
    print("  " + "-" * 70)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=12, n_jobs=-1, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        ),
        "XGBoost": xgb.XGBRegressor(
            objective='reg:squarederror', n_estimators=150,
            learning_rate=0.1, max_depth=6, random_state=42,
            verbosity=0
        ),
    }

    best_model = None
    best_score = float('inf')
    best_name = ""
    all_results = []

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)

            status = ""
            if rmse < best_score:
                best_score = rmse
                best_model = model
                best_name = name
                status = "<-- BEST"

            all_results.append({
                'name': name, 'rmse': rmse, 'mae': mae, 'r2': r2
            })
            print(f"  {name:<30} {rmse:<10.4f} {mae:<10.4f} {r2:<10.4f} {status}")

        except Exception as e:
            print(f"  {name:<30} FAILED: {e}")

    return best_model, best_name, best_score, feature_cols, all_results


# ============================================================
# STEP 4b: LSTM DEEP LEARNING MODEL (TensorFlow/Keras)
# ============================================================
def train_lstm_model(clean_df, all_results, best_score, best_name, best_model):
    """Train LSTM and compare with best ML model."""
    print("\n" + "=" * 60)
    print("  STEP 4b: LSTM DEEP LEARNING MODEL (TensorFlow)")
    print("=" * 60)

    try:
        from src.lstm_model import train_lstm
        _, _, lstm_metrics = train_lstm(clean_df, MODELS_DIR)
        all_results.append(lstm_metrics)

        # Check if LSTM beats the current best
        if lstm_metrics['rmse'] < best_score:
            print(f"\n  🧠 LSTM beats {best_name}! ({lstm_metrics['rmse']:.4f} < {best_score:.4f})")
            best_name = lstm_metrics['name']
            best_score = lstm_metrics['rmse']
            # LSTM is saved separately as .keras, so best_model stays as ML backup
        else:
            print(f"\n  📊 ML model ({best_name}) still better ({best_score:.4f} < {lstm_metrics['rmse']:.4f})")

        print(f"  {lstm_metrics['name']:<30} {lstm_metrics['rmse']:<10.4f} {lstm_metrics['mae']:<10.4f} {lstm_metrics['r2']:<10.4f}")

    except Exception as e:
        print(f"  [LSTM] Training failed: {e}")
        print(f"  [LSTM] Continuing with ML models only.")

    return all_results, best_score, best_name


# ============================================================
# STEP 5: SAVE MODEL ARTIFACTS
# ============================================================
def save_model(model, features, model_name, results):
    """Save the best model and feature list."""
    print("\n" + "=" * 60)
    print("  STEP 5: SAVING MODEL ARTIFACTS")
    print("=" * 60)

    model_path = os.path.join(MODELS_DIR, "model.pkl")
    features_path = os.path.join(MODELS_DIR, "features.pkl")

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  Saved ML model: {model_path}")
    print(f"  Model type: {type(model).__name__}")
    print(f"  Model size: {os.path.getsize(model_path) / 1024:.1f} KB")

    with open(features_path, 'wb') as f:
        pickle.dump(features, f)
    print(f"  Saved features: {features_path}")

    # Check if LSTM model also exists
    lstm_path = os.path.join(MODELS_DIR, "lstm_model.keras")
    if os.path.exists(lstm_path):
        print(f"  LSTM model: {lstm_path} ({os.path.getsize(lstm_path) / 1024:.1f} KB)")

    # Save results summary
    results_df = pd.DataFrame(results)
    results_path = os.path.join(MODELS_DIR, "training_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"  Saved leaderboard: {results_path}")

    print(f"\n  OVERALL WINNER: {model_name}")


def save_shap_importance(model, df, feature_cols):
    """Compute and save SHAP feature importance using the best ML model."""
    print("\n" + "=" * 60)
    print("  STEP 5b: COMPUTING SHAP FEATURE IMPORTANCE")
    print("=" * 60)
    try:
        import shap
        df_abbottabad = df[df['district'] == 'Abbottabad'].copy()
        X = df_abbottabad[feature_cols]
        sample_size = min(200, len(X))
        X_sample = X.sample(n=sample_size, random_state=42)
        
        model_name = type(model).__name__
        print(f"  [SHAP] Computing SHAP values using {model_name}...")
        
        if model_name in ['XGBRegressor', 'RandomForestRegressor', 'GradientBoostingRegressor']:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        else:
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample).values
            
        if hasattr(shap_values, "values"):
            shap_values = shap_values.values
            
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        if len(mean_shap.shape) > 1:
            mean_shap = mean_shap.mean(axis=1)
            
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': mean_shap
        }).sort_values('importance', ascending=False)
        
        importance_path = os.path.join(MODELS_DIR, "shap_importance.csv")
        importance_df.to_csv(importance_path, index=False)
        print(f"  [SHAP] Saved feature importance to: {importance_path}")
    except Exception as e:
        print(f"  [SHAP] Failed to compute/save SHAP: {e}")


# ============================================================
# STEP 6: HOPSWORKS FEATURE STORE (Optional)
# ============================================================
def push_to_hopsworks(feat_df, rmse=None):
    """Upload features and model to Hopsworks if API key is available."""
    print("\n" + "=" * 60)
    print("  STEP 6: HOPSWORKS FEATURE STORE")
    print("=" * 60)

    try:
        from src.feature_store import upload_features, register_model
        
        success = upload_features(feat_df)
        if success:
            metrics = {"rmse": float(rmse)} if rmse is not None else {}
            register_model(MODELS_DIR, metrics=metrics)
        else:
            print("  [Hopsworks] Skipped (no API key or connection failed)")
    except ImportError:
        print("  [Hopsworks] hopsworks package not installed. Skipping.")
    except Exception as e:
        print(f"  [Hopsworks] Error: {e}")


# ============================================================
# MAIN PIPELINE
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  HAZARA DIVISION AQI - COMPLETE PIPELINE")
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Step 1: Collect data from Open-Meteo API
    raw_df = collect_all_data()

    # Step 2: Clean data
    clean_df = clean_data(raw_df)

    # Step 3: Engineer features
    feat_df, feature_cols = engineer_features(clean_df)

    # Step 4a: Train ML models (statistical + ensemble + boosting)
    best_model, best_name, best_score, features, results = train_models(feat_df, feature_cols)

    # Step 4b: Train LSTM deep learning model (TensorFlow)
    results, best_score, best_name = train_lstm_model(clean_df, results, best_score, best_name, best_model)

    print(f"\n  {'='*70}")
    print(f"  OVERALL WINNER: {best_name} (RMSE: {best_score:.4f})")
    print(f"  {'='*70}")

    # Step 5: Save all artifacts
    save_model(best_model, features, best_name, results)
    save_shap_importance(best_model, feat_df, feature_cols)

    # Step 6: Hopsworks Feature Store (optional)
    push_to_hopsworks(feat_df, rmse=best_score)

    print("\n" + "=" * 60)
    print(f"  PIPELINE COMPLETE!")
    print(f"  Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print(f"\n  Files created:")
    print(f"    data/raw_hazara_aqi.csv")
    print(f"    data/cleaned_hazara_aqi.csv")
    print(f"    data/features_hazara_aqi.csv")
    print(f"    models/model.pkl          (best ML model)")
    print(f"    models/features.pkl")
    print(f"    models/shap_importance.csv (feature importance)")
    print(f"    models/lstm_model.keras   (deep learning model)")
    print(f"    models/lstm_scaler.pkl")
    print(f"    models/training_results.csv")
    print()
