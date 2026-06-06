"""
FastAPI backend for Hazara AQI dashboard.
Defines endpoints for fetching current air quality index and forecasting.
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.preprocessing import preprocess_data

# App Setup
app = FastAPI(
    title="Hazara Division AQI Intelligence",
    description="Air Quality Prediction API for Hazara Division, KPK",
    version="1.0.0"
)

templates_dir = os.path.join(current_dir, "templates")
static_dir = os.path.join(current_dir, "static")
os.makedirs(templates_dir, exist_ok=True)
os.makedirs(static_dir, exist_ok=True)

templates = Jinja2Templates(directory=templates_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.on_event("startup")
async def startup_event():
    print("  [FastAPI] Pre-loading model and data from disk at startup...")
    load_model()
    load_data()
    print("  [FastAPI] Startup initialization complete.")


# Constants
HAZARA_DISTRICTS = [
    "Abbottabad", "Mansehra", "Haripur",
    "Battagram", "Upper Kohistan", "Torghar"
]

MODEL_PATH = os.path.join(project_root, "models", "model.pkl")
FEATURES_PATH = os.path.join(project_root, "models", "features.pkl")
DATA_PATH = os.path.join(project_root, "data", "cleaned_hazara_aqi.csv")
RESULTS_PATH = os.path.join(project_root, "models", "training_results.csv")
SHAP_PATH = os.path.join(project_root, "models", "shap_importance.csv")


# Global Cache
_cached_model = None
_cached_features = None
_cached_model_name = "Unknown"
_cached_model_mtime = 0

_cached_data = None
_cached_data_mtime = 0
_cached_forecasts = {}


# Helper Functions
def load_model():
    """Load model and features from local disk."""
    global _cached_model, _cached_features, _cached_model_name, _cached_model_mtime
    
    try:
        current_mtime = os.path.getmtime(MODEL_PATH)
    except Exception:
        current_mtime = 0

    if _cached_model is not None and current_mtime == _cached_model_mtime:
        return _cached_model, _cached_features, _cached_model_name

    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(FEATURES_PATH, "rb") as f:
            features = pickle.load(f)
        _cached_model_mtime = current_mtime
        print(f"  [FastAPI] Loaded fresh model from {MODEL_PATH} (mtime: {current_mtime})")
    except Exception as e:
        print(f"  [FastAPI] Local model load failed: {e}")
        return None, None, "Unknown"

    name = type(model).__name__
    name_map = {
        "LinearRegression": "Linear Regression",
        "Ridge": "Ridge Regression",
        "XGBRegressor": "XGBoost",
        "RandomForestRegressor": "Random Forest",
        "GradientBoostingRegressor": "Gradient Boosting",
        "Sequential": "LSTM (Deep Learning)",
    }
    
    model_name = name_map.get(name, name)
    _cached_model = model
    _cached_features = features
    _cached_model_name = model_name
    return _cached_model, _cached_features, _cached_model_name


def load_data(district=None):
    """Load cleaned data from local CSV."""
    global _cached_data, _cached_data_mtime, _cached_forecasts
    
    try:
        current_mtime = os.path.getmtime(DATA_PATH)
    except Exception:
        current_mtime = 0

    if _cached_data is not None and current_mtime == _cached_data_mtime:
        df = _cached_data
    else:
        try:
            df = pd.read_csv(DATA_PATH, parse_dates=["date"])
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], utc=True)
            _cached_data = df
            _cached_data_mtime = current_mtime
            _cached_forecasts.clear()
            print(f"  [FastAPI] Loaded fresh data from {DATA_PATH} (mtime: {current_mtime})")
        except Exception as e:
            print(f"  [FastAPI] Local data load failed: {e}")
            return pd.DataFrame()

    if district and "district" in df.columns:
        df = df[df["district"] == district]
    return df.sort_values("date")


def get_aqi_category(aqi):
    """Return (label, color, advice) for AQI value."""
    if aqi <= 50:
        return "Good", "#22c55e", "Air quality is satisfactory."
    elif aqi <= 100:
        return "Moderate", "#eab308", "Acceptable. Sensitive groups should limit outdoor exertion."
    elif aqi <= 150:
        return "Unhealthy (Sensitive)", "#f97316", "Sensitive groups may experience health effects."
    elif aqi <= 200:
        return "Unhealthy", "#ef4444", "Everyone may begin to experience health effects."
    elif aqi <= 300:
        return "Very Unhealthy", "#a855f7", "Health alert! Avoid outdoor activities."
    else:
        return "Hazardous", "#78716c", "Emergency conditions. Stay indoors."


def generate_forecast(model, features, recent_data, full_df=None, hours=72):
    """Generate forecast for given hours."""
    predictions = []
    history = recent_data.tail(100).copy()

    for i in range(hours):
        proc = preprocess_data(history.copy(), is_training=False)
        row = proc.iloc[-1:]
        for feat in features:
            if feat not in row.columns:
                row[feat] = 0
        X = row[features]

        try:
            raw = model.predict(X)[0]
        except Exception:
            raw = float(model.predict(X))

        prev = history["us_aqi"].iloc[-1]
        pred = np.clip(raw, prev * 0.85, prev * 1.15)
        pred = np.clip(pred, 0, 500)

        next_dt = history["date"].iloc[-1] + pd.Timedelta(hours=1)
        
        # Look up the actual future row in full_df to update variables like pm2_5, pm10, etc.
        if full_df is not None and not full_df.empty:
            future_match = full_df[full_df["date"] == next_dt]
            if not future_match.empty:
                new_row = future_match.iloc[0:1].copy()
            else:
                new_row = history.iloc[-1:].copy()
                new_row["date"] = next_dt
        else:
            new_row = history.iloc[-1:].copy()
            new_row["date"] = next_dt

        new_row["us_aqi"] = pred
        history = pd.concat([history, new_row], ignore_index=True)
        
        # Convert timezone to local Pakistan time for display/API output
        local_dt = next_dt.tz_convert("Asia/Karachi")
        predictions.append({
            "date": local_dt.isoformat(),
            "hour": i + 1,
            "predicted_aqi": round(float(pred), 1)
        })

    return predictions


# API Endpoints

@app.head("/", include_in_schema=False)
async def head_dashboard():
    return JSONResponse(status_code=200, content={})

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, district: str = "Abbottabad"):
    """Serve the main dashboard HTML page."""
    df = load_data(district)
    model, features, model_name = load_model()

    current_aqi = 0
    category = ("N/A", "#666", "No data")
    last_updated = "N/A"
    pm25 = 0
    forecast = []
    daily_forecast = []
    history_chart = []

    if not df.empty:
        # Filter data to only include records up to now
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        past_df = df[df["date"] <= now]
        if past_df.empty:
            past_df = df
            
        latest = past_df.iloc[-1]
        current_aqi = int(latest.get("us_aqi", 0))
        category = get_aqi_category(current_aqi)
        
        # Convert last updated to local Pakistan timezone for display
        local_updated = pd.Timestamp(latest["date"]).tz_convert("Asia/Karachi")
        last_updated = local_updated.strftime("%Y-%m-%d %H:%M:%S")
        
        pm25 = round(float(latest.get("pm2_5", 0)), 1)

        # History for chart (last 48 hours) - convert to local Pakistan timezone
        hist_48 = past_df.tail(48)
        history_chart = [
            {
                "date": pd.Timestamp(r["date"]).tz_convert("Asia/Karachi").isoformat(),
                "aqi": round(float(r["us_aqi"]), 1)
            }
            for _, r in hist_48.iterrows()
        ]

        # Forecast - we generate 96 hours to cover 3 full local days after today
        if model and features:
            cache_key = (district, 96, str(latest["date"]))
            if cache_key in _cached_forecasts:
                forecast = _cached_forecasts[cache_key]
            else:
                recent = past_df.tail(100).copy()
                if "district" in recent.columns:
                    recent = recent.drop(columns=["district"])
                try:
                    forecast = generate_forecast(model, features, recent, full_df=df, hours=96)
                    _cached_forecasts[cache_key] = forecast
                except Exception as e:
                    print(f"Forecast error: {e}")
                    forecast = []
            
            if forecast:
                try:
                    # Daily averages for 3 day cards using the full 96h forecast
                    fc_df = pd.DataFrame(forecast)
                    fc_df["date_parsed"] = pd.to_datetime(fc_df["date"])
                    fc_df["date_local"] = fc_df["date_parsed"].dt.tz_convert("Asia/Karachi")
                    fc_df["day"] = fc_df["date_local"].dt.date
                    today = pd.Timestamp.now(tz="Asia/Karachi").date()
                    fc_df = fc_df[fc_df["day"] > today]
                    daily = fc_df.groupby("day")["predicted_aqi"].mean().head(3)
                    for day, avg in daily.items():
                        label, color, _ = get_aqi_category(int(avg))
                        daily_forecast.append({
                            "day_name": pd.Timestamp(day).strftime("%A, %d %b"),
                            "avg_aqi": int(round(avg)),
                            "color": color,
                            "label": label,
                        })
                except Exception as e:
                    print(f"Forecast parsing error: {e}")

    # Model results
    model_results = []
    if os.path.exists(RESULTS_PATH):
        try:
            res = pd.read_csv(RESULTS_PATH)
            for _, r in res.iterrows():
                model_results.append({
                    "name": r.get("name", "Unknown"),
                    "rmse": round(float(r.get("test_rmse", r.get("rmse", 0))), 4),
                    "r2": round(float(r.get("r2", 0)), 4),
                })
            model_results = sorted(model_results, key=lambda x: x["rmse"])
        except Exception:
            pass

    # SHAP feature importance results
    shap_importance = []
    if os.path.exists(SHAP_PATH):
        try:
            shap_df = pd.read_csv(SHAP_PATH)
            for _, r in shap_df.iterrows():
                shap_importance.append({
                    "feature": r.get("feature", "Unknown"),
                    "importance": round(float(r.get("importance", 0)), 4)
                })
        except Exception:
            pass

    context = {
        "request": request,
        "districts": HAZARA_DISTRICTS,
        "selected": district,
        "current_aqi": current_aqi,
        "cat_label": category[0],
        "cat_color": category[1],
        "cat_advice": category[2],
        "last_updated": last_updated,
        "pm25": pm25,
        "model_name": model_name,
        "daily_forecast": daily_forecast,
        "forecast_json": forecast[:72],
        "history_json": history_chart,
        "model_results": model_results,
        "shap_importance": shap_importance,
        "alert": current_aqi > 100,
    }
    return templates.TemplateResponse(request, "dashboard.html", context)


@app.get("/api/current")
async def api_current(district: str = "Abbottabad"):
    """Get current AQI for a district."""
    df = load_data(district)
    if df.empty:
        return JSONResponse({"error": "No data"}, status_code=404)
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    past_df = df[df["date"] <= now]
    if past_df.empty:
        past_df = df
    latest = past_df.iloc[-1]
    aqi = int(latest.get("us_aqi", 0))
    cat = get_aqi_category(aqi)
    return {
        "district": district,
        "aqi": aqi,
        "category": cat[0],
        "color": cat[1],
        "advice": cat[2],
        "pm25": round(float(latest.get("pm2_5", 0)), 1),
        "pm10": round(float(latest.get("pm10", 0)), 1),
        "timestamp": str(latest["date"]),
    }


@app.get("/api/forecast")
async def api_forecast(district: str = "Abbottabad", hours: int = 72):
    """Get n-hour AQI forecast."""
    model, features, name = load_model()
    if not model:
        return JSONResponse({"error": "Model not loaded"}, status_code=500)
    df = load_data(district)
    if df.empty:
        return JSONResponse({"error": "No data"}, status_code=404)
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    past_df = df[df["date"] <= now]
    if past_df.empty:
        past_df = df
    latest = past_df.iloc[-1]
    cache_key = (district, hours, str(latest["date"]))
    if cache_key in _cached_forecasts:
        preds = _cached_forecasts[cache_key]
    else:
        recent = past_df.tail(100).copy()
        if "district" in recent.columns:
            recent = recent.drop(columns=["district"])
        preds = generate_forecast(model, features, recent, full_df=df, hours=min(hours, 168))
        _cached_forecasts[cache_key] = preds
    return {"district": district, "model": name, "forecast": preds}


@app.get("/api/districts")
async def api_districts():
    """List all available districts."""
    return {"districts": HAZARA_DISTRICTS}


@app.get("/api/history")
async def api_history(district: str = "Abbottabad", hours: int = 168):
    """Get historical AQI data for a district."""
    df = load_data(district)
    if df.empty:
        return JSONResponse({"error": "No data"}, status_code=404)
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    past_df = df[df["date"] <= now]
    if past_df.empty:
        past_df = df
    past_df = past_df.tail(hours)
    data = [
        {
            "date": pd.Timestamp(r["date"]).tz_convert("Asia/Karachi").isoformat(),
            "aqi": round(float(r["us_aqi"]), 1),
            "pm25": round(float(r.get("pm2_5", 0)), 1)
        }
        for _, r in past_df.iterrows()
    ]
    return {"district": district, "records": len(data), "data": data}


@app.get("/api/models")
async def api_models():
    """Get model performance leaderboard."""
    model, _, name = load_model()
    results = []
    if os.path.exists(RESULTS_PATH):
        try:
            res = pd.read_csv(RESULTS_PATH)
            results = res.to_dict("records")
        except Exception:
            pass
    return {"active_model": name, "results": results}
