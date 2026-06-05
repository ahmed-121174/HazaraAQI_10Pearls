import sys
import os

# --- PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
# ----------------

import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import datetime, timedelta

# Hazara Division District Coordinates
HAZARA_DISTRICTS = {
    "Abbottabad":      {"lat": 34.1463, "lon": 73.2117},
    "Mansehra":        {"lat": 34.3302, "lon": 73.1968},
    "Haripur":         {"lat": 33.9942, "lon": 72.9333},
    "Battagram":       {"lat": 34.6837, "lon": 73.0261},
    "Upper Kohistan":  {"lat": 35.2097, "lon": 73.3456},
    "Torghar":         {"lat": 34.6300, "lon": 72.9400},
}

def fetch_district_data(district_name, latitude, longitude):
    """
    Fetches historical + short-term forecast data for a single district
    from the Open-Meteo Air Quality API.
    """
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    end_date = datetime.now().date() + timedelta(days=4)
    start_date = datetime.now().date() - timedelta(days=6*30)

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
                    "sulphur_dioxide", "ozone", "us_aqi"]
    }

    print(f"  Fetching {district_name} ({latitude}, {longitude})...")
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    hourly = response.Hourly()
    hourly_data = {"date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )}

    for i, col in enumerate(["pm10", "pm2_5", "carbon_monoxide",
                              "nitrogen_dioxide", "sulphur_dioxide", "ozone", "us_aqi"]):
        hourly_data[col] = hourly.Variables(i).ValuesAsNumpy()

    df = pd.DataFrame(data=hourly_data)
    df['district'] = district_name
    return df


def fetch_all_districts():
    """
    Fetches air quality data for all Hazara Division districts.
    Returns a combined DataFrame with a 'district' column.
    """
    all_data = []
    for name, coords in HAZARA_DISTRICTS.items():
        try:
            df = fetch_district_data(name, coords['lat'], coords['lon'])
            all_data.append(df)
            print(f"  ✅ {name}: {len(df)} records fetched.")
        except Exception as e:
            print(f"  ❌ {name}: Failed - {e}")

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


if __name__ == "__main__":
    print("=" * 50)
    print("  Hazara Division AQI Data Ingestion")
    print("=" * 50)
    print(f"\nFetching data for {len(HAZARA_DISTRICTS)} districts...\n")

    df = fetch_all_districts()
    print(f"\n✅ Total: {len(df)} records fetched.")

    if not df.empty:
        data_dir = os.path.join(project_root, "data")
        os.makedirs(data_dir, exist_ok=True)
        csv_path = os.path.join(data_dir, "raw_hazara_aqi.csv")
        df.to_csv(csv_path, index=False)
        print(f"✅ Data saved to {csv_path}")

        # Upload to Hopsworks Feature Store (if configured)
        try:
            from src.feature_store import upload_features
            upload_features(df)
        except Exception:
            pass
    else:
        print("❌ No data was fetched. Check API connectivity.")
