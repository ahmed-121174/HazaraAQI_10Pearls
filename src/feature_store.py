"""
Hopsworks feature store integration.
Handles saving and retrieving features and model registry.
"""

import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "HazaraAQI")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")

def get_hopsworks_connection():
    """Connect to Hopsworks API."""
    if not HOPSWORKS_API_KEY:
        print("  [Hopsworks] HOPSWORKS_API_KEY not set. Skipping Feature Store.")
        return None

    try:
        import hopsworks
        project = hopsworks.login(
            host=HOPSWORKS_HOST,
            api_key_value=HOPSWORKS_API_KEY,
            project=HOPSWORKS_PROJECT
        )
        print(f"  [Hopsworks] Connected to project: {HOPSWORKS_PROJECT}")
        return project
    except Exception as e:
        print(f"  [Hopsworks] Connection failed: {e}")
        return None


def upload_features(df, feature_group_name="hazara_aqi_features", version=1):
    """
    Upload engineered features to Hopsworks.
    """
    project = get_hopsworks_connection()
    if project is None:
        return False

    try:
        fs = project.get_feature_store()

        # Add event_time column
        upload_df = df.copy()
        if 'date' in upload_df.columns:
            upload_df['event_time'] = pd.to_datetime(upload_df['date'])
        
        # Drop non-numeric columns
        drop_cols = [c for c in upload_df.columns if upload_df[c].dtype == 'object' and c != 'district']
        upload_df = upload_df.drop(columns=drop_cols, errors='ignore')

        fg = fs.get_or_create_feature_group(
            name=feature_group_name,
            version=version,
            description="Hazara Division AQI features for 6 districts",
            primary_key=["district", "date"],
            event_time="event_time",
        )

        fg.insert(upload_df, write_options={"wait_for_job": True})
        print(f"  [Hopsworks] Uploaded {len(upload_df)} records to '{feature_group_name}' v{version}")
        return True

    except Exception as e:
        print(f"  [Hopsworks] Upload failed: {e}")
        return False


def get_features(feature_group_name="hazara_aqi_features", version=1, district=None):
    """
    Get features from Hopsworks.
    """
    project = get_hopsworks_connection()
    if project is None:
        return None

    try:
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name=feature_group_name, version=version)

        query = fg.select_all()
        df = query.read()

        if district and 'district' in df.columns:
            df = df[df['district'] == district]

        print(f"  [Hopsworks] Retrieved {len(df)} records from '{feature_group_name}'")
        return df

    except Exception as e:
        print(f"  [Hopsworks] Retrieval failed: {e}")
        return None


def register_model(model_path, model_name="hazara_aqi_model", metrics=None):
    """
    Save trained model in registry.
    """
    project = get_hopsworks_connection()
    if project is None:
        return False

    try:
        mr = project.get_model_registry()

        model_entry = mr.python.create_model(
            name=model_name,
            metrics=metrics or {},
            description="Best AQI prediction model for Hazara Division"
        )

        model_entry.save(model_path)
        print(f"  [Hopsworks] Model '{model_name}' registered successfully")
        return True

    except Exception as e:
        print(f"  [Hopsworks] Model registration failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("  Hopsworks Feature Store Test")
    print("=" * 50)

    project = get_hopsworks_connection()
    if project:
        print("  Connection successful!")
        # Upload a small sample to test
        sample = pd.DataFrame({
            'date': pd.date_range('2026-01-01', periods=5, freq='h'),
            'district': 'Abbottabad',
            'us_aqi': [50, 55, 60, 58, 52],
            'pm2_5': [30, 32, 35, 33, 31],
        })
        upload_features(sample, feature_group_name="hazara_aqi_test", version=1)
    else:
        print("  Set HOPSWORKS_API_KEY in .env to enable Feature Store.")
