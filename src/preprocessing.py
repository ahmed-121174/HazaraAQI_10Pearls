import pandas as pd
import numpy as np
# Import the new function from the dedicated feature engineering module
from src.feature_engineering import create_features

def preprocess_data(df, is_training=True):
    """
    Cleans the data and applies feature engineering.
    - Handles missing values (interpolation/backfill)
    - Handle Outliers
    - Calls feature_engineering module
    """
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # 1. Handle Missing Values
    # Forward fill then backward fill for continuous time series data
    df = df.ffill().bfill()
    
    # 2. Outlier Handling (Clipping to 1st and 99th percentiles) only during training
    # This keeps inference deterministic and based on actual recent data
    if is_training:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'date':
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                df[col] = df[col].clip(lower, upper)

    # 3. Apply Feature Engineering
    # We delegate the creation of lags and rolling windows to the dedicated module
    df = create_features(df)
    
    return df
