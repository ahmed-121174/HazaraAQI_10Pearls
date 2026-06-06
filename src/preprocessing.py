import pandas as pd
import numpy as np
# Import feature engineering function
from src.feature_engineering import create_features

def preprocess_data(df, is_training=True):
    """
    Cleans raw data and runs feature engineering steps.

    Handles missing values and clips outliers.
    """
    # Avoid SettingWithCopyWarning
    df = df.copy()
    
    # Handle missing values
    # Forward and backward fill for time series
    df = df.ffill().bfill()
    
    # Clip outliers to 1st and 99th percentiles during training
    if is_training:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'date':
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                df[col] = df[col].clip(lower, upper)

    # Generate engineered features
    df = create_features(df)
    
    return df
