import pandas as pd

def create_features(df):
    """
    Generate date, rolling average, lag, and change rate features.
    """
    df = df.copy()
    
    # Date components
    if 'date' in df.columns:
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
    
    # Rolling averages
    # min_periods=1 ensures values at start of series
    for col in ['pm10', 'pm2_5', 'us_aqi']:
        if col in df.columns:
            df[f'{col}_rolling_24h'] = df[col].rolling(window=24, min_periods=1).mean()
            df[f'{col}_rolling_6h'] = df[col].rolling(window=6, min_periods=1).mean()
    
    # Lag features
    if 'us_aqi' in df.columns:
        df['lag_1h_aqi'] = df['us_aqi'].shift(1)
        df['lag_2h_aqi'] = df['us_aqi'].shift(2)
        df['lag_24h_aqi'] = df['us_aqi'].shift(24)
    
    # AQI change rate
    if 'us_aqi' in df.columns:
        df['aqi_change_rate'] = df['us_aqi'].diff()
        
    return df
