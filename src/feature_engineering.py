import pandas as pd

def create_features(df):
    """
    Generates time-series features for AQI forecasting.
    Includes: Date components, Rolling Averages, and Lag features.
    """
    df = df.copy()
    
    # 1. Date Components
    if 'date' in df.columns:
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
    
    # 2. Rolling Averages
    # Using min_periods=1 to ensure we get values even at the start of the data
    for col in ['pm10', 'pm2_5', 'us_aqi']:
        if col in df.columns:
            df[f'{col}_rolling_24h'] = df[col].rolling(window=24, min_periods=1).mean()
            df[f'{col}_rolling_6h'] = df[col].rolling(window=6, min_periods=1).mean()
    
    # 3. Lag Features
    # Note: Lags introduce NaNs at the beginning which need to be handled downstream
    if 'us_aqi' in df.columns:
        df['lag_1h_aqi'] = df['us_aqi'].shift(1)
        df['lag_2h_aqi'] = df['us_aqi'].shift(2)
        df['lag_24h_aqi'] = df['us_aqi'].shift(24)
    
    # 4. AQI Change Rate (derived feature)
    if 'us_aqi' in df.columns:
        df['aqi_change_rate'] = df['us_aqi'].diff()
        
    return df
