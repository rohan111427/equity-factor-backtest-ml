import pandas as pd


def calculate_value_factor(df, lookback_period = 252):
    if 'Close' not in df.columns:
        raise ValueError("'Close' must be present in order to calculate value")
    
    trailing_median = df['Close'].rolling(window=lookback_period).median()
    value_score = df['Close']/trailing_median
    
    return value_score
    