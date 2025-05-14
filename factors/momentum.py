import pandas as pd


def calculate_momentum(df,lookback_months = 12, skip_recent_months=1):
    if 'Adj Close' not in df:
        raise ValueError("Adj Close expected in the data frame")
    monthly_price = df['Adj Close'].resample('ME').last()
    past = monthly_price.shift(lookback_months)
    recent = monthly_price.shift(skip_recent_months)
    momentum = (recent/past)-1
    
    return momentum