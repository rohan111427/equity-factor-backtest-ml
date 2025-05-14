import pandas as pd


def calculate_volatility(df, window = 90):
    if 'Adj Close' not in df:
        raise ValueError("Adj Close necessary to calculate volatility")
    daily_returns = df['Adj Close'].pct_change()
    volatility = daily_returns.rolling(window=window).std()
    return volatility