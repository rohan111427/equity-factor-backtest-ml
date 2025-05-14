import pandas as pd


def calculate_sharpe_ratio(df, risk_free_return=0.0002):
    
    if 'Close' not in df.columns:
        raise ValueError("'Close' needed for claculating the Sharpe Ratio")
    
    daily_returns = df['Close'].pct_change().dropna()
    excess_returns = daily_returns - risk_free_return
    sharpe_ratio = excess_returns.mean()/excess_returns.std()
    
    return sharpe_ratio