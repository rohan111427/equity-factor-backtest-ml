import pandas as pd


def calculate_sharpe_ratio(df, risk_free_return=0.07365):  # Annual risk-free rate (India, 7.365%)
    if 'Close' not in df.columns:
        raise ValueError("'Close' needed for calculating the Sharpe Ratio")
    
    risk_free_rate_monthly = (1 + risk_free_return)**(1/12) - 1
    daily_returns = df['Close'].pct_change(fill_method=None).dropna()
    excess_returns = daily_returns - risk_free_rate_monthly
    sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * (12 ** 0.5)
    return sharpe_ratio