
import pandas as pd

def analyze_performance(portfolio_series, risk_free_rate=0.0002):
    """
    Analyzes the performance of a portfolio value series.

    Parameters:
    - portfolio_series (pd.Series): Series of portfolio value indexed by date
    - risk_free_rate (float): Monthly risk-free rate (default = 0.02% annual ≈ 0.0002 monthly)

    Returns:
    - dict: Dictionary containing cumulative return, volatility, Sharpe ratio, and max drawdown
    """
    returns = portfolio_series.pct_change().dropna()

    # Cumulative return over the full period
    cumulative_return = (portfolio_series.iloc[-1] / portfolio_series.iloc[0]) - 1

    # Volatility = standard deviation of monthly returns
    volatility = returns.std()

    # Sharpe Ratio = (mean excess return) / volatility
    sharpe_ratio = (returns.mean() - risk_free_rate) / volatility if volatility != 0 else 0

    # Max Drawdown = worst drop from peak to trough
    rolling_max = portfolio_series.cummax()
    drawdown = portfolio_series / rolling_max - 1
    max_drawdown = drawdown.min()

    return {
        "Cumulative Return": cumulative_return,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown
    }