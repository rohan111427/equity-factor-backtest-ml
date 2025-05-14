import pandas as pd


def analyze_performance(portfolio_series, risk_free_rate=0.0002, ranked_stocks_per_month=None):
    """
    Analyzes the performance of a portfolio value series.

    Parameters:
    - portfolio_series (pd.Series): Series of portfolio value indexed by date
    - risk_free_rate (float): Monthly risk-free rate (default = 0.02% annual ≈ 0.0002 monthly)
    - ranked_stocks_per_month (dict): Optional, used to compute turnover (keys: Timestamps, values: list of tickers)

    Returns:
    - dict: Dictionary containing cumulative return, volatility, Sharpe ratio, max drawdown, and turnover
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

    # Turnover = % of portfolio that changes each month (avg)
    turnover = None
    if ranked_stocks_per_month is not None:
        months = sorted(ranked_stocks_per_month.keys())
        changes = []
        for i in range(1, len(months)):
            prev = set(ranked_stocks_per_month[months[i - 1]])
            curr = set(ranked_stocks_per_month[months[i]])
            diff = prev.symmetric_difference(curr)
            change_ratio = len(diff) / max(len(prev.union(curr)), 1)
            changes.append(change_ratio)
        turnover = sum(changes) / len(changes) if changes else 0

    return {
        "Cumulative Return": cumulative_return,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Turnover": turnover
    }