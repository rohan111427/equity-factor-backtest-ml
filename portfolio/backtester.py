import pandas as pd


def run_backtest(stock_data, ranked_stocks_per_month, capital=1_000_000):
    """
    Simulates a simple monthly-rebalanced long-only backtest.

    Parameters:
    - stock_data: dict of {ticker: price DataFrame}
    - ranked_stocks_per_month: dict {month-end Timestamp: [top N tickers]}
    - capital: starting portfolio value (default: 1,000,000)

    Returns:
    - pd.Series: portfolio value over time
    """
    portfolio_value = {}
    current_cash = capital

    for date, tickers in ranked_stocks_per_month.items():
        returns = []

        for ticker in tickers:
            df = stock_data.get(ticker)
            if df is None or date not in df.index:
                continue

            try:
                entry_price = df.loc[date]['Close']

                # Determine sell date
                next_month = date + pd.DateOffset(months=1)
                next_month_data = df[df.index >= next_month]
                if next_month_data.empty:
                    continue
                exit_price = next_month_data.iloc[0]['Close']

                # Calculate return
                ret = exit_price / entry_price
                returns.append(ret)

            except Exception as e:
                print(f"Error processing {ticker} on {date}: {e}")

        if returns:
            avg_return = sum(returns) / len(returns)
            current_cash *= avg_return
            portfolio_value[date] = current_cash

    return pd.Series(portfolio_value)