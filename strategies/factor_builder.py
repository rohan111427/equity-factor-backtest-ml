# strategies/compute_factor_scores.py

import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from factors.momentum import calculate_momentum
from factors.sharpe_ratio import calculate_sharpe_ratio
from factors.value import calculate_value_factor
from factors.volatility import calculate_volatility


def compute_all_factors(stock_data):
    """
    stock_data_dict: dict of {ticker: DataFrame}
    returns: DataFrame where rows are factors, columns are tickers
    """
    factor_dict = {
        "momentum": {},
        "volatility": {},
        "value": {},
        "sharpe": {}
    }

    for ticker, df in stock_data.items():
        try:
            # Calculate and store each factor
            factor_dict["momentum"][ticker] = calculate_momentum(df).iloc[-1]
            factor_dict["volatility"][ticker] = calculate_volatility(df).iloc[-1]
            factor_dict["value"][ticker] = calculate_value_factor(df).iloc[-1]
            factor_dict["sharpe"][ticker] = calculate_sharpe_ratio(df)
        except Exception as e:
            print(f"Error calculating factors for {ticker}: {e}")

    # Create DataFrame
    factor_df = pd.DataFrame(factor_dict).T  # Transpose so rows = factors
    return factor_df

