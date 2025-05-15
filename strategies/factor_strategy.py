import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from strategies.factor_builder import compute_all_factors


def rank_stocks(factor_df,weights = None,ascending_factors = None, top_n=5):
    """
    Ranks stocks based on multiple factors using optional weights and order direction.

    Parameters:
    - factor_df: pd.DataFrame with rows = factor names, columns = stock tickers
    - ascending_factors: list of factors to rank in ascending order
    - weights: dict of weights for each factor
    - top_n: number of top-ranked stocks to return

    Returns:
    - pd.Series of top N ranked stocks
    """
    if ascending_factors is None:
        ascending_factors = []

    if weights is None:
        weights = {factor: 1.0 / len(factor_df) for factor in factor_df.index}

    total_weight = sum(weights.values())
    normalized_weights = {k: v / total_weight for k, v in weights.items()}

    scores = pd.Series(0.0, index=factor_df.columns)

    for factor in factor_df.index:
        ascending = factor in ascending_factors
        ranked = factor_df.loc[factor].rank(ascending=ascending)
        scores += ranked * normalized_weights.get(factor, 0)

    return scores.sort_values().head(top_n)