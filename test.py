import os

from strategies.factor_builder import compute_all_factors
from strategies.factor_strategy import rank_stocks
from utils.data_loader import load_multiple_stocks

# Step 1: Choose stock tickers that match your CSV file names
tickers = ["INFY.NS", "TCS.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"]

# Step 2: Load their data
stock_data = load_multiple_stocks(tickers, data_dir="data")

# Step 3: Compute factors
factor_df = compute_all_factors(stock_data)
print("Factor Scores:\n", factor_df)

# Step 4: Rank stocks
ascending_factors = ["volatility"]  # lower is better
top_stocks = rank_stocks(factor_df, ascending_factors=ascending_factors, top_n=3)
print("\n🏆 Top Ranked Stocks:\n", top_stocks)