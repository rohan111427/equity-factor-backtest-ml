import time

import matplotlib.pyplot as plt
import pandas as pd

from portfolio.backtester import run_backtest
from strategies.monthly_selector import get_monthly_ranked_stocks
# Step 1: Load stock data
from utils.data_loader import (load_all_stocks_from_folder, load_benchmark,
                               load_multiple_stocks)
from utils.metrics import analyze_performance

start = time.time()
tickers = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "KOTAKBANK.NS", "SBIN.NS", "AXISBANK.NS", "HINDUNILVR.NS", "ITC.NS",
    "LT.NS", "MARUTI.NS", "BAJFINANCE.NS", "BHARTIARTL.NS", "ASIANPAINT.NS",
    "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "POWERGRID.NS", "ULTRACEMCO.NS",
    "NTPC.NS", "TITAN.NS", "SUNPHARMA.NS", "DRREDDY.NS", "NESTLEIND.NS",
    "JSWSTEEL.NS", "COALINDIA.NS", "DIVISLAB.NS", "ADANIENT.NS", "HINDALCO.NS"
]
stock_data = load_multiple_stocks(tickers=tickers)

# Limit to first 100 stocks for manageable processing
tickers = list(stock_data.keys())[:100]  # Limit to first 100 stocks
stock_data = {ticker: stock_data[ticker] for ticker in tickers}

# Step 2: Get ranked stocks
ranked_stocks_per_month = get_monthly_ranked_stocks(
    stock_data, start_date="2015-01-01", end_date="2020-12-31", top_n=10
)
for date, picks in list(ranked_stocks_per_month.items())[:5]:
    print(date.date(), "→", picks)
# Step 3: Run backtest
start = time.time()
portfolio_series = run_backtest(stock_data, ranked_stocks_per_month)
print(f"\n✅ Backtest completed in {time.time() - start:.2f} seconds")

# Step 4: Analyze performance
print("\n📊 Analyzing performance...")
metrics = analyze_performance(portfolio_series.dropna(), ranked_stocks_per_month=ranked_stocks_per_month)
for metric, value in metrics.items():
    print(f"{metric}: {value:.2%}" if isinstance(value, float) else f"{metric}: {value}")

# Step 5: Load & normalize benchmark
try:
    nifty = load_benchmark()
    nifty = nifty.astype(str).str.replace(",", "")  # remove commas
    nifty = pd.to_numeric(nifty, errors="coerce")
    nifty = nifty.dropna()
    if nifty.empty:
        raise ValueError("❌ Benchmark data is empty after cleaning.")
    print("✅ Benchmark loaded with", len(nifty), "entries")
except Exception as e:
    print(f"⚠️ Failed to load benchmark: {e}")
    nifty = pd.Series(dtype='float64')  # fallback to empty series

nifty = nifty[nifty.index.isin(portfolio_series.index)]
nifty = nifty / nifty.iloc[0]
portfolio_normalized = portfolio_series / portfolio_series.iloc[0]

from utils.metrics import compare_to_benchmark

benchmark_metrics = compare_to_benchmark(portfolio_series, nifty)
print("\n📊 Benchmark Comparison:")
for k, v in benchmark_metrics.items():
    print(f"{k}: {v:.2%}")
print("✅ Benchmark comparison complete.")

# Resample both to month-end
strategy_monthly = portfolio_series.resample('M').last()
nifty_monthly = nifty.resample('M').last()

# Normalize both
strategy_norm = strategy_monthly / strategy_monthly.iloc[0]
nifty_norm = nifty_monthly / nifty_monthly.iloc[0]
print("📈 Generating plot...")
plt.figure(figsize=(12, 6))
plt.plot(strategy_norm, label="Strategy", linewidth=2)
plt.plot(nifty_norm, label="NIFTY 50", linestyle="--", alpha=0.7)
plt.title("Monthly Portfolio Value vs Benchmark")
plt.xlabel("Date")
plt.ylabel("Normalized Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()