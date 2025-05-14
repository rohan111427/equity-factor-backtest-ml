from portfolio.backtester import run_backtest
from strategies.monthly_selector import get_monthly_ranked_stocks
from utils.data_loader import load_multiple_stocks
from utils.metrics import analyze_performance

# Step 1: Load stock data
tickers = [
    "INFY.NS", "TCS.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS",
    "LTIM.NS", "MINDTREE.NS", "PERSISTENT.NS", "COFORGE.NS", "MPHASIS.NS"
]
stock_data = load_multiple_stocks(tickers)

# Step 2: Get monthly top-N ranked stocks
ranked_stocks_per_month = get_monthly_ranked_stocks(
    stock_data, start_date="2015-01-01", end_date="2019-12-31", top_n=5
)

# Step 3: Run backtest
portfolio_series = run_backtest(stock_data, ranked_stocks_per_month)

# Step 4: Analyze performance
metrics = analyze_performance(portfolio_series.dropna(), ranked_stocks_per_month=ranked_stocks_per_month)
for metric, value in metrics.items():
    print(f"{metric}: {value}")