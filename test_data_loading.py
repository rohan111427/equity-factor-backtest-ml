from utils.data_loader import load_multiple_stocks

tickers = ["INFY.NS", "TCS.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"]

stock_data = load_multiple_stocks(tickers)

for ticker, df in stock_data.items():
    print(f"\n{ticker} Data Sample:")
    print(df.head())