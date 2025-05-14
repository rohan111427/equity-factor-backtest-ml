import os
import time

import yfinance as yf

tickers = ["INFY.NS", "TCS.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"]
start_date = "2010-01-01"
end_date = "2024-12-31"

save_dir = "data"
os.makedirs(save_dir, exist_ok=True)

for ticker in tickers:
    print(f"Downloading {ticker}...")
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if not df.empty:
            df.to_csv(os.path.join(save_dir, f"{ticker}.csv"))
        else:
            print(f"No data found for {ticker}")
    except Exception as e:
        print(f"Failed to download {ticker}: {e}")
    
    time.sleep(5)  # ⏱ Add delay of 5 seconds per ticker