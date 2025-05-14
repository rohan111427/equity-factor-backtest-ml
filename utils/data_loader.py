import os

import pandas as pd


def load_stock_data(ticker, data_dir = "data"):
    file_path = os.path.join(data_dir, f"{ticker}.csv")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found for ticker: {ticker}")
    
    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
    df=df.sort_index()
    return df

def load_multiple_stocks(tickers,data_dir="data"):
    stock_data = {}
    for ticker in tickers:
        try:
            stock_data[ticker]  =load_stock_data(ticker, data_dir)
        except FileNotFoundError as e:
            print(e) 

    return stock_data
        