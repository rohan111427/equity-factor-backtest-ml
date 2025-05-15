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
        
def load_benchmark(file_path="data/NIFTY_50.csv"):
    import pandas as pd

    # Load with flexible separator and encoding
    df = pd.read_csv(file_path, encoding="utf-8", sep=None, engine="python")

    # Clean column names
    df.columns = df.columns.str.strip().str.replace('"', '').str.replace('\ufeff', '').str.lower()

    # Dynamically find date and close/price columns
    date_col = next((col for col in df.columns if "date" in col), None)
    close_col = next((col for col in df.columns if "close" in col or "price" in col), None)

    if not date_col or not close_col:
        raise ValueError(f"Required columns not found. Columns available: {df.columns.tolist()}")

    # Parse and clean data
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[date_col, close_col])
    df = df[[date_col, close_col]].rename(columns={date_col: "Date", close_col: "Close"})
    df.set_index("Date", inplace=True)

    return df["Close"]

def load_all_stocks_from_folder(data_dir="data"):
    stock_data = {}
    all_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    total = len(all_files)

    for i, file in enumerate(all_files, 1):
        ticker = file.replace(".csv", "")
        file_path = os.path.join(data_dir, file)

        print(f"[{i}/{total}] Loading {ticker}...")

        try:
            df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
            df = df.sort_index()
            stock_data[ticker] = df
        except Exception as e:
            print(f"❌ Failed to load {ticker}: {e}")
    
    print(f"\n✅ Finished loading {len(stock_data)} of {total} files.")
    return stock_data

try:
    nifty = load_benchmark("data/NIFTY_50.csv")
    print("✅ Benchmark loaded successfully!")
    print(nifty.head())  # show first 5 rows
except Exception as e:
    print("❌ Error loading benchmark:", e)