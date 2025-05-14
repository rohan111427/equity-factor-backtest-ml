import pandas as pd

from strategies.factor_builder import compute_all_factors
from strategies.factor_strategy import rank_stocks


def get_monthly_ranked_stocks(stock_data,start_date,end_date,top_n=5):
    dates = pd.date_range(start=start_date,end=end_date,freq='ME')
    ranked_stocks_per_month = {}
    
    for date in dates:
        sliced_data={}
        
        for ticker,df in stock_data.items():
            try:
                df_filtered = df[df.index <= date]
                if len(df_filtered) < 252:
                    continue
                sliced_data[ticker] = df_filtered
            except:
                continue
        if not sliced_data:
            continue
        factor_df = compute_all_factors(sliced_data)
        top_stocks = rank_stocks(factor_df,ascending_factors=['Volatility'],top_n=top_n)
        ranked_stocks_per_month[date] = list(top_stocks.index)
        
    return ranked_stocks_per_month
                