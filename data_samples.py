import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt
import requests



##################################
# Can add other data by following the same pipelien (synthetic, other real data) - to be implemented later
##################################

###########################################################################################


def get_returns(
    tickers: list = None,
    days_back: int = 500,
    batch_size: int = 50,
    min_data_ratio: float = 0.9
) -> pd.DataFrame:

    
    # Clean tickers for yfinance (BRK.B -> BRK-B)
    tickers = [t.replace(".", "-") for t in tickers]
    
    # date range
    end = dt.date.today()
    start = end - dt.timedelta(days=days_back)
    
    # download in batches
    all_data = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"Downloading batch {i} to {i + len(batch)}")
        data = yf.download(batch, start=start, end=end)
        
        # use Adj Close if available, else fallback to Close
        if isinstance(data.columns, pd.MultiIndex):
            if "Adj Close" in data.columns.levels[0]:
                data = data["Adj Close"]
            else:
                data = data["Close"]
        all_data.append(data)
    
    
    adj_close = pd.concat(all_data, axis=1)
    
    # fill missing values
    adj_close = adj_close.fillna(method="ffill").fillna(method="bfill")
    
    # drop tickers with too much missing data
    min_non_nan = int(min_data_ratio * len(adj_close))
    adj_clean = adj_close.dropna(axis=1, thresh=min_non_nan)
    


    # compute returns
    returns = adj_clean.pct_change().dropna()
    

    return returns



##########################################################################################


