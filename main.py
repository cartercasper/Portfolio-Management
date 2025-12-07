import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt
import requests
from data_samples import get_returns 
from estimators import shrinkage_target, MP_est, sample_cov


# data source for tickers
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
headers = {"User-Agent": "Mozilla/5.0"}  # avoid blocking
resp = requests.get(url, headers=headers)
resp.raise_for_status()
df = pd.read_html(resp.text, header=0)[0]
tickers = df["Symbol"].tolist()


# calculate returns based on our ticker list
returns = get_returns(tickers, days_back=1500)

print(returns)

# calculate our estimators: sample covariance matrix, shrinkage target matrix, Marchenko-Pastur matrix
SCM = sample_cov(returns)
ST = shrinkage_target(SCM)
MP = MP_est(returns, SCM)
