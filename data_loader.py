"""
Data Loading Module

Functions for downloading and processing financial data for portfolio optimization.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime


def get_sp500_tickers(num_stocks=100, random_sample=True, seed=42):
    """
    Download S&P 500 tickers from Wikipedia.
    
    Parameters:
        num_stocks: Number of stocks to return
        random_sample: If True, randomly sample stocks; otherwise take first N
        seed: Random seed for reproducibility
    
    Returns:
        tickers: List of ticker symbols
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    table = pd.read_html(resp.text)[0]
    tickers = table['Symbol'].str.replace('.', '-', regex=False).tolist()
    
    if random_sample and num_stocks < len(tickers):
        np.random.seed(seed)
        tickers = np.random.choice(tickers, num_stocks, replace=False).tolist()
    else:
        tickers = tickers[:num_stocks]
    
    return tickers


def download_prices(tickers, start_date='2021-01-01', end_date=None, verbose=True):
    """
    Download adjusted closing prices for given tickers.
    
    Parameters:
        tickers: List of ticker symbols
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD), None for current date
        verbose: Print progress information
    
    Returns:
        prices: DataFrame of adjusted closing prices
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    if verbose:
        print(f"Downloading data for {len(tickers)} assets...")
    
    data = yf.download(tickers, start=start_date, end=end_date, 
                       progress=verbose, auto_adjust=True)['Close']
    
    # Handle single ticker case
    if isinstance(data, pd.Series):
        data = data.to_frame()
    
    # Remove stocks with missing data
    data = data.dropna(axis=1, how='any')
    
    if verbose:
        print(f"Successfully loaded: {len(data.columns)} assets, {len(data)} days")
    
    return data


def calculate_returns(prices, method='simple'):
    """
    Calculate returns from price data.
    
    Parameters:
        prices: DataFrame of prices
        method: 'simple' for simple returns, 'log' for log returns
    
    Returns:
        returns: DataFrame of returns
    """
    if method == 'simple':
        returns = prices.pct_change().dropna()
    elif method == 'log':
        returns = np.log(prices / prices.shift(1)).dropna()
    else:
        raise ValueError(f"Unknown method: {method}. Use 'simple' or 'log'.")
    
    return returns


def download_returns(tickers, start_date='2021-01-01', end_date=None, 
                     method='simple', verbose=True):
    """
    Download prices and convert to returns in one step.
    
    Parameters:
        tickers: List of ticker symbols
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD), None for current date
        method: 'simple' for simple returns, 'log' for log returns
        verbose: Print progress information
    
    Returns:
        returns: DataFrame of daily returns
    """
    prices = download_prices(tickers, start_date, end_date, verbose)
    returns = calculate_returns(prices, method)
    
    if verbose:
        print(f"Returns shape: {returns.shape}")
        print(f"Date range: {returns.index[0]} to {returns.index[-1]}")
    
    return returns


def load_custom_data(filepath, date_column=None, parse_dates=True):
    """
    Load custom CSV data file.
    
    Parameters:
        filepath: Path to CSV file
        date_column: Name of date column to use as index
        parse_dates: Whether to parse dates
    
    Returns:
        data: DataFrame with data
    """
    if date_column:
        data = pd.read_csv(filepath, index_col=date_column, parse_dates=parse_dates)
    else:
        data = pd.read_csv(filepath, index_col=0, parse_dates=parse_dates)
    
    return data


# ============================================================================
# Market Index Datasets (matching paper's Table I and II)
# ============================================================================

# Dataset configurations for paper replication
DATASETS = {
    'S&P 500 (USA)': {
        'source': 'sp500',
        'description': 'S&P 500 Index constituents',
    },
    'NASDAQ (USA)': {
        'source': 'nasdaq100',
        'description': 'NASDAQ 100 Index constituents',
    },
    'NIKKEI (Japan)': {
        'source': 'nikkei225',
        'description': 'Nikkei 225 Index constituents',
    },
    'NSE (India)': {
        'source': 'nifty50',
        'description': 'NIFTY 50 Index constituents',
    },
    'BSE (India)': {
        'source': 'bse_sensex',
        'description': 'BSE SENSEX Index constituents',
    },
    'FTSE (UK)': {
        'source': 'ftse100',
        'description': 'FTSE 100 Index constituents',
    },
}


def get_nasdaq100_tickers(num_stocks=100, random_sample=True, seed=42):
    """Download NASDAQ 100 tickers from Wikipedia."""
    url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    tables = pd.read_html(resp.text)
    
    # Find the table with tickers
    for table in tables:
        if 'Ticker' in table.columns or 'Symbol' in table.columns:
            col = 'Ticker' if 'Ticker' in table.columns else 'Symbol'
            tickers = table[col].str.replace('.', '-', regex=False).tolist()
            break
    else:
        # Fallback: use known NASDAQ 100 tickers
        tickers = [
            'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'GOOG', 'TSLA', 
            'AVGO', 'COST', 'PEP', 'ADBE', 'CSCO', 'NFLX', 'CMCSA', 'AMD',
            'INTC', 'TMUS', 'INTU', 'TXN', 'QCOM', 'AMGN', 'AMAT', 'ISRG',
            'BKNG', 'HON', 'SBUX', 'MDLZ', 'LRCX', 'ADI', 'VRTX', 'REGN',
            'ADP', 'GILD', 'PANW', 'MU', 'KLAC', 'SNPS', 'CDNS', 'MELI',
            'PYPL', 'ASML', 'MAR', 'ORLY', 'CTAS', 'MNST', 'CSX', 'NXPI',
            'MRVL', 'PCAR', 'ADSK', 'FTNT', 'WDAY', 'CHTR', 'KDP', 'AEP',
            'PAYX', 'KHC', 'CPRT', 'MCHP', 'DXCM', 'MRNA', 'ODFL', 'EXC',
            'LULU', 'AZN', 'ROST', 'IDXX', 'CSGP', 'BIIB', 'FAST', 'XEL',
            'VRSK', 'CTSH', 'EA', 'DLTR', 'WBD', 'GEHC', 'ZS', 'ANSS',
            'FANG', 'ILMN', 'TEAM', 'DDOG', 'ALGN', 'BKR', 'WBA', 'EBAY',
            'CRWD', 'ENPH', 'JD', 'SIRI', 'LCID', 'RIVN', 'CEG'
        ]
    
    if random_sample and num_stocks < len(tickers):
        np.random.seed(seed)
        tickers = np.random.choice(tickers, num_stocks, replace=False).tolist()
    else:
        tickers = tickers[:num_stocks]
    
    return tickers


def get_nikkei225_tickers(num_stocks=100, random_sample=True, seed=42):
    """
    Get Nikkei 225 tickers.
    
    Note: Japanese stocks on Yahoo Finance use .T suffix for Tokyo Stock Exchange.
    """
    # Major Nikkei 225 constituents (Yahoo Finance format with .T suffix)
    tickers = [
        '7203.T', '6758.T', '9984.T', '8306.T', '6861.T', '9432.T', '6501.T',
        '7267.T', '4502.T', '8035.T', '6902.T', '7751.T', '6954.T', '8316.T',
        '7974.T', '9433.T', '6752.T', '4503.T', '6367.T', '7201.T', '8411.T',
        '4063.T', '6301.T', '8031.T', '4568.T', '6503.T', '2914.T', '8058.T',
        '3382.T', '9020.T', '7269.T', '8766.T', '4661.T', '6981.T', '8801.T',
        '9022.T', '6326.T', '5108.T', '8002.T', '7011.T', '6473.T', '4519.T',
        '6645.T', '9531.T', '5401.T', '4901.T', '8591.T', '6701.T', '6762.T',
        '1925.T', '4755.T', '9021.T', '6702.T', '7733.T', '4543.T', '8725.T',
        '5802.T', '4507.T', '6857.T', '8309.T', '7832.T', '2802.T', '4578.T',
        '8354.T', '9613.T', '4452.T', '7270.T', '3407.T', '9735.T', '2502.T',
        '4188.T', '8253.T', '7211.T', '4021.T', '6703.T', '5713.T', '3861.T',
        '9062.T', '5020.T', '7186.T', '8604.T', '4004.T', '7762.T', '8308.T',
        '5332.T', '6971.T', '1928.T', '9064.T', '6724.T', '4324.T', '6976.T',
        '5803.T', '4911.T', '2801.T', '8795.T', '5411.T', '8303.T', '9501.T',
        '7912.T'
    ]
    
    if random_sample and num_stocks < len(tickers):
        np.random.seed(seed)
        tickers = np.random.choice(tickers, num_stocks, replace=False).tolist()
    else:
        tickers = tickers[:num_stocks]
    
    return tickers


def get_nifty50_tickers(num_stocks=50, random_sample=True, seed=42):
    """
    Get NIFTY 50 (NSE India) tickers.
    
    Note: Indian stocks on Yahoo Finance use .NS suffix for NSE.
    """
    # NIFTY 50 constituents (Yahoo Finance format with .NS suffix)
    tickers = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
        'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'HCLTECH.NS',
        'SUNPHARMA.NS', 'TITAN.NS', 'BAJFINANCE.NS', 'DMART.NS', 'ULTRACEMCO.NS',
        'NTPC.NS', 'NESTLEIND.NS', 'WIPRO.NS', 'M&M.NS', 'POWERGRID.NS',
        'TATAMOTORS.NS', 'ONGC.NS', 'JSWSTEEL.NS', 'BAJAJFINSV.NS', 'TATASTEEL.NS',
        'ADANIENT.NS', 'TECHM.NS', 'HDFCLIFE.NS', 'COALINDIA.NS', 'GRASIM.NS',
        'LTIM.NS', 'INDUSINDBK.NS', 'DIVISLAB.NS', 'BRITANNIA.NS', 'SBILIFE.NS',
        'CIPLA.NS', 'EICHERMOT.NS', 'APOLLOHOSP.NS', 'HEROMOTOCO.NS', 'DRREDDY.NS',
        'BPCL.NS', 'TATACONSUM.NS', 'ADANIPORTS.NS', 'UPL.NS', 'HINDALCO.NS'
    ]
    
    if random_sample and num_stocks < len(tickers):
        np.random.seed(seed)
        tickers = np.random.choice(tickers, num_stocks, replace=False).tolist()
    else:
        tickers = tickers[:num_stocks]
    
    return tickers


def get_bse_sensex_tickers(num_stocks=30, random_sample=True, seed=42):
    """
    Get BSE SENSEX tickers.
    
    Note: Indian stocks on Yahoo Finance use .BO suffix for BSE.
    """
    # BSE SENSEX 30 constituents (Yahoo Finance format with .BO suffix)
    tickers = [
        'RELIANCE.BO', 'TCS.BO', 'HDFCBANK.BO', 'INFY.BO', 'ICICIBANK.BO',
        'HINDUNILVR.BO', 'ITC.BO', 'SBIN.BO', 'BHARTIARTL.BO', 'KOTAKBANK.BO',
        'LT.BO', 'AXISBANK.BO', 'ASIANPAINT.BO', 'MARUTI.BO', 'HCLTECH.BO',
        'SUNPHARMA.BO', 'TITAN.BO', 'BAJFINANCE.BO', 'ULTRACEMCO.BO', 'NTPC.BO',
        'NESTLEIND.BO', 'WIPRO.BO', 'M&M.BO', 'POWERGRID.BO', 'TATAMOTORS.BO',
        'JSWSTEEL.BO', 'BAJAJFINSV.BO', 'TATASTEEL.BO', 'TECHM.BO', 'INDUSINDBK.BO'
    ]
    
    if random_sample and num_stocks < len(tickers):
        np.random.seed(seed)
        tickers = np.random.choice(tickers, num_stocks, replace=False).tolist()
    else:
        tickers = tickers[:num_stocks]
    
    return tickers


def get_ftse100_tickers(num_stocks=100, random_sample=True, seed=42):
    """
    Get FTSE 100 tickers.
    
    Note: UK stocks on Yahoo Finance use .L suffix for London Stock Exchange.
    """
    # Major FTSE 100 constituents (Yahoo Finance format with .L suffix)
    tickers = [
        'SHEL.L', 'AZN.L', 'HSBA.L', 'ULVR.L', 'BP.L', 'GSK.L', 'RIO.L',
        'DGE.L', 'BATS.L', 'REL.L', 'LSEG.L', 'AAL.L', 'NG.L', 'CPG.L',
        'VOD.L', 'PRU.L', 'GLEN.L', 'BHP.L', 'BARC.L', 'LLOY.L', 'EXPN.L',
        'AHT.L', 'RKT.L', 'SSE.L', 'CRH.L', 'BA.L', 'IMB.L', 'NWG.L',
        'ABF.L', 'TSCO.L', 'ANTO.L', 'STAN.L', 'LGEN.L', 'MNG.L', 'SGRO.L',
        'AVV.L', 'RR.L', 'SMT.L', 'HLMA.L', 'SDR.L', 'WPP.L', 'SN.L',
        'IHG.L', 'III.L', 'STJ.L', 'MNDI.L', 'JD.L', 'SPX.L', 'PSN.L',
        'SBRY.L', 'LAND.L', 'BT-A.L', 'AUTO.L', 'WTB.L', 'SGE.L', 'PSON.L',
        'FRAS.L', 'BRBY.L', 'ENT.L', 'CRDA.L', 'INF.L', 'RS1.L', 'SMIN.L',
        'ICP.L', 'BNZL.L', 'HIK.L', 'FERG.L', 'EVR.L', 'RMV.L', 'WEIR.L',
        'BME.L', 'ADM.L', 'DARK.L', 'PHNX.L', 'ITRK.L', 'KGF.L', 'CCH.L',
        'OCDO.L', 'TW.L', 'EDV.L', 'BDEV.L', 'UU.L', 'SVT.L', 'JMAT.L',
        'HLN.L', 'SMDS.L', 'BNKE.L', 'RTO.L', 'VCT.L', 'FLTR.L', 'DCC.L',
        'HSX.L', 'MGGT.L', 'POLY.L', 'HWDN.L', 'BKG.L', 'VMUK.L', 'BLND.L'
    ]
    
    if random_sample and num_stocks < len(tickers):
        np.random.seed(seed)
        tickers = np.random.choice(tickers, num_stocks, replace=False).tolist()
    else:
        tickers = tickers[:num_stocks]
    
    return tickers


def get_index_tickers(index_name, num_stocks=100, random_sample=True, seed=42):
    """
    Get tickers for a specific market index.
    
    Parameters:
        index_name: One of 'sp500', 'nasdaq100', 'nikkei225', 'nifty50', 'bse_sensex', 'ftse100'
        num_stocks: Number of stocks to return
        random_sample: If True, randomly sample stocks
        seed: Random seed for reproducibility
    
    Returns:
        tickers: List of ticker symbols
    """
    index_funcs = {
        'sp500': get_sp500_tickers,
        'nasdaq100': get_nasdaq100_tickers,
        'nikkei225': get_nikkei225_tickers,
        'nifty50': get_nifty50_tickers,
        'bse_sensex': get_bse_sensex_tickers,
        'ftse100': get_ftse100_tickers,
    }
    
    if index_name not in index_funcs:
        raise ValueError(f"Unknown index: {index_name}. Available: {list(index_funcs.keys())}")
    
    return index_funcs[index_name](num_stocks, random_sample, seed)


def download_index_returns(index_name, num_stocks=100, start_date='2021-01-01', 
                           end_date=None, random_sample=True, seed=42, verbose=True):
    """
    Download returns for a specific market index.
    
    Parameters:
        index_name: One of 'sp500', 'nasdaq100', 'nikkei225', 'nifty50', 'bse_sensex', 'ftse100'
        num_stocks: Number of stocks to use
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD), None for current date
        random_sample: If True, randomly sample stocks
        seed: Random seed for reproducibility
        verbose: Print progress
    
    Returns:
        returns: DataFrame of daily returns
    """
    if verbose:
        print(f"\n--- Loading {index_name.upper()} ---")
    
    tickers = get_index_tickers(index_name, num_stocks, random_sample, seed)
    returns = download_returns(tickers, start_date, end_date, verbose=verbose)
    
    return returns


def get_all_datasets(num_stocks=50, start_date='2020-01-01', end_date=None,
                     random_sample=True, seed=42, verbose=True):
    """
    Download returns for all market indices used in the paper.
    
    Parameters:
        num_stocks: Number of stocks per index
        start_date: Start date
        end_date: End date
        random_sample: Random sampling
        seed: Random seed
        verbose: Print progress
    
    Returns:
        datasets: Dictionary of {name: returns_dataframe}
    """
    indices = ['sp500', 'nasdaq100', 'nikkei225', 'nifty50', 'bse_sensex', 'ftse100']
    display_names = {
        'sp500': 'S&P 500 (USA)',
        'nasdaq100': 'NASDAQ (USA)',
        'nikkei225': 'NIKKEI (Japan)',
        'nifty50': 'NSE (India)',
        'bse_sensex': 'BSE (India)',
        'ftse100': 'FTSE (UK)',
    }
    
    datasets = {}
    
    for idx in indices:
        try:
            returns = download_index_returns(
                idx, num_stocks, start_date, end_date, random_sample, seed, verbose
            )
            if len(returns) > 0 and len(returns.columns) > 10:
                datasets[display_names[idx]] = returns
            else:
                if verbose:
                    print(f"  Warning: Insufficient data for {idx}, skipping...")
        except Exception as e:
            if verbose:
                print(f"  Error loading {idx}: {e}")
    
    return datasets
