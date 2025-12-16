import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from data_samples import get_returns 
from optimization import (
    optimize_portfolio, 
    plot_performance_heatmap, 
    compare_estimators,
    plot_performance_timeseries
)

#########################################################
# CONFIGURATION - ADJUST ALL PARAMETERS HERE
#########################################################

CONFIG = {
    # Stock selection
    'num_stocks': 150,            # Number of stocks to use (None = all 503)
                                 # Recommended: 50 (fast), 100 (medium), 150 (slow)
    'random_sample': True,       # True: Random sample from S&P 500
                                 # False: Top N by market cap (biased toward mega-caps)
    'random_seed': 42,           # Seed for reproducibility (change to get different samples)
    
    # Time windows
    'N_train': 200,              # Training window size (days)
    'N_test': 30,                # Test window size (days)
    'rebalance_freq': 30,        # Rebalancing frequency (days)
    'days_back': 1500,           # Historical data to download
    
    # Optimization settings (now 3D: θ × φ × ψ)
    'grid_size': 11,              # Grid resolution for (θ, φ, ψ) - cubic grid
                                 # Options: 4 (64 combos), 6 (216 combos, recommended), 
                                 #          8 (512 combos), 11 (1331 combos, paper precision)
    'n_jobs': -1,                # CPU cores (-1 = all available)
    'use_gmvp': True,            # Use Global Minimum Variance Portfolio (no return constraint)
                                 # True: Focus purely on covariance estimation (paper's approach)
                                 # False: Add minimum return constraint (more infeasibility issues)
    
    # Output
    'verbose': True,             # Print progress information
    'save_plots': True,          # Generate and save visualization plots
}


#########################################################
# END CONFIGURATION
#########################################################

print("="*60)
print("PORTFOLIO OPTIMIZATION (with Tyler's M-Estimator)")
print("="*60)

# Data source for tickers
print("\nDownloading S&P 500 tickers...")
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
headers = {"User-Agent": "Mozilla/5.0"}
resp = requests.get(url, headers=headers)
resp.raise_for_status()
df = pd.read_html(resp.text, header=0)[0]

# Select stocks based on configuration
if CONFIG['num_stocks'] is None:
    print("\nUsing all S&P 500 stocks...")
    tickers = df["Symbol"].tolist()
else:
    if CONFIG['random_sample']:
        print(f"\nRandomly sampling {CONFIG['num_stocks']} stocks (seed={CONFIG['random_seed']})...")
        np.random.seed(CONFIG['random_seed'])
        all_symbols = df["Symbol"].tolist()
        tickers = np.random.choice(all_symbols, size=CONFIG['num_stocks'], replace=False).tolist()
    else:
        print(f"\nSelecting top {CONFIG['num_stocks']} stocks by market cap...")
        tickers = df["Symbol"].head(CONFIG['num_stocks']).tolist()
print(f"✓ Selected {len(tickers)} stocks from S&P 500")

# Download returns data
print("\nDownloading historical returns...")
returns = get_returns(tickers, days_back=CONFIG['days_back'])
print(f"✓ Returns data shape: {returns.shape}")
print(f"✓ Date range: {returns.index[0]} to {returns.index[-1]}")

# Run hyperparameter optimization (Section III of paper)
print("\nRunning hyperparameter optimization...")
print(f"Configuration: {len(tickers)} stocks, {CONFIG['grid_size']}³ grid ({CONFIG['grid_size']**3} combinations)")
print("Estimators: SCM, MP, Tyler's M-estimator, Ledoit-Wolf shrinkage target")
print("Optimizations: cached estimators + parallel QP solving")
print(f"Portfolio approach: {'GMVP (Global Minimum Variance)' if CONFIG['use_gmvp'] else 'Mean-Variance with return constraint'}")
results = optimize_portfolio(
    returns=returns,
    N_train=CONFIG['N_train'],
    N_test=CONFIG['N_test'],
    rebalance_freq=CONFIG['rebalance_freq'],
    grid_size=CONFIG['grid_size'],
    n_jobs=CONFIG['n_jobs'],
    use_gmvp=CONFIG['use_gmvp'],
    verbose=CONFIG['verbose']
)

# Compare with baseline estimators
print("\nComparing with baseline estimators...")
comparison_df = compare_estimators(results)
print("\n" + comparison_df.to_string(index=False))

# Generate visualizations
if CONFIG['save_plots']:
    print("\nGenerating visualizations...")
    plot_performance_heatmap(results, save_path='performance_heatmap.png')
    plot_performance_timeseries(
        results, 
        returns, 
        N_train=CONFIG['N_train'], 
        N_test=CONFIG['N_test'], 
        rebalance_freq=CONFIG['rebalance_freq'],
        save_path='performance_timeseries.png'
    )

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Optimal hyperparameters:")
print(f"  θ = {results['theta_opt']:.2f} (F vs RobustBlend mixing)")
print(f"  φ = {results['phi_opt']:.2f} (Regularized vs SCM mixing)")
print(f"  ψ = {results['psi_opt']:.2f} (MP vs Tyler mixing)")
print(f"\nFormula: Σ*(θ,φ,ψ) = φ[θF + (1-θ)(ψ·MP + (1-ψ)·Tyler)] + (1-φ)·SCM")
print(f"\nPerformance improvement over SCM:")
scm_vol = np.sqrt(results['avg_performance'][0, 0, 0] * 252)  # 3D now
opt_vol = np.sqrt(results['best_variance'] * 252)
improvement = (scm_vol - opt_vol) / scm_vol * 100
print(f"  SCM volatility: {scm_vol:.4f}")
print(f"  Optimal volatility: {opt_vol:.4f}")
print(f"  Improvement: {improvement:.2f}%")
print("="*60)