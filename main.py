"""
Main entry point for Portfolio Optimization

Usage:
    python main.py                    # Run full paper replication
    python main.py --quick            # Quick test with one dataset, one frequency
    python main.py --dataset sp500    # Run specific dataset
"""

import sys
import numpy as np
import pandas as pd
from data_loader import (
    get_index_tickers, 
    download_returns, 
    get_all_datasets,
    DATASETS
)
from optimization import run_optimization
from visualization import (
    generate_comparison_table, 
    print_latex_table, 
    print_summary,
    plot_comparison_bar_chart,
    plot_heatmap_2d,
    plot_heatmap_3d_slices,
    generate_paper_tables
)


# Configuration
CONFIG = {
    # Data settings
    'num_stocks': 100,
    'random_sample': True,
    'random_seed': 42,
    'start_date': '2020-01-01',
    'end_date': None,
    
    # Optimization settings
    'N_train': 200,
    'grid_size': 11,
    
    # Paper replication: run all datasets and frequencies
    'datasets': ['sp500', 'nasdaq100', 'nikkei225', 'nifty50', 'bse_sensex', 'ftse100'],
    'frequencies': [30, 60, 90],  # Rebalancing frequencies (days)
}

# Display names for datasets
DATASET_DISPLAY_NAMES = {
    'sp500': 'S&P',
    'nasdaq100': 'NASDAQ',
    'nikkei225': 'NIKKEI',
    'nifty50': 'NSE',
    'bse_sensex': 'BSE',
    'ftse100': 'FTSE',
}


def run_single_experiment(returns, dataset_name, frequency, grid_size=11, N_train=200):
    """Run optimization for a single dataset and frequency."""
    results = run_optimization(
        returns,
        N_train=N_train,
        N_test=frequency,
        rebalance_freq=frequency,
        grid_size=grid_size,
        verbose=False
    )
    return results


def run_paper_replication(config):
    """
    Run full paper replication: all datasets × all frequencies.
    
    Returns:
        all_results: Dict of {(dataset, freq): results}
    """
    print("\n" + "="*70)
    print("PAPER REPLICATION: Running all datasets and frequencies")
    print("="*70)
    print(f"Datasets: {config['datasets']}")
    print(f"Frequencies: {config['frequencies']} days")
    print("="*70 + "\n")
    
    all_results = {}
    
    for dataset in config['datasets']:
        print(f"\n{'='*50}")
        print(f"Loading {DATASET_DISPLAY_NAMES.get(dataset, dataset)} data...")
        print(f"{'='*50}")
        
        try:
            # Get tickers for this dataset
            tickers = get_index_tickers(
                dataset,
                num_stocks=config['num_stocks'],
                random_sample=config['random_sample'],
                seed=config['random_seed']
            )
            
            # Download returns
            returns = download_returns(
                tickers,
                start_date=config['start_date'],
                end_date=config['end_date'],
                verbose=True
            )
            
            if len(returns) == 0 or len(returns.columns) < 10:
                print(f"  Insufficient data for {dataset}, skipping...")
                continue
                
            print(f"  Dataset: {len(returns.columns)} assets, {len(returns)} trading days")
            
            # Run for each frequency
            for freq in config['frequencies']:
                print(f"\n  Running {freq}-day rebalancing...")
                
                results = run_single_experiment(
                    returns, 
                    dataset, 
                    freq, 
                    grid_size=config['grid_size'],
                    N_train=config['N_train']
                )
                
                all_results[(dataset, freq)] = results
                
                # Quick summary
                if results.get('our_3d_method', {}).get('avg_variance') is not None:
                    vol = np.sqrt(results['our_3d_method']['avg_variance'] * 252) * 100
                    print(f"    Our 3D: {vol:.3f}%")
                    
        except Exception as e:
            print(f"  Error processing {dataset}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return all_results


def main():
    # Parse command line arguments
    quick_mode = '--quick' in sys.argv
    specific_dataset = None
    for arg in sys.argv:
        if arg.startswith('--dataset='):
            specific_dataset = arg.split('=')[1]
    
    if quick_mode:
        # Quick test: single dataset, single frequency
        config = CONFIG.copy()
        config['datasets'] = ['sp500']
        config['frequencies'] = [30]
        print("\n[QUICK MODE] Running S&P 500 with 30-day frequency only")
    elif specific_dataset:
        # Specific dataset
        config = CONFIG.copy()
        config['datasets'] = [specific_dataset]
        print(f"\n[SINGLE DATASET] Running {specific_dataset} only")
    else:
        config = CONFIG
    
    # Run paper replication
    all_results = run_paper_replication(config)
    
    if not all_results:
        print("\nNo results generated. Check data availability.")
        return
    
    # Generate paper-style tables
    print("\n" + "="*70)
    print("RESULTS TABLES (Paper Format)")
    print("="*70)
    
    generate_paper_tables(all_results, config['frequencies'])
    
    # Generate detailed results for each dataset/frequency
    for (dataset, freq), results in all_results.items():
        print(f"\n{'='*70}")
        print(f"Detailed Results: {DATASET_DISPLAY_NAMES.get(dataset, dataset)} - {freq} days")
        print(f"{'='*70}")
        
        table = generate_comparison_table(results)
        print(table.to_string(index=False))
        
        # Generate plots for first successful result
        if dataset == list(all_results.keys())[0][0] and freq == config['frequencies'][0]:
            print("\nGenerating visualizations for first result...")
            plot_comparison_bar_chart(table, save_path=f'comparison_{dataset}_{freq}d.png')
            if results.get('paper_2d', {}).get('all_results'):
                plot_heatmap_2d(results, save_path=f'heatmap_2d_{dataset}_{freq}d.png')
            if results.get('our_3d', {}).get('best_params'):
                plot_heatmap_3d_slices(results, save_path=f'heatmap_3d_{dataset}_{freq}d.png')
    
    print("\n" + "="*70)
    print("PAPER REPLICATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
