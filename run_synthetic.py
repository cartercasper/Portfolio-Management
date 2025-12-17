"""
Run Synthetic Data Experiments

Runs portfolio optimization on synthetic data to test estimator performance
under different distributional assumptions.

Usage:
    python run_synthetic.py                    # Run all distributions
    python run_synthetic.py --dist gaussian    # Run specific distribution
    python run_synthetic.py --quick            # Quick test (fewer seeds, all frequencies)
"""

import sys
import numpy as np
import pandas as pd
from synthetic_data import (
    generate_synthetic_returns,
    get_all_synthetic_datasets,
    print_synthetic_summary,
    SYNTHETIC_TYPES,
    SYNTHETIC_CONFIG
)
from optimization import run_optimization
from visualization import (
    generate_comparison_table, 
    print_latex_table,
    print_summary,
    plot_comparison_bar_chart,
    plot_heatmap_2d,
    plot_heatmap_3d_slices,
    generate_paper_tables,
    ESTIMATOR_ORDER
)


# Configuration for synthetic experiments
SYNTHETIC_EXPERIMENT_CONFIG = {
    # Data dimensions (matching real data scale)
    'N': 100,              # Number of assets
    'T': 1200,             # Observations (~5 years daily)
    
    # Optimization settings
    'N_train': 200,        # Training window (same as real data)
    'grid_size': 11,       # Grid search resolution
    'frequencies': [30, 60, 90],  # Rebalancing frequencies
    
    # Monte Carlo settings
    'n_seeds': 5,          # Number of random seeds for averaging
    'base_seed': 50,
}


def run_single_synthetic_experiment(returns, freq, grid_size=11, N_train=200, verbose=False):
    """Run optimization on a single synthetic dataset."""
    results = run_optimization(
        returns,
        N_train=N_train,
        N_test=freq,
        rebalance_freq=freq,
        grid_size=grid_size,
        verbose=verbose
    )
    return results


def run_synthetic_experiments(config, distributions=None, verbose=True):
    """
    Run experiments on synthetic data.
    
    Parameters:
        config: Experiment configuration dict
        distributions: List of distributions to test, or None for all
        verbose: Print progress
    
    Returns:
        all_results: Dict of {(distribution, freq, seed): results}
    """
    if distributions is None:
        distributions = list(SYNTHETIC_TYPES.keys())
    
    print("\n" + "="*70)
    print("SYNTHETIC DATA EXPERIMENTS")
    print("="*70)
    print(f"Distributions: {distributions}")
    print(f"Assets: {config['N']}, Observations: {config['T']}")
    print(f"Frequencies: {config['frequencies']} days")
    print(f"Seeds: {config['n_seeds']} (for Monte Carlo averaging)")
    print("="*70 + "\n")
    
    all_results = {}
    
    for dist in distributions:
        print(f"\n{'='*60}")
        print(f"Distribution: {SYNTHETIC_TYPES[dist]['name']}")
        print(f"{'='*60}")
        
        for seed_idx in range(config['n_seeds']):
            seed = config['base_seed'] + seed_idx
            
            # Generate synthetic data
            returns, true_cov, info = generate_synthetic_returns(
                distribution=dist,
                N=config['N'],
                T=config['T'],
                seed=seed
            )
            
            if seed_idx == 0 and verbose:
                print_synthetic_summary(returns, true_cov, info)
            
            # Run for each frequency
            for freq in config['frequencies']:
                if verbose:
                    print(f"  Seed {seed}, {freq}-day rebalancing...", end=' ')
                
                try:
                    results = run_single_synthetic_experiment(
                        returns, freq,
                        grid_size=config['grid_size'],
                        N_train=config['N_train'],
                        verbose=False
                    )
                    
                    # Store true covariance for PRIAL calculation
                    results['true_cov'] = true_cov
                    results['distribution'] = dist
                    results['seed'] = seed
                    
                    all_results[(dist, freq, seed)] = results
                    
                    if verbose:
                        vol = np.sqrt(results['our_3d_method']['avg_variance'] * 252) * 100
                        print(f"3D Vol: {vol:.2f}%")
                        
                except Exception as e:
                    print(f"Error: {e}")
                    continue
    
    return all_results


def generate_synthetic_visualizations(all_results, config, output_prefix='synthetic'):
    """
    Generate the same visualizations as real data for synthetic experiments.
    
    Parameters:
        all_results: Dict of {(distribution, freq, seed): results}
        config: Experiment configuration
        output_prefix: Prefix for output files
    """
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Group results by distribution
    distributions = sorted(list(set(dist for dist, _, _ in all_results.keys())))
    
    for dist in distributions:
        dist_name = SYNTHETIC_TYPES[dist]['name'].replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
        
        # Get first seed's results for each frequency (for detailed plots)
        for freq in config['frequencies']:
            key = (dist, freq, config['base_seed'])
            if key not in all_results:
                continue
            
            results = all_results[key]
            
            # Generate comparison table
            table = generate_comparison_table(results)
            
            # Generate bar chart
            save_path = f'{output_prefix}_{dist_name}_{freq}d_comparison.png'
            plot_comparison_bar_chart(table, save_path=save_path)
            
            # Generate 2D heatmap
            if results.get('dual_method_2d', {}).get('avg_performance') is not None:
                save_path = f'{output_prefix}_{dist_name}_{freq}d_heatmap_2d.png'
                plot_heatmap_2d(results, save_path=save_path)
            
            # Generate 3D heatmap slices
            if results.get('our_3d_method', {}).get('avg_performance') is not None:
                save_path = f'{output_prefix}_{dist_name}_{freq}d_heatmap_3d.png'
                plot_heatmap_3d_slices(results, save_path=save_path)


def generate_synthetic_paper_tables(all_results, config):
    """
    Generate paper-style tables for synthetic data (matching real data format).
    
    Rows: Estimators
    Columns: Distributions × Frequencies
    """
    distributions = sorted(list(set(dist for dist, _, _ in all_results.keys())))
    frequencies = config['frequencies']
    
    # Build table for each frequency
    for freq in frequencies:
        print(f"\n{'='*80}")
        print(f"TABLE: Annualized Volatility (%) - {freq}-day Rebalancing")
        print(f"{'='*80}")
        
        # Collect data for this frequency
        rows = []
        for estimator_key, estimator_display in ESTIMATOR_ORDER:
            row = {'Estimator': estimator_display}
            
            for dist in distributions:
                # Average across seeds
                vols = []
                for seed in range(config['base_seed'], config['base_seed'] + config['n_seeds']):
                    key = (dist, freq, seed)
                    if key not in all_results:
                        continue
                    
                    results = all_results[key]
                    
                    # Extract volatility for this estimator
                    if estimator_key in ['paper_2d', 'our_3d']:
                        if estimator_key == 'paper_2d':
                            data = results.get('dual_method_2d', {})
                        else:
                            data = results.get('our_3d_method', {})
                        if data.get('avg_variance') is not None:
                            vols.append(np.sqrt(data['avg_variance'] * 252) * 100)
                    else:
                        fixed = results.get('fixed_estimators', {})
                        if estimator_key in fixed:
                            var = fixed[estimator_key].get('avg_variance')
                            if var is not None:
                                vols.append(np.sqrt(var * 252) * 100)
                
                # Store mean ± std
                dist_short = SYNTHETIC_TYPES[dist]['name'].split()[0]  # First word
                if vols:
                    row[dist_short] = f"{np.mean(vols):.2f}"
                else:
                    row[dist_short] = '-'
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        
        # LaTeX version
        print("\n--- LaTeX ---")
        print(df.to_latex(index=False, escape=False))


def print_detailed_results_by_distribution(all_results, config):
    """Print detailed results for each distribution (like real data output)."""
    
    distributions = sorted(list(set(dist for dist, _, _ in all_results.keys())))
    
    for dist in distributions:
        print(f"\n{'='*80}")
        print(f"DETAILED RESULTS: {SYNTHETIC_TYPES[dist]['name']}")
        print(f"Description: {SYNTHETIC_TYPES[dist]['description']}")
        print(f"{'='*80}")
        
        for freq in config['frequencies']:
            # Use first seed for detailed table
            key = (dist, freq, config['base_seed'])
            if key not in all_results:
                continue
            
            results = all_results[key]
            
            print(f"\n--- {freq}-day Rebalancing (seed={config['base_seed']}) ---")
            table = generate_comparison_table(results)
            print(table.to_string(index=False))
            
            # Print optimal parameters
            ext_3d = results.get('our_3d_method', {})
            if ext_3d:
                print(f"\nOptimal 3D Parameters: θ={ext_3d.get('theta_opt', 'N/A'):.2f}, "
                      f"φ={ext_3d.get('phi_opt', 'N/A'):.2f}, ψ={ext_3d.get('psi_opt', 'N/A'):.2f}")


def aggregate_results(all_results, frequencies):
    """
    Aggregate results across seeds for each distribution/frequency.
    
    Returns:
        summary: DataFrame with mean and std of results
    """
    rows = []
    
    # Get unique distributions
    distributions = sorted(list(set(dist for dist, _, _ in all_results.keys())))
    
    for dist in distributions:
        for freq in frequencies:
            # Collect results for this dist/freq across seeds
            seed_results = {k: v for k, v in all_results.items() 
                          if k[0] == dist and k[1] == freq}
            
            if not seed_results:
                continue
            
            # Extract metrics
            scm_vols = []
            mp_vols = []
            tyler_vols = []
            dual_2d_vols = []
            our_3d_vols = []
            
            for (_, _, seed), res in seed_results.items():
                fixed = res['fixed_estimators']
                
                scm_vols.append(np.sqrt(fixed['SCM']['avg_variance'] * 252) * 100)
                mp_vols.append(np.sqrt(fixed['Marchenko-Pastur (MP)']['avg_variance'] * 252) * 100)
                tyler_vols.append(np.sqrt(fixed["Tyler's M-Estimator"]['avg_variance'] * 252) * 100)
                dual_2d_vols.append(np.sqrt(res['dual_method_2d']['avg_variance'] * 252) * 100)
                our_3d_vols.append(np.sqrt(res['our_3d_method']['avg_variance'] * 252) * 100)
            
            # Calculate stats
            row = {
                'Distribution': SYNTHETIC_TYPES[dist]['name'],
                'Frequency': f'{freq}d',
                'SCM (%)': f"{np.mean(scm_vols):.2f}±{np.std(scm_vols):.2f}",
                'MP (%)': f"{np.mean(mp_vols):.2f}±{np.std(mp_vols):.2f}",
                'Tyler (%)': f"{np.mean(tyler_vols):.2f}±{np.std(tyler_vols):.2f}",
                '2D Dual (%)': f"{np.mean(dual_2d_vols):.2f}±{np.std(dual_2d_vols):.2f}",
                '3D Ext (%)': f"{np.mean(our_3d_vols):.2f}±{np.std(our_3d_vols):.2f}",
                'Best': min([
                    ('SCM', np.mean(scm_vols)),
                    ('MP', np.mean(mp_vols)),
                    ('Tyler', np.mean(tyler_vols)),
                    ('2D', np.mean(dual_2d_vols)),
                    ('3D', np.mean(our_3d_vols)),
                ], key=lambda x: x[1])[0],
            }
            rows.append(row)
    
    return pd.DataFrame(rows)


def print_synthetic_tables(all_results, frequencies):
    """Print paper-style tables for synthetic results."""
    
    # Aggregate across seeds
    summary = aggregate_results(all_results, frequencies)
    
    print("\n" + "="*80)
    print("SYNTHETIC DATA RESULTS (Annualized Volatility %)")
    print("="*80)
    print(summary.to_string(index=False))
    print("="*80)
    
    # Print insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    # Check if Tyler helps for heavy-tailed distributions
    for dist in ['student_t', 'pareto']:
        dist_results = summary[summary['Distribution'].str.contains(
            'Student' if dist == 'student_t' else 'Pareto', case=False
        )]
        if len(dist_results) > 0:
            best_methods = dist_results['Best'].value_counts()
            print(f"\n{SYNTHETIC_TYPES[dist]['name']}:")
            print(f"  Best method distribution: {dict(best_methods)}")
            if 'Tyler' in best_methods.index or '3D' in best_methods.index:
                print(f"  → Tyler/3D shows benefit for heavy-tailed data ✓")
            else:
                print(f"  → Tyler shows limited benefit")
    
    # Check Gaussian baseline
    gaussian_results = summary[summary['Distribution'].str.contains('Gaussian', case=False)]
    if len(gaussian_results) > 0:
        print(f"\nGaussian N(0,Σ) (baseline):")
        print(f"  Best methods: {dict(gaussian_results['Best'].value_counts())}")
        print(f"  → SCM should be near-optimal for Gaussian data")
    
    # Factor model
    factor_results = summary[summary['Distribution'].str.contains('Factor', case=False)]
    if len(factor_results) > 0:
        print(f"\nFactor Model:")
        print(f"  Best methods: {dict(factor_results['Best'].value_counts())}")
        print(f"  → MP should excel at eigenvalue cleaning")


def print_optimal_parameters(all_results):
    """Print optimal parameters for each distribution."""
    print("\n" + "="*80)
    print("OPTIMAL PARAMETERS BY DISTRIBUTION")
    print("="*80)
    
    distributions = sorted(list(set(dist for dist, _, _ in all_results.keys())))
    
    for dist in distributions:
        print(f"\n{SYNTHETIC_TYPES[dist]['name']}:")
        
        # Collect optimal parameters
        theta_vals, phi_vals, psi_vals = [], [], []
        
        for (d, freq, seed), res in all_results.items():
            if d == dist:
                theta_vals.append(res['our_3d_method']['theta_opt'])
                phi_vals.append(res['our_3d_method']['phi_opt'])
                psi_vals.append(res['our_3d_method']['psi_opt'])
        
        if theta_vals:
            print(f"  θ (F vs RobustBlend): {np.mean(theta_vals):.2f} ± {np.std(theta_vals):.2f}")
            print(f"  φ (Regularized vs SCM): {np.mean(phi_vals):.2f} ± {np.std(phi_vals):.2f}")
            print(f"  ψ (MP vs Tyler): {np.mean(psi_vals):.2f} ± {np.std(psi_vals):.2f}")
            
            # Interpret ψ
            avg_psi = np.mean(psi_vals)
            if avg_psi < 0.3:
                print(f"  → Tyler-dominant (ψ={avg_psi:.2f})")
            elif avg_psi > 0.7:
                print(f"  → MP-dominant (ψ={avg_psi:.2f})")
            else:
                print(f"  → Balanced MP/Tyler (ψ={avg_psi:.2f})")


def main():
    # Parse arguments
    quick_mode = '--quick' in sys.argv
    specific_dist = None
    
    for arg in sys.argv:
        if arg.startswith('--dist='):
            specific_dist = arg.split('=')[1]
    
    # Configure
    config = SYNTHETIC_EXPERIMENT_CONFIG.copy()
    
    if quick_mode:
        config['N'] = 50
        config['T'] = 600
        config['n_seeds'] = 2
        # Run all three frequencies even in quick mode
        config['frequencies'] = [30, 60, 90]
        print("\n[QUICK MODE] Reduced dimensions for testing")
        print(f"[QUICK MODE] Running frequencies: {config['frequencies']}")
    
    distributions = [specific_dist] if specific_dist else None
    
    # Run experiments
    all_results = run_synthetic_experiments(config, distributions, verbose=True)
    
    if not all_results:
        print("\nNo results generated.")
        return
    
    # Generate paper-style tables (matching real data format)
    generate_synthetic_paper_tables(all_results, config)
    
    # Print detailed results by distribution
    print_detailed_results_by_distribution(all_results, config)
    
    # Print optimal parameters
    print_optimal_parameters(all_results)
    
    # Generate visualizations (same as real data)
    generate_synthetic_visualizations(all_results, config, output_prefix='synthetic')
    
    print("\n" + "="*70)
    print("SYNTHETIC EXPERIMENTS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    for dist_key in SYNTHETIC_TYPES.keys():
        dist_name = SYNTHETIC_TYPES[dist_key]['name'].replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
        for freq in config['frequencies']:
            print(f"  - synthetic_{dist_name}_{freq}d_comparison.png")
            print(f"  - synthetic_{dist_name}_{freq}d_heatmap_2d.png")
            print(f"  - synthetic_{dist_name}_{freq}d_heatmap_3d.png")


if __name__ == '__main__':
    main()
