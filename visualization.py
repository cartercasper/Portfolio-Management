"""
Visualization Module

Functions for plotting and table generation for portfolio optimization results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_comparison_table(results):
    """
    Generate Table I style comparison table.
    
    Parameters:
        results: Dictionary from run_paper_comparison()
    
    Returns:
        table: pandas DataFrame with comparison results
    """
    rows = []
    
    # Fixed estimators
    for name, data in results['fixed_estimators'].items():
        rows.append({
            'Method': name,
            'Parameters': '-',
            'Avg Variance': data['avg_variance'],
            'Ann. Volatility': np.sqrt(data['avg_variance'] * 252),
        })
    
    # Paper's 2D Dual Method
    dual_2d = results['dual_method_2d']
    rows.append({
        'Method': "Paper's Dual Method Σ*(θ,φ)",
        'Parameters': f"θ={dual_2d['theta_opt']:.2f}, φ={dual_2d['phi_opt']:.2f}",
        'Avg Variance': dual_2d['avg_variance'],
        'Ann. Volatility': np.sqrt(dual_2d['avg_variance'] * 252),
    })
    
    # Our 3D Extension
    ext_3d = results['our_3d_method']
    rows.append({
        'Method': "Our 3D Extension Σ*(θ,φ,ψ)",
        'Parameters': f"θ={ext_3d['theta_opt']:.2f}, φ={ext_3d['phi_opt']:.2f}, ψ={ext_3d['psi_opt']:.2f}",
        'Avg Variance': ext_3d['avg_variance'],
        'Ann. Volatility': np.sqrt(ext_3d['avg_variance'] * 252),
    })
    
    table = pd.DataFrame(rows)
    
    # Calculate improvement vs SCM
    scm_vol = table.loc[table['Method'] == 'SCM', 'Ann. Volatility'].values[0]
    table['Improvement vs SCM (%)'] = (scm_vol - table['Ann. Volatility']) / scm_vol * 100
    
    # Sort by volatility
    table = table.sort_values('Ann. Volatility')
    table = table.reset_index(drop=True)
    
    return table


def print_latex_table(table):
    """
    Print table in LaTeX format.
    
    Parameters:
        table: DataFrame from generate_comparison_table()
    """
    print("\n" + "="*70)
    print("LaTeX Table:")
    print("="*70)
    print(table.to_latex(index=False, float_format="%.4f"))


def print_summary(table, results):
    """
    Print summary statistics.
    
    Parameters:
        table: DataFrame from generate_comparison_table()
        results: Dictionary from run_paper_comparison()
    """
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    best_method = table.iloc[0]['Method']
    best_vol = table.iloc[0]['Ann. Volatility']
    best_improvement = table.iloc[0]['Improvement vs SCM (%)']
    
    scm_row = table[table['Method'] == 'SCM'].iloc[0]
    
    print(f"Best Method: {best_method}")
    print(f"Best Volatility: {best_vol:.4f}")
    print(f"Improvement vs SCM: {best_improvement:.2f}%")
    print(f"\nSCM Baseline Volatility: {scm_row['Ann. Volatility']:.4f}")
    
    # Compare 2D vs 3D
    dual_row = table[table['Method'].str.contains('Dual Method')].iloc[0]
    ext_row = table[table['Method'].str.contains('3D Extension')].iloc[0]
    
    print(f"\nPaper's 2D Dual Method: {dual_row['Ann. Volatility']:.4f} ({dual_row['Improvement vs SCM (%)']:.2f}%)")
    print(f"Our 3D Extension:       {ext_row['Ann. Volatility']:.4f} ({ext_row['Improvement vs SCM (%)']:.2f}%)")
    
    improvement_3d_vs_2d = (dual_row['Ann. Volatility'] - ext_row['Ann. Volatility']) / dual_row['Ann. Volatility'] * 100
    print(f"3D vs 2D Improvement:   {improvement_3d_vs_2d:.2f}%")
    
    print("="*70)


def plot_comparison_bar_chart(table, save_path='comparison_bar_chart.png'):
    """
    Create bar chart comparing all estimators.
    
    Parameters:
        table: DataFrame from generate_comparison_table()
        save_path: Path to save the figure
    """
    # Sort by volatility for plotting
    table_sorted = table.sort_values('Ann. Volatility', ascending=True)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # === Left plot: Annualized Volatility ===
    ax1 = axes[0]
    methods = table_sorted['Method'].values
    volatilities = table_sorted['Ann. Volatility'].values
    
    # Color coding: our 3D best (green), paper's 2D (blue), baselines (gray), SCM (red)
    colors = []
    for m in methods:
        if '3D' in m:
            colors.append('#2ecc71')  # Green for our 3D
        elif 'Dual' in m:
            colors.append('#3498db')  # Blue for paper's 2D
        elif m == 'SCM':
            colors.append('#e74c3c')  # Red for SCM baseline
        elif 'Tyler' in m:
            colors.append('#9b59b6')  # Purple for Tyler
        else:
            colors.append('#95a5a6')  # Gray for others
    
    bars1 = ax1.barh(methods, volatilities, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Annualized Volatility', fontsize=12)
    ax1.set_title('Portfolio Volatility by Estimator\n(Lower is Better)', fontsize=14)
    
    # Find SCM index
    scm_idx = np.where(methods == 'SCM')[0]
    if len(scm_idx) > 0:
        ax1.axvline(x=volatilities[scm_idx[0]], 
                    color='#e74c3c', linestyle='--', alpha=0.7, linewidth=2,
                    label='SCM Baseline')
    ax1.legend(loc='lower right')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, vol in zip(bars1, volatilities):
        ax1.text(vol + 0.002, bar.get_y() + bar.get_height()/2, 
                f'{vol:.4f}', va='center', fontsize=9)
    
    # === Right plot: Improvement vs SCM ===
    ax2 = axes[1]
    improvements = table_sorted['Improvement vs SCM (%)'].values
    
    # Color based on positive/negative improvement
    colors2 = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
    
    bars2 = ax2.barh(methods, improvements, color=colors2, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Improvement vs SCM (%)', fontsize=12)
    ax2.set_title('Improvement Over Sample Covariance Matrix\n(Higher is Better)', fontsize=14)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, imp in zip(bars2, improvements):
        offset = 0.5 if imp >= 0 else -0.5
        ha = 'left' if imp >= 0 else 'right'
        ax2.text(imp + offset, bar.get_y() + bar.get_height()/2, 
                f'{imp:.2f}%', va='center', ha=ha, fontsize=9)
    
    plt.suptitle('Covariance Matrix Estimator Comparison (Table I Replication)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison bar chart saved to: {save_path}")
    plt.close()


def plot_heatmap_2d(results, save_path='heatmap_2d_dual.png'):
    """
    Plot heatmap for paper's 2D dual method.
    
    Parameters:
        results: Dictionary from run_paper_comparison()
        save_path: Path to save the figure
    """
    dual_2d = results['dual_method_2d']
    avg_performance = dual_2d['avg_performance']
    theta_grid = dual_2d['theta_grid']
    phi_grid = dual_2d['phi_grid']
    
    # Convert to annualized volatility
    vol_surface = np.sqrt(avg_performance * 252)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(vol_surface.T, cmap='RdYlGn_r', aspect='auto', origin='lower',
                   extent=[theta_grid[0], theta_grid[-1], phi_grid[0], phi_grid[-1]])
    
    # Mark optimal point
    ax.plot(dual_2d['theta_opt'], dual_2d['phi_opt'], 'b*', markersize=20,
            label=f"Optimal: θ={dual_2d['theta_opt']:.2f}, φ={dual_2d['phi_opt']:.2f}")
    
    ax.set_xlabel('θ (Shrinkage Target F vs Marchenko-Pastur MP)', fontsize=12)
    ax.set_ylabel('φ (Regularized vs Sample Covariance Matrix)', fontsize=12)
    ax.set_title("Paper's 2D Dual Method: Σ*(θ,φ) = φ[θF + (1-θ)MP] + (1-φ)SCM", fontsize=14)
    
    plt.colorbar(im, ax=ax, label='Annualized Volatility')
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 2D heatmap saved to: {save_path}")
    plt.close()


def plot_heatmap_3d_slices(results, save_path='heatmap_3d_slices.png'):
    """
    Plot 2D slices of the 3D optimization surface.
    
    Creates a 2x2 subplot showing θ-φ, θ-ψ, and φ-ψ slices at optimal values.
    
    Parameters:
        results: Dictionary from run_paper_comparison()
        save_path: Path to save the figure
    """
    ext_3d = results['our_3d_method']
    avg_performance = ext_3d['avg_performance']
    theta_grid = ext_3d['theta_grid']
    phi_grid = ext_3d['phi_grid']
    psi_grid = ext_3d['psi_grid']
    
    theta_opt = ext_3d['theta_opt']
    phi_opt = ext_3d['phi_opt']
    psi_opt = ext_3d['psi_opt']
    
    # Find optimal indices
    theta_opt_idx = np.argmin(np.abs(theta_grid - theta_opt))
    phi_opt_idx = np.argmin(np.abs(phi_grid - phi_opt))
    psi_opt_idx = np.argmin(np.abs(psi_grid - psi_opt))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # === Subplot 1: θ vs φ at optimal ψ ===
    ax1 = axes[0, 0]
    slice_theta_phi = avg_performance[:, :, psi_opt_idx]
    vol_theta_phi = np.sqrt(slice_theta_phi * 252)
    
    im1 = ax1.imshow(vol_theta_phi.T, cmap='RdYlGn_r', aspect='auto', origin='lower')
    ax1.plot(theta_opt_idx, phi_opt_idx, 'b*', markersize=15)
    ax1.set_xticks(range(len(theta_grid)))
    ax1.set_yticks(range(len(phi_grid)))
    ax1.set_xticklabels([f'{x:.1f}' for x in theta_grid])
    ax1.set_yticklabels([f'{y:.1f}' for y in phi_grid])
    ax1.set_xlabel('θ (F vs RobustBlend)', fontsize=10)
    ax1.set_ylabel('φ (Regularized vs SCM)', fontsize=10)
    ax1.set_title(f'θ vs φ at ψ={psi_opt:.2f}', fontsize=12)
    plt.colorbar(im1, ax=ax1, label='Ann. Volatility')
    
    # === Subplot 2: θ vs ψ at optimal φ ===
    ax2 = axes[0, 1]
    slice_theta_psi = avg_performance[:, phi_opt_idx, :]
    vol_theta_psi = np.sqrt(slice_theta_psi * 252)
    
    im2 = ax2.imshow(vol_theta_psi.T, cmap='RdYlGn_r', aspect='auto', origin='lower')
    ax2.plot(theta_opt_idx, psi_opt_idx, 'b*', markersize=15)
    ax2.set_xticks(range(len(theta_grid)))
    ax2.set_yticks(range(len(psi_grid)))
    ax2.set_xticklabels([f'{x:.1f}' for x in theta_grid])
    ax2.set_yticklabels([f'{y:.1f}' for y in psi_grid])
    ax2.set_xlabel('θ (F vs RobustBlend)', fontsize=10)
    ax2.set_ylabel('ψ (MP vs Tyler)', fontsize=10)
    ax2.set_title(f'θ vs ψ at φ={phi_opt:.2f}', fontsize=12)
    plt.colorbar(im2, ax=ax2, label='Ann. Volatility')
    
    # === Subplot 3: φ vs ψ at optimal θ ===
    ax3 = axes[1, 0]
    slice_phi_psi = avg_performance[theta_opt_idx, :, :]
    vol_phi_psi = np.sqrt(slice_phi_psi * 252)
    
    im3 = ax3.imshow(vol_phi_psi.T, cmap='RdYlGn_r', aspect='auto', origin='lower')
    ax3.plot(phi_opt_idx, psi_opt_idx, 'b*', markersize=15)
    ax3.set_xticks(range(len(phi_grid)))
    ax3.set_yticks(range(len(psi_grid)))
    ax3.set_xticklabels([f'{x:.1f}' for x in phi_grid])
    ax3.set_yticklabels([f'{y:.1f}' for y in psi_grid])
    ax3.set_xlabel('φ (Regularized vs SCM)', fontsize=10)
    ax3.set_ylabel('ψ (MP vs Tyler)', fontsize=10)
    ax3.set_title(f'φ vs ψ at θ={theta_opt:.2f}', fontsize=12)
    plt.colorbar(im3, ax=ax3, label='Ann. Volatility')
    
    # === Subplot 4: Summary text ===
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    best_vol = np.sqrt(ext_3d['avg_variance'] * 252)
    summary_text = f"""
    OPTIMAL HYPERPARAMETERS
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    θ = {theta_opt:.2f}  (F vs RobustBlend)
        θ=0: Pure RobustBlend (MP/Tyler mix)
        θ=1: Pure Shrinkage Target (F)
    
    φ = {phi_opt:.2f}  (Regularized vs SCM)
        φ=0: Pure Sample Covariance (SCM)
        φ=1: Pure Regularized Estimator
    
    ψ = {psi_opt:.2f}  (MP vs Tyler)
        ψ=0: Pure Tyler's M-Estimator
        ψ=1: Pure Marchenko-Pastur (MP)
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Best Volatility: {best_vol:.4f}
    """
    
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, 
             fontsize=11, verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('3D Hyperparameter Optimization: Σ*(θ,φ,ψ)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 3D heatmap slices saved to: {save_path}")
    plt.close()


def generate_all_plots(results, table, output_dir='.'):
    """
    Generate all visualization plots.
    
    Parameters:
        results: Dictionary from run_paper_comparison()
        table: DataFrame from generate_comparison_table()
        output_dir: Directory to save plots
    """
    import os
    
    print("\nGenerating visualizations...")
    
    plot_comparison_bar_chart(table, save_path=os.path.join(output_dir, 'comparison_bar_chart.png'))
    plot_heatmap_2d(results, save_path=os.path.join(output_dir, 'heatmap_2d_dual.png'))
    plot_heatmap_3d_slices(results, save_path=os.path.join(output_dir, 'heatmap_3d_slices.png'))
    
    print("✓ All visualizations generated successfully!")


# ============================================================================
# Paper-Style Tables (Table I and II format)
# ============================================================================

# Display names for datasets
DATASET_DISPLAY_NAMES = {
    'sp500': 'S&P',
    'nasdaq100': 'NASDAQ',
    'nikkei225': 'NIKKEI',
    'nifty50': 'NSE',
    'bse_sensex': 'BSE',
    'ftse100': 'FTSE',
}

# Estimator display names (matching paper format)
ESTIMATOR_ORDER = [
    ('Identity (I)', 'Σ_Identity'),
    ('Scaled Identity (sigma^2*I)', 'Σ_Scaled'),
    ('SCM', 'Σ_SCM'),
    ('Shrinkage Target (F)', 'Σ_Target'),
    ('Marchenko-Pastur (MP)', 'Σ_MP'),
    ("Tyler's M-Estimator", 'Σ_Tyler'),
    ('paper_2d', 'Σ* (Paper 2D)'),
    ('our_3d', 'Σ* (Our 3D)'),
]


def extract_volatility(results, estimator_key):
    """Extract annualized volatility for an estimator from results."""
    if estimator_key in ['paper_2d', 'our_3d']:
        # Combined estimators
        if estimator_key == 'paper_2d':
            data = results.get('paper_2d', results.get('dual_method_2d', {}))
        else:
            data = results.get('our_3d', results.get('our_3d_method', {}))
        
        best_vol = data.get('best_vol', data.get('avg_variance'))
        if best_vol is not None:
            if best_vol < 0.1:  # Already in variance form
                return np.sqrt(best_vol * 252) * 100
            return best_vol * 100
    else:
        # Fixed estimators
        fixed = results.get('fixed_estimators', results.get('fixed', {}))
        if estimator_key in fixed:
            var = fixed[estimator_key].get('avg_variance', fixed[estimator_key].get('variance'))
            if var is not None:
                return np.sqrt(var * 252) * 100
    
    return None


def generate_paper_tables(all_results, frequencies, save_latex=True):
    """
    Generate paper-style tables (Table I and Table II format).
    
    Paper format:
    - Rows: Estimators (Σ_Identity, Σ_Shrink, Σ_SCM, Σ_MP, Σ_RIE, Σ*)
    - Columns: Datasets grouped by frequency (30 days, 60 days, 90 days)
    
    Parameters:
        all_results: Dict of {(dataset, freq): results}
        frequencies: List of frequencies [30, 60, 90]
        save_latex: Whether to print LaTeX tables
    """
    # Get unique datasets
    datasets = sorted(list(set(ds for ds, _ in all_results.keys())))
    
    # Define Table I datasets (as in paper) and Table II datasets
    table1_datasets = [ds for ds in datasets if ds in ['nifty50', 'nikkei225', 'sp500', 'bse_sensex']]
    table2_datasets = [ds for ds in datasets if ds in ['nasdaq100', 'ftse100']]
    
    # If we don't have enough for separate tables, combine
    if len(table1_datasets) < 2 and len(table2_datasets) < 2:
        table1_datasets = datasets
        table2_datasets = []
    
    def build_table(datasets_subset, table_name):
        """Build a single paper-style table."""
        if not datasets_subset:
            return None
        
        # Build table data
        table_data = []
        
        for est_key, est_name in ESTIMATOR_ORDER:
            row = {'Estimator': est_name}
            
            for freq in frequencies:
                for ds in datasets_subset:
                    if (ds, freq) in all_results:
                        vol = extract_volatility(all_results[(ds, freq)], est_key)
                        col_name = f"{DATASET_DISPLAY_NAMES.get(ds, ds)} ({freq}d)"
                        row[col_name] = vol
            
            # Only add row if it has at least one value
            if any(v is not None for k, v in row.items() if k != 'Estimator'):
                table_data.append(row)
        
        if not table_data:
            return None
        
        df = pd.DataFrame(table_data)
        return df
    
    # Print Table I
    if table1_datasets:
        table1 = build_table(table1_datasets, "Table I")
        if table1 is not None:
            print(f"\n{'='*80}")
            print(f"TABLE I: Annualized Volatility (%) - {', '.join(DATASET_DISPLAY_NAMES.get(d, d) for d in table1_datasets)}")
            print(f"{'='*80}")
            
            # Format the table nicely
            print(format_paper_table(table1))
            
            if save_latex:
                print("\n--- LaTeX ---")
                print(table1.to_latex(index=False, float_format="%.2f", na_rep='-'))
    
    # Print Table II
    if table2_datasets:
        table2 = build_table(table2_datasets, "Table II")
        if table2 is not None:
            print(f"\n{'='*80}")
            print(f"TABLE II: Annualized Volatility (%) - {', '.join(DATASET_DISPLAY_NAMES.get(d, d) for d in table2_datasets)}")
            print(f"{'='*80}")
            
            print(format_paper_table(table2))
            
            if save_latex:
                print("\n--- LaTeX ---")
                print(table2.to_latex(index=False, float_format="%.2f", na_rep='-'))
    
    # Also print combined summary table
    print(f"\n{'='*80}")
    print("SUMMARY: Best Performance by Dataset and Frequency")
    print(f"{'='*80}")
    print_summary_table(all_results, frequencies)


def format_paper_table(df):
    """Format DataFrame for nice console output."""
    # Convert None to '-'
    df_display = df.fillna('-')
    
    # Format numeric columns
    for col in df_display.columns:
        if col != 'Estimator':
            df_display[col] = df_display[col].apply(
                lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x
            )
    
    return df_display.to_string(index=False)


def print_summary_table(all_results, frequencies):
    """Print a summary showing best method for each dataset/frequency."""
    rows = []
    
    for (dataset, freq), results in sorted(all_results.items()):
        ds_name = DATASET_DISPLAY_NAMES.get(dataset, dataset)
        
        # Find best method
        best_method = None
        best_vol = float('inf')
        
        for est_key, est_name in ESTIMATOR_ORDER:
            vol = extract_volatility(results, est_key)
            if vol is not None and vol < best_vol:
                best_vol = vol
                best_method = est_name
        
        # Get SCM baseline
        scm_vol = extract_volatility(results, 'SCM')
        
        # Calculate improvement
        if scm_vol is not None and best_vol < float('inf'):
            improvement = (scm_vol - best_vol) / scm_vol * 100
        else:
            improvement = None
        
        rows.append({
            'Dataset': ds_name,
            'Frequency': f'{freq}d',
            'Best Method': best_method,
            'Best Vol (%)': best_vol if best_vol < float('inf') else None,
            'SCM Vol (%)': scm_vol,
            'Improvement': f'{improvement:.2f}%' if improvement else '-',
        })
    
    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))
    
    # Print overall best
    if rows:
        best_overall = min(rows, key=lambda x: x['Best Vol (%)'] or float('inf'))
        print(f"\nOverall Best: {best_overall['Dataset']} {best_overall['Frequency']} - "
              f"{best_overall['Best Method']} at {best_overall['Best Vol (%)']:.2f}%")


def generate_paper_figure(all_results, frequencies, save_path='paper_results.png'):
    """
    Generate a figure similar to the paper's results visualization.
    
    Creates a grouped bar chart showing volatility by estimator for each 
    dataset and frequency combination.
    """
    datasets = sorted(list(set(ds for ds, _ in all_results.keys())))
    
    if not datasets:
        print("No results to plot.")
        return
    
    n_freq = len(frequencies)
    n_datasets = len(datasets)
    
    fig, axes = plt.subplots(1, n_freq, figsize=(6*n_freq, 8), sharey=True)
    if n_freq == 1:
        axes = [axes]
    
    for ax_idx, freq in enumerate(frequencies):
        ax = axes[ax_idx]
        
        # Collect data for this frequency
        x_positions = np.arange(len(ESTIMATOR_ORDER))
        width = 0.8 / n_datasets
        
        for ds_idx, ds in enumerate(datasets):
            if (ds, freq) not in all_results:
                continue
            
            results = all_results[(ds, freq)]
            vols = []
            
            for est_key, _ in ESTIMATOR_ORDER:
                vol = extract_volatility(results, est_key)
                vols.append(vol if vol is not None else 0)
            
            offset = (ds_idx - n_datasets/2 + 0.5) * width
            bars = ax.bar(x_positions + offset, vols, width, 
                         label=DATASET_DISPLAY_NAMES.get(ds, ds), alpha=0.8)
        
        ax.set_xlabel('Estimator', fontsize=11)
        ax.set_ylabel('Annualized Volatility (%)', fontsize=11)
        ax.set_title(f'{freq}-Day Rebalancing', fontsize=12, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels([name for _, name in ESTIMATOR_ORDER], rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Portfolio Volatility: Paper Replication Results', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Paper results figure saved to: {save_path}")
    plt.close()

