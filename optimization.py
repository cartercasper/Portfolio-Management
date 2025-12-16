import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from estimators import MP_est, sample_cov, shrinkage_target, tyler_m_estimator
import scipy.linalg as la
import warnings

# SCM is sample covariance matrix
# SMP is covariance matrix reconstructed from Marchenko-Pastur clipped eigenvalues
# ST is shrinkage target matrix 

#######################################
# Section III Implementation (Extended with Tyler's M-Estimator)
#######################################
# Key fixes implemented:
# 1. Removed double-demeaning (let pandas .cov() handle it)
# 2. Changed to GMVP (no return constraint) to match paper's focus on covariance
# 3. Fixed MP estimator to replace noise with mean of SIGNAL eigenvalues
# 4. Global (θ, φ, ψ) selection by averaging across ALL periods (no look-ahead bias)
#
# Performance optimizations:
# 5. Parallelize over PERIODS (not grid points) - reduces serialization overhead
# 6. Precompute matrix differences D1, D2, D3 for faster sigma_star
# 7. Analytical GMVP solution when possible (avoids QP solver overhead)
#
# Three-way mixing formula (with Tyler):
# Σ*(θ,φ,ψ) = φ[θF + (1-θ)(ψ·MP + (1-ψ)·Tyler)] + (1-φ)·SCM
#######################################


def _solve_gmvp_analytical(sigma_star, weight_bounds=(0, 1)):
    """
    Solve GMVP analytically when possible.
    
    Analytical solution: p* = Σ⁻¹ 1 / (1ᵀ Σ⁻¹ 1)
    
    Falls back to QP if bounds are violated.
    
    Parameters:
    -----------
    sigma_star : np.ndarray
        Covariance matrix
    weight_bounds : tuple
        (lower, upper) bounds for weights
    
    Returns:
    --------
    p : np.ndarray or None
        Portfolio weights (None if analytical solution violates bounds)
    """
    n = sigma_star.shape[0]
    ones = np.ones(n)
    
    try:
        # Use Cholesky decomposition for numerical stability (faster than direct inverse)
        L = la.cholesky(sigma_star, lower=True)
        # Solve L @ y = ones
        y = la.solve_triangular(L, ones, lower=True)
        # Solve L.T @ x = y  ->  x = Σ⁻¹ @ ones
        sigma_inv_ones = la.solve_triangular(L.T, y, lower=False)
        
        # p* = Σ⁻¹ 1 / (1ᵀ Σ⁻¹ 1)
        denom = ones @ sigma_inv_ones
        p = sigma_inv_ones / denom
        
        # Check if bounds are satisfied
        if np.all(p >= weight_bounds[0] - 1e-8) and np.all(p <= weight_bounds[1] + 1e-8):
            # Clip to bounds (for numerical precision)
            p = np.clip(p, weight_bounds[0], weight_bounds[1])
            # Renormalize to sum to 1
            p = p / p.sum()
            return p
        else:
            # Bounds violated, need QP
            return None
    except (la.LinAlgError, np.linalg.LinAlgError):
        # Matrix not positive definite
        return None


def _process_single_period(args):
    """
    Process a single rebalancing period - computes performance for ALL (θ,φ,ψ) combinations.
    
    This is called in parallel across periods (not across grid points).
    Much more efficient because we only serialize data once per period.
    
    Three-way mixing formula (with Tyler):
    Σ*(θ,φ,ψ) = φ[θF + (1-θ)(ψ·MP + (1-ψ)·Tyler)] + (1-φ)·SCM
    
    Parameters:
    -----------
    args : tuple
        (period_idx, t0, returns_values, columns, N_train, N_test, 
         theta_grid, phi_grid, psi_grid, use_gmvp)
    
    Returns:
    --------
    tuple : (period_idx, results_grid)
        period_idx: which period this is
        results_grid: 3D array of variances for all (θ,φ,ψ) combinations
    """
    period_idx, t0, returns_values, columns, N_train, N_test, theta_grid, phi_grid, psi_grid, use_gmvp = args
    
    grid_size_theta = len(theta_grid)
    grid_size_phi = len(phi_grid)
    grid_size_psi = len(psi_grid)
    results_grid = np.zeros((grid_size_theta, grid_size_phi, grid_size_psi))
    
    # Split data
    R_train = returns_values[t0:t0+N_train]
    R_test = returns_values[t0+N_train:t0+N_train+N_test]
    
    # Convert to DataFrame for estimator functions (they expect DataFrames)
    R_train_df = pd.DataFrame(R_train, columns=columns)
    
    # Compute estimators ONCE per period
    SCM_df = sample_cov(R_train_df)
    F_df = shrinkage_target(SCM_df)
    MP_arr = MP_est(R_train_df, SCM_df)
    Tyler_arr = tyler_m_estimator(R_train_df, SCM_df)  # NEW: Tyler's M-estimator
    
    # Convert to numpy
    SCM = SCM_df.values
    F = F_df.values
    MP = MP_arr
    Tyler = Tyler_arr
    
    # OPTIMIZATION: Precompute differences for efficient sigma_star computation
    # 
    # Full formula: Σ*(θ,φ,ψ) = φ[θF + (1-θ)(ψ·MP + (1-ψ)·Tyler)] + (1-φ)·SCM
    # 
    # Let RobustBlend = ψ·MP + (1-ψ)·Tyler = Tyler + ψ·(MP - Tyler) = Tyler + ψ·D3
    # Then: Σ* = φ[θF + (1-θ)·RobustBlend] + (1-φ)·SCM
    #          = φ·θ·F + φ·(1-θ)·RobustBlend + (1-φ)·SCM
    #          = SCM + φ·(RobustBlend - SCM) + φ·θ·(F - RobustBlend)
    # 
    # Precompute:
    D3 = MP - Tyler      # MP - Tyler (for RobustBlend computation)
    D_F_Tyler = F - Tyler  # F - Tyler (for final mixing)
    D_Tyler_SCM = Tyler - SCM  # Tyler - SCM (base difference)
    
    # Grid search over (θ, φ, ψ)
    for i, theta in enumerate(theta_grid):
        for j, phi in enumerate(phi_grid):
            for k, psi in enumerate(psi_grid):
                # Compute RobustBlend = Tyler + ψ·(MP - Tyler)
                # sigma_star = SCM + φ·(RobustBlend - SCM) + φ·θ·(F - RobustBlend)
                #            = SCM + φ·(Tyler + ψ·D3 - SCM) + φ·θ·(F - Tyler - ψ·D3)
                #            = SCM + φ·D_Tyler_SCM + φ·ψ·D3 + φ·θ·D_F_Tyler - φ·θ·ψ·D3
                #            = SCM + φ·D_Tyler_SCM + φ·ψ·D3·(1 - θ) + φ·θ·D_F_Tyler
                
                sigma_star = (SCM + phi * D_Tyler_SCM + 
                             phi * psi * (1 - theta) * D3 + 
                             phi * theta * D_F_Tyler)
                
                # Ensure PSD with adaptive ridge
                min_eig = np.linalg.eigvalsh(sigma_star).min()
                if min_eig < 1e-8:
                    sigma_star = sigma_star + (abs(min_eig) + 1e-6) * np.eye(sigma_star.shape[0])
                else:
                    sigma_star = sigma_star + 1e-8 * np.eye(sigma_star.shape[0])
                
                # OPTIMIZATION: Try analytical GMVP first
                if use_gmvp:
                    p = _solve_gmvp_analytical(sigma_star)
                    if p is None:
                        # Fallback to QP solver
                        p = _solve_markowitz_fast(sigma_star)
                else:
                    p = _solve_markowitz_fast(sigma_star)
                
                if p is not None:
                    # Realized returns on test period
                    portfolio_returns = R_test @ p
                    results_grid[i, j, k] = np.var(portfolio_returns)
                else:
                    results_grid[i, j, k] = np.inf
    
    return (period_idx, results_grid)


def _solve_markowitz_fast(sigma_star, weight_bounds=(0, 1)):
    """
    Fast Markowitz solver for GMVP (no return constraint).
    
    Parameters:
    -----------
    sigma_star : np.ndarray
        Covariance matrix (already PSD-enforced)
    weight_bounds : tuple
        (lower, upper) bounds
    
    Returns:
    --------
    p : np.ndarray or None
        Portfolio weights
    """
    n_assets = sigma_star.shape[0]
    
    p = cp.Variable(n_assets)
    objective = cp.Minimize(cp.quad_form(p, sigma_star))
    constraints = [
        cp.sum(p) == 1,
        p >= weight_bounds[0],
        p <= weight_bounds[1]
    ]
    
    problem = cp.Problem(objective, constraints)
    
    try:
        # Suppress CVXPY warnings about inaccurate solutions
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Use OSQP with adjusted settings for better convergence
            problem.solve(solver=cp.OSQP, verbose=False, 
                         eps_abs=1e-5, eps_rel=1e-5,  # Slightly looser tolerance
                         max_iter=4000,               # More iterations allowed
                         polish=True)                 # Polish solution for accuracy
        
        if problem.status in ['optimal', 'optimal_inaccurate']:
            return p.value
        else:
            return None
    except Exception:
        return None


def optimize_portfolio(returns, N_train=200, N_test=30, rebalance_freq=30, 
                      grid_size=6, n_jobs=-1, use_gmvp=True, verbose=True):
    """
    Find optimal (θ, φ, ψ) using rolling window approach with Tyler's M-estimator.
    
    Implements Section III of the paper with three-way mixing:
    Σ*(θ,φ,ψ) = φ[θF + (1-θ)(ψ·MP + (1-ψ)·Tyler)] + (1-φ)·SCM
    
    Grid search over (θ, φ, ψ) to find the combination that minimizes 
    average out-of-sample portfolio variance.
    
    OPTIMIZATIONS:
    - Parallelizes over PERIODS (not grid points) - much less serialization
    - Precomputes matrix differences for faster sigma_star computation
    - Uses analytical GMVP when bounds allow (100x faster than QP)
    - Caches estimators (SCM, F, MP, Tyler) per period
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns data (T × M)
    N_train : int
        Training window size (default: 200 days)
    N_test : int
        Test window size (default: 30 days)
    rebalance_freq : int
        Rebalancing frequency (default: 30 days)
    grid_size : int
        Number of grid points for θ, φ, and ψ (default: 6)
        Use 6 for quick testing (216 combinations)
        Use 11 for paper-level precision (1331 combinations)
    n_jobs : int
        Number of parallel workers (default: -1 uses all CPUs)
        Set to 1 to disable parallelization
    use_gmvp : bool
        If True, use Global Minimum Variance Portfolio (no return constraint)
        If False, add minimum return constraint (may cause infeasibility)
        Paper focuses on covariance estimation, so GMVP is recommended
    verbose : bool
        Print progress information
    
    Returns:
    --------
    results : dict
        Dictionary containing:
        - theta_opt: optimal θ value (F vs RobustBlend mixing)
        - phi_opt: optimal φ value (regularized vs SCM mixing)
        - psi_opt: optimal ψ value (MP vs Tyler mixing)
        - best_variance: average out-of-sample variance
        - avg_performance: 3D array of average performance (θ × φ × ψ)
        - performance_4d: 4D array of all period performances
        - theta_grid, phi_grid, psi_grid: parameter grids
    """
    # Setup grids for all three parameters
    theta_grid = np.linspace(0, 1, grid_size)
    phi_grid = np.linspace(0, 1, grid_size)
    psi_grid = np.linspace(0, 1, grid_size)
    
    total_combinations = grid_size ** 3
    
    # Determine number of workers
    if n_jobs == -1:
        n_jobs = cpu_count()
    n_jobs = max(1, n_jobs)
    
    # Identify rebalancing times
    T = len(returns)
    rebalance_times = list(range(0, T - N_train - N_test, rebalance_freq))
    num_periods = len(rebalance_times)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"HYPERPARAMETER OPTIMIZATION (with Tyler's M-Estimator)")
        print(f"{'='*60}")
        print(f"Total periods: {T}")
        print(f"Training window: {N_train} days")
        print(f"Test window: {N_test} days")
        print(f"Rebalancing frequency: {rebalance_freq} days")
        print(f"Number of rebalancing periods: {num_periods}")
        print(f"Grid size: {grid_size}×{grid_size}×{grid_size} = {total_combinations} combinations")
        print(f"Parameters: θ (F vs RobustBlend), φ (Regularized vs SCM), ψ (MP vs Tyler)")
        print(f"Parallel workers: {n_jobs} CPUs")
        print(f"Portfolio type: {'GMVP (no return constraint)' if use_gmvp else 'Mean-variance (with return constraint)'}")
        print(f"Optimizations: period-parallel, precomputed diffs, analytical GMVP")
        print(f"{'='*60}\n")
    
    # Storage: (θ × φ × ψ × num_periods)
    performance = np.zeros((grid_size, grid_size, grid_size, num_periods))
    
    # Convert returns to numpy once (avoid repeated conversion)
    returns_values = returns.values
    columns = returns.columns.tolist()
    
    # Prepare arguments for each period
    period_args = [
        (period_idx, t0, returns_values, columns, N_train, N_test, 
         theta_grid, phi_grid, psi_grid, use_gmvp)
        for period_idx, t0 in enumerate(rebalance_times)
    ]
    
    if n_jobs > 1 and num_periods > 1:
        # OPTIMIZATION: Parallelize over PERIODS (not grid points)
        # This drastically reduces serialization overhead
        if verbose:
            print(f"Processing {num_periods} periods in parallel...")
        
        with Pool(n_jobs) as pool:
            # Use imap for progress tracking
            results_iter = pool.imap(_process_single_period, period_args)
            
            if verbose:
                results_list = list(tqdm(results_iter, total=num_periods, desc="Rebalancing periods"))
            else:
                results_list = list(results_iter)
        
        # Unpack results
        for period_idx, results_grid in results_list:
            performance[:, :, :, period_idx] = results_grid
    else:
        # Serial execution (for debugging or single period)
        iterator = tqdm(period_args, desc="Rebalancing periods") if verbose else period_args
        for args in iterator:
            period_idx, results_grid = _process_single_period(args)
            performance[:, :, :, period_idx] = results_grid
    
    # Average across periods
    avg_performance = np.mean(performance, axis=3)
    
    # Find optimal (θ, φ, ψ)
    optimal_idx = np.unravel_index(np.argmin(avg_performance), avg_performance.shape)
    theta_opt = theta_grid[optimal_idx[0]]
    phi_opt = phi_grid[optimal_idx[1]]
    psi_opt = psi_grid[optimal_idx[2]]
    best_variance = avg_performance[optimal_idx]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"OPTIMIZATION RESULTS")
        print(f"{'='*60}")
        print(f"Optimal θ: {theta_opt:.2f} (F vs RobustBlend)")
        print(f"Optimal φ: {phi_opt:.2f} (Regularized vs SCM)")
        print(f"Optimal ψ: {psi_opt:.2f} (MP vs Tyler)")
        print(f"Average out-of-sample variance: {best_variance:.8f}")
        print(f"Average out-of-sample volatility (annualized): {np.sqrt(best_variance * 252):.4f}")
        print(f"{'='*60}\n")
    
    # Return results
    return {
        'theta_opt': theta_opt,
        'phi_opt': phi_opt,
        'psi_opt': psi_opt,
        'best_variance': best_variance,
        'avg_performance': avg_performance,
        'performance_4d': performance,
        'theta_grid': theta_grid,
        'phi_grid': phi_grid,
        'psi_grid': psi_grid
    }

def ensure_psd(matrix, epsilon=1e-8):
    """
    Ensure matrix is positive semi-definite by adding small ridge.
    
    Parameters:
    -----------
    matrix : np.ndarray or pd.DataFrame
        Covariance matrix
    epsilon : float
        Ridge parameter
    
    Returns:
    --------
    psd_matrix : np.ndarray
        Positive semi-definite matrix
    """
    # Convert to numpy if pandas DataFrame
    if isinstance(matrix, pd.DataFrame):
        matrix = matrix.values
    
    # Add small ridge to diagonal
    psd_matrix = matrix + epsilon * np.eye(matrix.shape[0])
    
    # Verify PSD by checking eigenvalues
    eigvals = np.linalg.eigvalsh(psd_matrix)
    if np.min(eigvals) < 0:
        # If still negative, add larger ridge
        psd_matrix = matrix + (abs(np.min(eigvals)) + epsilon) * np.eye(matrix.shape[0])
    
    return psd_matrix


def solve_markowitz(sigma_star, g, r_daily=None, weight_bounds=(0, 1)):
    """
    Solve the Markowitz portfolio optimization QP.
    
    min_p  p^T Σ*(θ,φ) p
    s.t.   1^T p = 1
           p ≥ 0
           p ≤ 1
           g^T p ≥ r_daily (optional)
    
    Parameters:
    -----------
    sigma_star : np.ndarray or pd.DataFrame
        Combined covariance estimator Σ*(θ,φ)
    g : np.ndarray
        Expected returns vector (not used if r_daily=None)
    r_daily : float or None
        Minimum daily return threshold (default: None for GMVP - Global Minimum Variance Portfolio)
        Paper focuses on covariance estimation, not return forecasting
    weight_bounds : tuple
        (lower, upper) bounds for portfolio weights
    
    Returns:
    --------
    p : np.ndarray
        Optimal portfolio weights (None if infeasible)
    """
    # Convert to numpy if DataFrame
    if isinstance(sigma_star, pd.DataFrame):
        sigma_star = sigma_star.values
    
    # Ensure PSD
    sigma_star = ensure_psd(sigma_star)
    
    n_assets = sigma_star.shape[0]
    
    # Define optimization variable
    p = cp.Variable(n_assets)
    
    # Objective: minimize portfolio variance
    objective = cp.Minimize(cp.quad_form(p, sigma_star))
    
    # Constraints
    constraints = [
        cp.sum(p) == 1,              # fully invested
        p >= weight_bounds[0],       # lower bound (no shorting)
        p <= weight_bounds[1]        # upper bound (no leverage)
    ]
    
    # Add return constraint if specified
    if r_daily is not None:
        constraints.append(g @ p >= r_daily)
    
    # Solve
    problem = cp.Problem(objective, constraints)
    
    try:
        problem.solve(solver=cp.OSQP, verbose=False)
        
        if problem.status in ['optimal', 'optimal_inaccurate']:
            return p.value
        else:
            # Infeasible or other issue
            return None
    except Exception as e:
        # Solver error
        return None


def plot_performance_heatmap(results, save_path='performance_heatmap.png'):
    """
    Plot heatmaps of average performance across (θ, φ) grid at optimal ψ,
    and (θ, ψ) at optimal φ, and (φ, ψ) at optimal θ.
    
    Creates a 2x2 subplot with three 2D slices of the 3D performance surface.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from optimize_portfolio
    save_path : str
        Path to save figure
    """
    avg_performance = results['avg_performance']  # Shape: (θ, φ, ψ)
    theta_grid = results['theta_grid']
    phi_grid = results['phi_grid']
    psi_grid = results['psi_grid']
    
    theta_opt = results['theta_opt']
    phi_opt = results['phi_opt']
    psi_opt = results['psi_opt']
    
    # Get indices for optimal values
    theta_opt_idx = np.argmin(np.abs(theta_grid - theta_opt))
    phi_opt_idx = np.argmin(np.abs(phi_grid - phi_opt))
    psi_opt_idx = np.argmin(np.abs(psi_grid - psi_opt))
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # ====================
    # Subplot 1: θ vs φ at optimal ψ
    # ====================
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
    
    # ====================
    # Subplot 2: θ vs ψ at optimal φ
    # ====================
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
    
    # ====================
    # Subplot 3: φ vs ψ at optimal θ
    # ====================
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
    
    # ====================
    # Subplot 4: Summary text
    # ====================
    ax4 = axes[1, 1]
    ax4.axis('off')
    
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
    Best Volatility: {np.sqrt(results['best_variance'] * 252):.4f}
    """
    
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, 
             fontsize=11, verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('3D Hyperparameter Optimization Results (Tyler + MP + F + SCM)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Performance heatmap saved to: {save_path}")
    plt.close()


def compare_estimators(results):
    """
    Compare the optimal combination with baseline estimators.
    
    Now includes Tyler's M-estimator in the comparisons.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from optimize_portfolio
    
    Returns:
    --------
    comparison : pd.DataFrame
        Comparison table
    """
    avg_performance = results['avg_performance']
    theta_grid = results['theta_grid']
    phi_grid = results['phi_grid']
    psi_grid = results['psi_grid']
    
    # Find indices for extreme values (0 and 1)
    theta_0_idx = np.argmin(np.abs(theta_grid - 0))
    theta_1_idx = np.argmin(np.abs(theta_grid - 1))
    phi_0_idx = np.argmin(np.abs(phi_grid - 0))
    phi_1_idx = np.argmin(np.abs(phi_grid - 1))
    psi_0_idx = np.argmin(np.abs(psi_grid - 0))
    psi_1_idx = np.argmin(np.abs(psi_grid - 1))
    
    # Extract performance for key combinations
    # Remember: avg_performance is indexed [theta, phi, psi]
    # Formula: Σ*(θ,φ,ψ) = φ[θF + (1-θ)(ψ·MP + (1-ψ)·Tyler)] + (1-φ)·SCM
    comparisons = {
        # SCM Only: φ=0 means pure SCM (θ and ψ don't matter)
        'SCM Only (φ=0)': avg_performance[theta_0_idx, phi_0_idx, psi_0_idx],
        
        # MP Only: φ=1, θ=0, ψ=1 means: 1·[0·F + 1·(1·MP + 0·Tyler)] = MP
        'MP Only (θ=0, φ=1, ψ=1)': avg_performance[theta_0_idx, phi_1_idx, psi_1_idx],
        
        # Tyler Only: φ=1, θ=0, ψ=0 means: 1·[0·F + 1·(0·MP + 1·Tyler)] = Tyler
        'Tyler Only (θ=0, φ=1, ψ=0)': avg_performance[theta_0_idx, phi_1_idx, psi_0_idx],
        
        # Shrinkage Only: φ=1, θ=1 means: 1·[1·F + 0·RobustBlend] = F
        'Shrinkage Only (θ=1, φ=1)': avg_performance[theta_1_idx, phi_1_idx, psi_0_idx],
        
        # Optimal combination
        f'Optimal (θ={results["theta_opt"]:.2f}, φ={results["phi_opt"]:.2f}, ψ={results["psi_opt"]:.2f})': results['best_variance']
    }
    
    # Convert to DataFrame with annualized volatility
    comparison_df = pd.DataFrame([
        {
            'Method': name,
            'Variance': var,
            'Annualized Volatility': np.sqrt(var * 252),
        }
        for name, var in comparisons.items()
    ])
    
    # Calculate improvement vs SCM
    scm_vol = comparison_df.loc[0, 'Annualized Volatility']
    comparison_df['Improvement vs SCM'] = (
        (scm_vol - comparison_df['Annualized Volatility']) / scm_vol * 100
    )
    
    return comparison_df


def plot_performance_timeseries(results, returns, N_train=200, N_test=30, 
                                rebalance_freq=30, save_path='performance_timeseries.png'):
    """
    Plot time series of realized variance for optimal (θ, φ, ψ) vs baselines.
    
    Now includes Tyler's M-estimator in the comparisons.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from optimize_portfolio
    returns : pd.DataFrame
        Full returns data
    N_train, N_test, rebalance_freq : int
        Same parameters used in optimization
    save_path : str
        Path to save figure
    """
    performance_4d = results['performance_4d']  # Shape: (θ, φ, ψ, periods)
    theta_grid = results['theta_grid']
    phi_grid = results['phi_grid']
    psi_grid = results['psi_grid']
    
    # Get indices for optimal values
    theta_opt_idx = np.argmin(np.abs(theta_grid - results['theta_opt']))
    phi_opt_idx = np.argmin(np.abs(phi_grid - results['phi_opt']))
    psi_opt_idx = np.argmin(np.abs(psi_grid - results['psi_opt']))
    
    # Find indices for extreme values
    theta_0_idx = np.argmin(np.abs(theta_grid - 0))
    theta_1_idx = np.argmin(np.abs(theta_grid - 1))
    phi_0_idx = np.argmin(np.abs(phi_grid - 0))
    phi_1_idx = np.argmin(np.abs(phi_grid - 1))
    psi_0_idx = np.argmin(np.abs(psi_grid - 0))
    psi_1_idx = np.argmin(np.abs(psi_grid - 1))
    
    # Extract time series for each method
    T = len(returns)
    rebalance_times = range(0, T - N_train - N_test, rebalance_freq)
    periods = list(range(len(rebalance_times)))
    
    # performance_4d indexed as [theta, phi, psi, period]
    optimal_series = performance_4d[theta_opt_idx, phi_opt_idx, psi_opt_idx, :]
    scm_series = performance_4d[theta_0_idx, phi_0_idx, psi_0_idx, :]  # φ=0 → SCM
    mp_series = performance_4d[theta_0_idx, phi_1_idx, psi_1_idx, :]   # θ=0, φ=1, ψ=1 → MP
    tyler_series = performance_4d[theta_0_idx, phi_1_idx, psi_0_idx, :]  # θ=0, φ=1, ψ=0 → Tyler
    shrinkage_series = performance_4d[theta_1_idx, phi_1_idx, psi_0_idx, :]  # θ=1, φ=1 → F
    
    # Convert to annualized volatility
    optimal_vol = np.sqrt(optimal_series * 252)
    scm_vol = np.sqrt(scm_series * 252)
    mp_vol = np.sqrt(mp_series * 252)
    tyler_vol = np.sqrt(tyler_series * 252)
    shrinkage_vol = np.sqrt(shrinkage_series * 252)
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(periods, scm_vol, label='SCM Only', linewidth=2, alpha=0.7, color='gray')
    ax.plot(periods, mp_vol, label='MP Only', linewidth=2, alpha=0.7, color='blue')
    ax.plot(periods, tyler_vol, label='Tyler Only', linewidth=2, alpha=0.7, color='green')
    ax.plot(periods, shrinkage_vol, label='Shrinkage Only (F)', linewidth=2, alpha=0.7, color='orange')
    ax.plot(periods, optimal_vol, 
            label=f'Optimal (θ={results["theta_opt"]:.1f}, φ={results["phi_opt"]:.1f}, ψ={results["psi_opt"]:.1f})', 
            linewidth=2.5, color='red')
    
    ax.set_xlabel('Rebalancing Period', fontsize=12)
    ax.set_ylabel('Annualized Volatility', fontsize=12)
    ax.set_title('Out-of-Sample Volatility Over Time (with Tyler\'s M-Estimator)', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Performance time series saved to: {save_path}")
    plt.close()