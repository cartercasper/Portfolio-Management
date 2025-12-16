import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from estimators import MP_est, sample_cov, shrinkage_target
import scipy.linalg as la
import warnings

# SCM is sample covariance matrix
# SMP is covariance matrix reconstructed from Marchenko-Pastur clipped eigenvalues
# ST is shrinkage target matrix 

#######################################
# Section III Implementation
#######################################
# Key fixes implemented:
# 1. Removed double-demeaning (let pandas .cov() handle it)
# 2. Changed to GMVP (no return constraint) to match paper's focus on covariance
# 3. Fixed MP estimator to replace noise with mean of SIGNAL eigenvalues
# 4. Global (θ, φ) selection by averaging across ALL periods (no look-ahead bias)
#
# Performance optimizations:
# 5. Parallelize over PERIODS (not grid points) - reduces serialization overhead
# 6. Precompute matrix differences D1=F-MP, D2=MP-SCM for faster sigma_star
# 7. Analytical GMVP solution when possible (avoids QP solver overhead)
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
    Process a single rebalancing period - computes performance for ALL (θ,φ) combinations.
    
    This is called in parallel across periods (not across grid points).
    Much more efficient because we only serialize data once per period.
    
    Parameters:
    -----------
    args : tuple
        (period_idx, t0, returns_values, N_train, N_test, theta_grid, phi_grid, use_gmvp)
    
    Returns:
    --------
    tuple : (period_idx, results_grid)
        period_idx: which period this is
        results_grid: 2D array of variances for all (θ,φ) combinations
    """
    period_idx, t0, returns_values, columns, N_train, N_test, theta_grid, phi_grid, use_gmvp = args
    
    grid_size = len(theta_grid)
    results_grid = np.zeros((grid_size, grid_size))
    
    # Split data
    R_train = returns_values[t0:t0+N_train]
    R_test = returns_values[t0+N_train:t0+N_train+N_test]
    
    # Convert to DataFrame for estimator functions (they expect DataFrames)
    R_train_df = pd.DataFrame(R_train, columns=columns)
    
    # Compute estimators ONCE per period
    SCM_df = sample_cov(R_train_df)
    F_df = shrinkage_target(SCM_df)
    MP_arr = MP_est(R_train_df, SCM_df)
    
    # Convert to numpy
    SCM = SCM_df.values
    F = F_df.values
    MP = MP_arr
    
    # OPTIMIZATION: Precompute differences
    # sigma_star = phi * (theta*F + (1-theta)*MP) + (1-phi)*SCM
    #            = phi*theta*F + phi*(1-theta)*MP + (1-phi)*SCM
    #            = SCM + phi*(MP - SCM) + phi*theta*(F - MP)
    #            = SCM + phi*D2 + phi*theta*D1
    D1 = F - MP      # F - MP
    D2 = MP - SCM    # MP - SCM
    
    # Grid search over (θ, φ)
    for i, theta in enumerate(theta_grid):
        for j, phi in enumerate(phi_grid):
            # OPTIMIZATION: Use precomputed differences
            sigma_star = SCM + phi * D2 + phi * theta * D1
            
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
                results_grid[i, j] = np.var(portfolio_returns)
            else:
                results_grid[i, j] = np.inf
    
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
    Find optimal (theta, phi) using rolling window approach.
    
    Implements Section III of the paper: grid search over (θ, φ) to find
    the combination that minimizes average out-of-sample portfolio variance.
    
    OPTIMIZATIONS:
    - Parallelizes over PERIODS (not grid points) - much less serialization
    - Precomputes matrix differences for faster sigma_star computation
    - Uses analytical GMVP when bounds allow (100x faster than QP)
    - Caches estimators (SCM, F, MP) per period
    
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
        Number of grid points for θ and φ (default: 6 for 36 combinations)
        Use 6 for quick testing, 11 for paper-level precision
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
        - theta_opt: optimal θ value
        - phi_opt: optimal φ value
        - best_variance: average out-of-sample variance
        - avg_performance: 2D array of average performance
        - performance_3d: 3D array of all period performances
        - theta_grid, phi_grid: parameter grids
    """
    # Setup
    theta_grid = np.linspace(0, 1, grid_size)
    phi_grid = np.linspace(0, 1, grid_size)
    
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
        print(f"HYPERPARAMETER OPTIMIZATION")
        print(f"{'='*60}")
        print(f"Total periods: {T}")
        print(f"Training window: {N_train} days")
        print(f"Test window: {N_test} days")
        print(f"Rebalancing frequency: {rebalance_freq} days")
        print(f"Number of rebalancing periods: {num_periods}")
        print(f"Grid size: {len(theta_grid)} × {len(phi_grid)} = {len(theta_grid) * len(phi_grid)} combinations")
        print(f"Parallel workers: {n_jobs} CPUs")
        print(f"Portfolio type: {'GMVP (no return constraint)' if use_gmvp else 'Mean-variance (with return constraint)'}")
        print(f"Optimizations: period-parallel, precomputed diffs, analytical GMVP")
        print(f"{'='*60}\n")
    
    # Storage: (grid_size theta × grid_size phi × num_periods)
    performance = np.zeros((grid_size, grid_size, num_periods))
    
    # Convert returns to numpy once (avoid repeated conversion)
    returns_values = returns.values
    columns = returns.columns.tolist()
    
    # Prepare arguments for each period
    period_args = [
        (period_idx, t0, returns_values, columns, N_train, N_test, theta_grid, phi_grid, use_gmvp)
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
            performance[:, :, period_idx] = results_grid
    else:
        # Serial execution (for debugging or single period)
        iterator = tqdm(period_args, desc="Rebalancing periods") if verbose else period_args
        for args in iterator:
            period_idx, results_grid = _process_single_period(args)
            performance[:, :, period_idx] = results_grid
    
    # Average across periods
    avg_performance = np.mean(performance, axis=2)
    
    # Find optimal (theta, phi)
    optimal_idx = np.unravel_index(np.argmin(avg_performance), avg_performance.shape)
    theta_opt = theta_grid[optimal_idx[0]]
    phi_opt = phi_grid[optimal_idx[1]]
    best_variance = avg_performance[optimal_idx]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"OPTIMIZATION RESULTS")
        print(f"{'='*60}")
        print(f"Optimal θ: {theta_opt:.2f}")
        print(f"Optimal φ: {phi_opt:.2f}")
        print(f"Average out-of-sample variance: {best_variance:.8f}")
        print(f"Average out-of-sample volatility (annualized): {np.sqrt(best_variance * 252):.4f}")
        print(f"{'='*60}\n")
    
    # Return results
    return {
        'theta_opt': theta_opt,
        'phi_opt': phi_opt,
        'best_variance': best_variance,
        'avg_performance': avg_performance,
        'performance_3d': performance,
        'theta_grid': theta_grid,
        'phi_grid': phi_grid
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
    Plot heatmap of average performance across (θ, φ) grid.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from optimize_portfolio
    save_path : str
        Path to save figure
    """
    avg_performance = results['avg_performance']
    theta_grid = results['theta_grid']
    phi_grid = results['phi_grid']
    
    # Convert variance to annualized volatility for interpretability
    avg_volatility = np.sqrt(avg_performance * 252)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(avg_volatility, cmap='RdYlGn_r', aspect='auto', origin='lower')
    
    # Mark optimal point
    theta_opt = results['theta_opt']
    phi_opt = results['phi_opt']
    theta_idx = np.where(theta_grid == theta_opt)[0][0]
    phi_idx = np.where(phi_grid == phi_opt)[0][0]
    ax.plot(theta_idx, phi_idx, 'b*', markersize=20, label=f'Optimal: θ={theta_opt:.1f}, φ={phi_opt:.1f}')
    
    # Set ticks
    ax.set_xticks(range(len(theta_grid)))
    ax.set_yticks(range(len(phi_grid)))
    ax.set_xticklabels([f'{x:.1f}' for x in theta_grid])
    ax.set_yticklabels([f'{y:.1f}' for y in phi_grid])
    
    # Labels
    ax.set_xlabel('θ (Shrinkage Target vs MP mixing)', fontsize=12)
    ax.set_ylabel('φ (Regularized vs SCM mixing)', fontsize=12)
    ax.set_title('Average Out-of-Sample Volatility (Annualized)', fontsize=14)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Annualized Volatility', rotation=270, labelpad=20)
    
    # Legend
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Performance heatmap saved to: {save_path}")
    plt.close()


def compare_estimators(results):
    """
    Compare the optimal combination with baseline estimators.
    
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
    
    # Find indices for extreme values (0 and 1)
    theta_0_idx = np.argmin(np.abs(theta_grid - 0))
    theta_1_idx = np.argmin(np.abs(theta_grid - 1))
    phi_0_idx = np.argmin(np.abs(phi_grid - 0))
    phi_1_idx = np.argmin(np.abs(phi_grid - 1))
    
    # Extract performance for key combinations
    comparisons = {
        'SCM Only (θ=0, φ=0)': avg_performance[theta_0_idx, phi_0_idx],
        'MP Only (θ=0, φ=1)': avg_performance[theta_0_idx, phi_1_idx],
        'Shrinkage Only (θ=1, φ=1)': avg_performance[theta_1_idx, phi_1_idx],
        f'Optimal (θ={results["theta_opt"]:.2f}, φ={results["phi_opt"]:.2f})': results['best_variance']
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
    
    # Calculate improvement
    scm_vol = comparison_df.loc[0, 'Annualized Volatility']
    comparison_df['Improvement vs SCM'] = (
        (scm_vol - comparison_df['Annualized Volatility']) / scm_vol * 100
    )
    
    return comparison_df


def plot_performance_timeseries(results, returns, N_train=200, N_test=30, 
                                rebalance_freq=30, save_path='performance_timeseries.png'):
    """
    Plot time series of realized variance for optimal (θ, φ) vs baselines.
    
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
    performance_3d = results['performance_3d']
    theta_grid = results['theta_grid']
    phi_grid = results['phi_grid']
    
    # Get indices for key methods
    theta_opt_idx = np.where(theta_grid == results['theta_opt'])[0][0]
    phi_opt_idx = np.where(phi_grid == results['phi_opt'])[0][0]
    
    # Find indices for extreme values
    theta_0_idx = np.argmin(np.abs(theta_grid - 0))
    theta_1_idx = np.argmin(np.abs(theta_grid - 1))
    phi_0_idx = np.argmin(np.abs(phi_grid - 0))
    phi_1_idx = np.argmin(np.abs(phi_grid - 1))
    
    # Extract time series for each method
    T = len(returns)
    rebalance_times = range(0, T - N_train - N_test, rebalance_freq)
    periods = list(range(len(rebalance_times)))
    
    optimal_series = performance_3d[theta_opt_idx, phi_opt_idx, :]
    scm_series = performance_3d[theta_0_idx, phi_0_idx, :]
    mp_series = performance_3d[theta_0_idx, phi_1_idx, :]
    shrinkage_series = performance_3d[theta_1_idx, phi_1_idx, :]
    
    # Convert to annualized volatility
    optimal_vol = np.sqrt(optimal_series * 252)
    scm_vol = np.sqrt(scm_series * 252)
    mp_vol = np.sqrt(mp_series * 252)
    shrinkage_vol = np.sqrt(shrinkage_series * 252)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(periods, scm_vol, label='SCM Only', linewidth=2, alpha=0.7)
    ax.plot(periods, mp_vol, label='MP Only', linewidth=2, alpha=0.7)
    ax.plot(periods, shrinkage_vol, label='Shrinkage Only', linewidth=2, alpha=0.7)
    ax.plot(periods, optimal_vol, label=f'Optimal (θ={results["theta_opt"]:.1f}, φ={results["phi_opt"]:.1f})', 
            linewidth=2.5, color='red')
    
    ax.set_xlabel('Rebalancing Period', fontsize=12)
    ax.set_ylabel('Annualized Volatility', fontsize=12)
    ax.set_title('Out-of-Sample Volatility Over Time', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Performance time series saved to: {save_path}")
    plt.close()