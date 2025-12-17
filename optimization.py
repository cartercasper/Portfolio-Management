"""
Portfolio Optimization Module

Implements covariance matrix estimation and hyperparameter optimization for
Global Minimum Variance Portfolio (GMVP) optimization.
"""

import numpy as np
import pandas as pd
import warnings
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import scipy.linalg as la

from estimators import sample_cov, shrinkage_target, MP_est, tyler_m_estimator

warnings.filterwarnings('ignore')


def identity_estimator(n_assets):
    """Identity matrix estimator (I)."""
    return np.eye(n_assets)


def scaled_identity_estimator(SCM):
    """Scaled Identity estimator (sigma^2 * I)."""
    if isinstance(SCM, pd.DataFrame):
        SCM_arr = SCM.values
    else:
        SCM_arr = SCM
    
    n = SCM_arr.shape[0]
    avg_var = np.trace(SCM_arr) / n
    return avg_var * np.eye(n)


def solve_gmvp(sigma, weight_bounds=(0, 1)):
    """Solve Global Minimum Variance Portfolio analytically."""
    if isinstance(sigma, pd.DataFrame):
        sigma = sigma.values
    
    min_eig = np.linalg.eigvalsh(sigma).min()
    if min_eig < 1e-8:
        sigma = sigma + (abs(min_eig) + 1e-6) * np.eye(sigma.shape[0])
    else:
        sigma = sigma + 1e-8 * np.eye(sigma.shape[0])
    
    n = sigma.shape[0]
    ones = np.ones(n)
    
    try:
        L = la.cholesky(sigma, lower=True)
        y = la.solve_triangular(L, ones, lower=True)
        sigma_inv_ones = la.solve_triangular(L.T, y, lower=False)
        
        denom = ones @ sigma_inv_ones
        p = sigma_inv_ones / denom
        
        if np.all(p >= weight_bounds[0] - 1e-6) and np.all(p <= weight_bounds[1] + 1e-6):
            p = np.clip(p, weight_bounds[0], weight_bounds[1])
            p = p / p.sum()
            return p
        else:
            return _solve_gmvp_constrained(sigma, weight_bounds)
    except:
        return _solve_gmvp_constrained(sigma, weight_bounds)


def _solve_gmvp_constrained(sigma, weight_bounds=(0, 1)):
    """Constrained GMVP using cvxpy."""
    import cvxpy as cp
    
    n = sigma.shape[0]
    p = cp.Variable(n)
    
    objective = cp.Minimize(cp.quad_form(p, sigma))
    constraints = [
        cp.sum(p) == 1,
        p >= weight_bounds[0],
        p <= weight_bounds[1]
    ]
    
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.OSQP, verbose=False)
        if problem.status in ['optimal', 'optimal_inaccurate']:
            return p.value
    except:
        pass
    
    return np.ones(n) / n


def paper_dual_method_2d(theta, phi, SCM, F, MP):
    """Paper's dual method: Sigma*(theta,phi) = phi[theta*F + (1-theta)*MP] + (1-phi)*SCM"""
    inner_blend = theta * F + (1 - theta) * MP
    sigma_star = phi * inner_blend + (1 - phi) * SCM
    return sigma_star


def our_3d_method(theta, phi, psi, SCM, F, MP, Tyler):
    """Our 3D extension with Tyler's M-estimator."""
    robust_blend = psi * MP + (1 - psi) * Tyler
    inner_blend = theta * F + (1 - theta) * robust_blend
    sigma_star = phi * inner_blend + (1 - phi) * SCM
    return sigma_star


def _evaluate_estimator_single_period(args):
    """Evaluate a single estimator on one period."""
    t0, R_train_values, R_test_values, columns, estimator_name, n_assets = args
    
    R_train_df = pd.DataFrame(R_train_values, columns=columns)
    
    if estimator_name == 'Identity':
        sigma = identity_estimator(n_assets)
    elif estimator_name == 'Scaled Identity':
        SCM = sample_cov(R_train_df)
        sigma = scaled_identity_estimator(SCM)
    elif estimator_name == 'SCM':
        sigma = sample_cov(R_train_df).values
    elif estimator_name == 'Shrinkage Target':
        SCM = sample_cov(R_train_df)
        sigma = shrinkage_target(SCM).values
    elif estimator_name == 'Marchenko-Pastur':
        SCM = sample_cov(R_train_df)
        sigma = MP_est(R_train_df, SCM)
    elif estimator_name == 'Tyler':
        SCM = sample_cov(R_train_df)
        sigma = tyler_m_estimator(R_train_df, SCM)
    else:
        raise ValueError(f"Unknown estimator: {estimator_name}")
    
    p = solve_gmvp(sigma)
    portfolio_returns = R_test_values @ p
    return np.var(portfolio_returns)


def _grid_search_2d_period(args):
    """Grid search over (theta, phi) for paper's 2D dual method on one period."""
    t0, R_train_values, R_test_values, columns, theta_grid, phi_grid = args
    
    R_train_df = pd.DataFrame(R_train_values, columns=columns)
    
    SCM = sample_cov(R_train_df).values
    F = shrinkage_target(sample_cov(R_train_df)).values
    MP = MP_est(R_train_df, sample_cov(R_train_df))
    
    results = np.zeros((len(theta_grid), len(phi_grid)))
    
    for i, theta in enumerate(theta_grid):
        for j, phi in enumerate(phi_grid):
            sigma_star = paper_dual_method_2d(theta, phi, SCM, F, MP)
            p = solve_gmvp(sigma_star)
            portfolio_returns = R_test_values @ p
            results[i, j] = np.var(portfolio_returns)
    
    return results


def _grid_search_3d_period(args):
    """Grid search over (theta, phi, psi) for our 3D extension on one period."""
    t0, R_train_values, R_test_values, columns, theta_grid, phi_grid, psi_grid = args
    
    R_train_df = pd.DataFrame(R_train_values, columns=columns)
    
    SCM = sample_cov(R_train_df).values
    F = shrinkage_target(sample_cov(R_train_df)).values
    MP = MP_est(R_train_df, sample_cov(R_train_df))
    Tyler = tyler_m_estimator(R_train_df, sample_cov(R_train_df))
    
    results = np.zeros((len(theta_grid), len(phi_grid), len(psi_grid)))
    
    for i, theta in enumerate(theta_grid):
        for j, phi in enumerate(phi_grid):
            for k, psi in enumerate(psi_grid):
                sigma_star = our_3d_method(theta, phi, psi, SCM, F, MP, Tyler)
                p = solve_gmvp(sigma_star)
                portfolio_returns = R_test_values @ p
                results[i, j, k] = np.var(portfolio_returns)
    
    return results


def run_optimization(returns, N_train=200, N_test=30, rebalance_freq=30,
                     grid_size=11, n_jobs=None, verbose=True):
    """Run full comparison of all estimators."""
    if n_jobs is None:
        n_jobs = min(cpu_count(), 14)
    
    T, n_assets = returns.shape
    columns = returns.columns.tolist()
    returns_values = returns.values
    
    rebalance_times = list(range(0, T - N_train - N_test, rebalance_freq))
    n_periods = len(rebalance_times)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"PORTFOLIO OPTIMIZATION")
        print(f"{'='*70}")
        print(f"Assets: {n_assets}")
        print(f"Training window: {N_train} days")
        print(f"Test window: {N_test} days")
        print(f"Rebalancing periods: {n_periods}")
        print(f"Grid size: {grid_size}")
        print(f"Parallel workers: {n_jobs}")
        print(f"{'='*70}\n")
    
    results = {
        'n_assets': n_assets,
        'N_train': N_train,
        'N_test': N_test,
        'n_periods': n_periods,
        'grid_size': grid_size,
    }
    
    # Evaluate fixed estimators
    fixed_estimators = [
        ('Identity (I)', 'Identity'),
        ('Scaled Identity (sigma^2*I)', 'Scaled Identity'),
        ('SCM', 'SCM'),
        ('Shrinkage Target (F)', 'Shrinkage Target'),
        ('Marchenko-Pastur (MP)', 'Marchenko-Pastur'),
        ("Tyler's M-Estimator", 'Tyler'),
    ]
    
    fixed_results = {}
    
    for display_name, estimator_name in fixed_estimators:
        if verbose:
            print(f"Evaluating: {display_name}...", end=' ')
        
        args_list = []
        for t0 in rebalance_times:
            R_train = returns_values[t0:t0+N_train]
            R_test = returns_values[t0+N_train:t0+N_train+N_test]
            args_list.append((t0, R_train, R_test, columns, estimator_name, n_assets))
        
        with Pool(n_jobs) as pool:
            variances = pool.map(_evaluate_estimator_single_period, args_list)
        
        avg_var = np.mean(variances)
        fixed_results[display_name] = {
            'avg_variance': avg_var,
            'period_variances': variances,
        }
        
        if verbose:
            print(f"Ann. Vol = {np.sqrt(avg_var * 252):.4f}")
    
    results['fixed_estimators'] = fixed_results
    
    # Paper's 2D Dual Method
    if verbose:
        print(f"\nOptimizing Paper's 2D Dual Method (theta, phi)...")
    
    theta_grid = np.linspace(0, 1, grid_size)
    phi_grid = np.linspace(0, 1, grid_size)
    
    args_list_2d = []
    for t0 in rebalance_times:
        R_train = returns_values[t0:t0+N_train]
        R_test = returns_values[t0+N_train:t0+N_train+N_test]
        args_list_2d.append((t0, R_train, R_test, columns, theta_grid, phi_grid))
    
    with Pool(n_jobs) as pool:
        period_results_2d = list(tqdm(
            pool.imap(_grid_search_2d_period, args_list_2d),
            total=n_periods,
            desc="2D Grid Search",
            disable=not verbose
        ))
    
    avg_performance_2d = np.mean(period_results_2d, axis=0)
    opt_idx_2d = np.unravel_index(np.argmin(avg_performance_2d), avg_performance_2d.shape)
    theta_opt_2d = theta_grid[opt_idx_2d[0]]
    phi_opt_2d = phi_grid[opt_idx_2d[1]]
    best_var_2d = avg_performance_2d[opt_idx_2d]
    
    results['dual_method_2d'] = {
        'theta_opt': theta_opt_2d,
        'phi_opt': phi_opt_2d,
        'avg_variance': best_var_2d,
        'avg_performance': avg_performance_2d,
        'theta_grid': theta_grid,
        'phi_grid': phi_grid,
    }
    
    if verbose:
        print(f"  Optimal theta={theta_opt_2d:.2f}, phi={phi_opt_2d:.2f}")
        print(f"  Ann. Vol = {np.sqrt(best_var_2d * 252):.4f}")
    
    # Our 3D Extension
    if verbose:
        print(f"\nOptimizing Our 3D Extension (theta, phi, psi)...")
    
    psi_grid = np.linspace(0, 1, grid_size)
    
    args_list_3d = []
    for t0 in rebalance_times:
        R_train = returns_values[t0:t0+N_train]
        R_test = returns_values[t0+N_train:t0+N_train+N_test]
        args_list_3d.append((t0, R_train, R_test, columns, theta_grid, phi_grid, psi_grid))
    
    with Pool(n_jobs) as pool:
        period_results_3d = list(tqdm(
            pool.imap(_grid_search_3d_period, args_list_3d),
            total=n_periods,
            desc="3D Grid Search",
            disable=not verbose
        ))
    
    avg_performance_3d = np.mean(period_results_3d, axis=0)
    opt_idx_3d = np.unravel_index(np.argmin(avg_performance_3d), avg_performance_3d.shape)
    theta_opt_3d = theta_grid[opt_idx_3d[0]]
    phi_opt_3d = phi_grid[opt_idx_3d[1]]
    psi_opt_3d = psi_grid[opt_idx_3d[2]]
    best_var_3d = avg_performance_3d[opt_idx_3d]
    
    results['our_3d_method'] = {
        'theta_opt': theta_opt_3d,
        'phi_opt': phi_opt_3d,
        'psi_opt': psi_opt_3d,
        'avg_variance': best_var_3d,
        'avg_performance': avg_performance_3d,
        'theta_grid': theta_grid,
        'phi_grid': phi_grid,
        'psi_grid': psi_grid,
    }
    
    if verbose:
        print(f"  Optimal theta={theta_opt_3d:.2f}, phi={phi_opt_3d:.2f}, psi={psi_opt_3d:.2f}")
        print(f"  Ann. Vol = {np.sqrt(best_var_3d * 252):.4f}")
    
    return results
