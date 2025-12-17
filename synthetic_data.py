"""
Synthetic Data Generation Module

Generates synthetic return data for testing covariance matrix estimators.
Supports multiple distributions to test estimator robustness.

Distributions:
    - Gaussian N(0,1): Baseline case where SCM is optimal
    - Student-t (ν=3.5): Heavy tails to test Tyler's robustness
    - Pareto: Extreme heavy tails stress test
    - Factor Model: Realistic correlation structure with known true Σ
"""

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================================
# Configuration
# ============================================================================

SYNTHETIC_CONFIG = {
    'N': 100,              # Number of assets (matches real data)
    'T': 1200,             # Number of observations (~5 years of daily data)
    'K': 5,                # Number of factors for factor model
    'nu': 3.5,             # Degrees of freedom for Student-t
    'alpha': 3.0,          # Shape parameter for Pareto
    'daily_vol': 0.02,     # Target daily volatility (~32% annualized)
    'correlation': 0.3,    # Average pairwise correlation
    'annual_return': 0.12, # Target annualized return (~12% to ensure 10% is feasible)
    'daily_return': 0.12 / 252,  # Daily expected return
}


# ============================================================================
# Correlation Structure Generation
# ============================================================================

def generate_correlation_matrix(n_assets, avg_corr=0.3, seed=None):
    """
    Generate a valid correlation matrix with specified average correlation.
    
    Uses a single-factor structure: ρ_ij = β_i * β_j for i ≠ j
    This ensures the matrix is positive definite.
    
    Parameters:
        n_assets: Number of assets
        avg_corr: Target average pairwise correlation
        seed: Random seed
    
    Returns:
        corr_matrix: n_assets x n_assets correlation matrix
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Single factor loadings chosen to achieve target correlation
    # For ρ = β², we need β = sqrt(ρ)
    beta = np.sqrt(avg_corr) * np.ones(n_assets)
    
    # Add some variation
    beta = beta * (1 + 0.2 * np.random.randn(n_assets))
    beta = np.clip(beta, 0.1, 0.9)
    
    # Correlation matrix: ρ_ij = β_i * β_j, ρ_ii = 1
    corr_matrix = np.outer(beta, beta)
    np.fill_diagonal(corr_matrix, 1.0)
    
    return corr_matrix


def correlation_to_covariance(corr_matrix, volatilities):
    """Convert correlation matrix to covariance matrix."""
    D = np.diag(volatilities)
    return D @ corr_matrix @ D


# ============================================================================
# Gaussian Returns
# ============================================================================

def generate_gaussian_returns(N=100, T=1200, daily_vol=0.02, avg_corr=0.3, 
                               daily_return=None, seed=42):
    """
    Generate Gaussian returns with realistic correlation structure and positive drift.
    
    r_t ~ N(μ, Σ)
    
    Parameters:
        N: Number of assets
        T: Number of time periods
        daily_vol: Target daily volatility
        avg_corr: Average pairwise correlation
        daily_return: Expected daily return (default: ~12% annualized)
        seed: Random seed
    
    Returns:
        returns: DataFrame of shape (T, N)
        true_cov: True covariance matrix (N, N)
    """
    np.random.seed(seed)
    
    if daily_return is None:
        daily_return = SYNTHETIC_CONFIG['daily_return']
    
    # Generate correlation and covariance
    corr = generate_correlation_matrix(N, avg_corr, seed)
    vols = daily_vol * (1 + 0.3 * np.random.randn(N))  # Some variation in volatilities
    vols = np.clip(vols, 0.01, 0.05)
    true_cov = correlation_to_covariance(corr, vols)
    
    # Generate expected returns with some variation across assets
    # Average ~12% annualized, range roughly 8-16%
    mu = daily_return * (1 + 0.3 * np.random.randn(N))
    mu = np.clip(mu, daily_return * 0.5, daily_return * 1.5)
    
    # Generate returns via Cholesky decomposition
    L = np.linalg.cholesky(true_cov)
    Z = np.random.randn(T, N)
    returns = Z @ L.T + mu  # Add drift
    
    # Create DataFrame
    columns = [f'Asset_{i}' for i in range(N)]
    returns_df = pd.DataFrame(returns, columns=columns)
    
    return returns_df, true_cov


# ============================================================================
# Student-t Returns (Heavy Tails)
# ============================================================================

def generate_student_t_returns(N=100, T=1200, nu=3.5, daily_vol=0.02, avg_corr=0.3, 
                                daily_return=None, seed=42):
    """
    Generate Student-t returns with heavy tails and positive drift.
    
    r_t ~ t_ν(μ, Σ) - multivariate Student-t with drift
    
    This tests Tyler's M-estimator which is designed for heavy-tailed distributions.
    ν = 3.5 gives heavy tails but finite variance (requires ν > 2).
    
    Parameters:
        N: Number of assets
        nu: Degrees of freedom (lower = heavier tails)
        T: Number of time periods
        daily_vol: Target daily volatility
        avg_corr: Average pairwise correlation
        daily_return: Expected daily return (default: ~12% annualized)
        seed: Random seed
    
    Returns:
        returns: DataFrame of shape (T, N)
        true_cov: True covariance matrix (N, N)
    """
    np.random.seed(seed)
    
    if daily_return is None:
        daily_return = SYNTHETIC_CONFIG['daily_return']
    
    # Generate correlation and covariance
    corr = generate_correlation_matrix(N, avg_corr, seed)
    vols = daily_vol * (1 + 0.3 * np.random.randn(N))
    vols = np.clip(vols, 0.01, 0.05)
    true_cov = correlation_to_covariance(corr, vols)
    
    # Generate expected returns with some variation across assets
    mu = daily_return * (1 + 0.3 * np.random.randn(N))
    mu = np.clip(mu, daily_return * 0.5, daily_return * 1.5)
    
    # Generate multivariate Student-t via:
    # X = Z / sqrt(W/nu) where Z ~ N(0, Σ) and W ~ χ²(nu)
    L = np.linalg.cholesky(true_cov)
    Z = np.random.randn(T, N)
    gaussian = Z @ L.T
    
    # Chi-squared scaling for Student-t
    W = np.random.chisquare(nu, T)
    scaling = np.sqrt(nu / W)
    returns = gaussian * scaling[:, np.newaxis]
    
    # Scale to match target volatility (Student-t has higher variance)
    # Var(t_nu) = nu/(nu-2) * Var(Normal) for nu > 2
    scale_factor = np.sqrt((nu - 2) / nu) if nu > 2 else 1.0
    returns = returns * scale_factor
    
    # Add drift
    returns = returns + mu
    
    # Create DataFrame
    columns = [f'Asset_{i}' for i in range(N)]
    returns_df = pd.DataFrame(returns, columns=columns)
    
    return returns_df, true_cov


# ============================================================================
# Pareto (Extreme Heavy Tails)
# ============================================================================

def generate_pareto_returns(N=100, T=1200, alpha=3.0, daily_vol=0.02, avg_corr=0.3, 
                            daily_return=None, seed=42):
    """
    Generate returns with Pareto-distributed residuals (extreme heavy tails) and positive drift.
    
    Uses factor model with Pareto innovations instead of Gaussian.
    This is a stress test - most estimators will struggle.
    
    Parameters:
        N: Number of assets
        T: Number of time periods
        alpha: Pareto shape parameter (lower = heavier tails, need α > 2 for finite variance)
        daily_vol: Target daily volatility
        avg_corr: Average pairwise correlation
        daily_return: Expected daily return (default: ~12% annualized)
        seed: Random seed
    
    Returns:
        returns: DataFrame of shape (T, N)
        true_cov: Approximate covariance matrix (N, N)
    """
    np.random.seed(seed)
    
    if daily_return is None:
        daily_return = SYNTHETIC_CONFIG['daily_return']
    
    # Generate correlation structure
    corr = generate_correlation_matrix(N, avg_corr, seed)
    vols = daily_vol * (1 + 0.3 * np.random.randn(N))
    vols = np.clip(vols, 0.01, 0.05)
    
    # Generate expected returns with some variation across assets
    mu = daily_return * (1 + 0.3 * np.random.randn(N))
    mu = np.clip(mu, daily_return * 0.5, daily_return * 1.5)
    
    # Generate Pareto samples using scipy for consistent parameterization
    # scipy.stats.pareto with loc=0, scale=1 gives standard Pareto on [1, inf)
    pareto_rv = stats.pareto(alpha)
    pareto_samples = pareto_rv.rvs(size=(T, N))
    
    # Standardize using theoretical moments for Pareto on [1, inf)
    # Mean: alpha / (alpha - 1) for alpha > 1
    # Var: alpha / ((alpha-1)^2 * (alpha-2)) for alpha > 2
    if alpha > 1:
        pareto_mean = alpha / (alpha - 1)
    else:
        pareto_mean = pareto_samples.mean()
    
    if alpha > 2:
        pareto_std = np.sqrt(alpha / ((alpha - 1)**2 * (alpha - 2)))
    else:
        pareto_std = pareto_samples.std()
    
    L = np.linalg.cholesky(corr)
    
    # Standardize to mean 0, variance 1
    standardized = (pareto_samples - pareto_mean) / pareto_std
    
    # Apply correlation structure
    correlated = standardized @ L.T
    
    # Scale to target volatilities and add drift
    returns = correlated * vols + mu
    
    # Approximate true covariance (correlation structure is correct, but tails differ)
    true_cov = correlation_to_covariance(corr, vols)
    
    # Create DataFrame
    columns = [f'Asset_{i}' for i in range(N)]
    returns_df = pd.DataFrame(returns, columns=columns)
    
    return returns_df, true_cov


# ============================================================================
# Factor Model (Realistic Structure)
# ============================================================================

def generate_factor_model_returns(N=100, T=1200, K=5, daily_vol=0.02, daily_return=None, seed=42):
    """
    Generate returns from a factor model with known true covariance and positive drift.
    
    r_t = μ + B @ f_t + ε_t
    
    where:
        μ is the expected return vector
        f_t ~ N(0, I_K) - K factors
        ε_t ~ N(0, D) - idiosyncratic noise
        B is N × K factor loading matrix
    
    True covariance: Σ = B @ B^T + D
    
    This is the most realistic synthetic data and allows exact PRIAL calculation.
    
    Parameters:
        N: Number of assets
        T: Number of time periods
        K: Number of factors
        daily_vol: Target daily volatility
        daily_return: Expected daily return (default: ~12% annualized)
        seed: Random seed
    
    Returns:
        returns: DataFrame of shape (T, N)
        true_cov: True covariance matrix (N, N) = B @ B^T + D
        factor_loadings: B matrix (N, K)
        idio_var: Diagonal of D
    """
    np.random.seed(seed)
    
    if daily_return is None:
        daily_return = SYNTHETIC_CONFIG['daily_return']
    
    # Generate expected returns with some variation across assets
    mu = daily_return * (1 + 0.3 * np.random.randn(N))
    mu = np.clip(mu, daily_return * 0.5, daily_return * 1.5)
    
    # Generate factor loadings
    # Scale so that factor variance explains ~50% of total variance
    factor_loadings = np.random.randn(N, K) * (daily_vol * 0.5 / np.sqrt(K))
    
    # Idiosyncratic variances (remaining ~50% of variance)
    idio_var = (daily_vol * 0.5)**2 * (1 + 0.5 * np.random.rand(N))
    
    # Generate factors and idiosyncratic returns
    factors = np.random.randn(T, K)
    idiosyncratic = np.random.randn(T, N) * np.sqrt(idio_var)
    
    # Returns = drift + factor returns + idiosyncratic
    returns = mu + factors @ factor_loadings.T + idiosyncratic
    
    # True covariance matrix
    true_cov = factor_loadings @ factor_loadings.T + np.diag(idio_var)
    
    # Create DataFrame
    columns = [f'Asset_{i}' for i in range(N)]
    returns_df = pd.DataFrame(returns, columns=columns)
    
    return returns_df, true_cov, factor_loadings, idio_var


# ============================================================================
# Unified Interface
# ============================================================================

SYNTHETIC_TYPES = {
    'gaussian': {
        'name': 'Gaussian N(0,Σ)',
        'description': 'Baseline Gaussian - SCM should be near-optimal',
        'generator': generate_gaussian_returns,
    },
    'student_t': {
        'name': 'Student-t (ν=3.5)',
        'description': 'Heavy tails - Tyler should outperform',
        'generator': generate_student_t_returns,
    },
    'pareto': {
        'name': 'Pareto (α=3.0)',
        'description': 'Extreme heavy tails - stress test',
        'generator': generate_pareto_returns,
    },
    'factor_model': {
        'name': 'Factor Model (K=5)',
        'description': 'Realistic structure - MP should excel at eigenvalue cleaning',
        'generator': generate_factor_model_returns,
    },
}


def generate_synthetic_returns(distribution='gaussian', N=100, T=1200, seed=42, **kwargs):
    """
    Generate synthetic returns for a specified distribution.
    
    Parameters:
        distribution: One of 'gaussian', 'student_t', 'pareto', 'factor_model'
        N: Number of assets
        T: Number of time periods
        seed: Random seed
        **kwargs: Additional parameters for specific distributions
    
    Returns:
        returns: DataFrame of shape (T, N)
        true_cov: True covariance matrix (or best approximation)
        info: Dict with distribution-specific information
    """
    if distribution not in SYNTHETIC_TYPES:
        raise ValueError(f"Unknown distribution: {distribution}. "
                        f"Available: {list(SYNTHETIC_TYPES.keys())}")
    
    config = SYNTHETIC_CONFIG.copy()
    config.update(kwargs)
    
    info = {
        'distribution': distribution,
        'name': SYNTHETIC_TYPES[distribution]['name'],
        'description': SYNTHETIC_TYPES[distribution]['description'],
        'N': N,
        'T': T,
        'seed': seed,
    }
    
    if distribution == 'gaussian':
        returns, true_cov = generate_gaussian_returns(
            N=N, T=T, 
            daily_vol=config['daily_vol'],
            avg_corr=config['correlation'],
            seed=seed
        )
        
    elif distribution == 'student_t':
        returns, true_cov = generate_student_t_returns(
            N=N, T=T,
            nu=config['nu'],
            daily_vol=config['daily_vol'],
            avg_corr=config['correlation'],
            seed=seed
        )
        info['nu'] = config['nu']
        
    elif distribution == 'pareto':
        returns, true_cov = generate_pareto_returns(
            N=N, T=T,
            alpha=config['alpha'],
            daily_vol=config['daily_vol'],
            avg_corr=config['correlation'],
            seed=seed
        )
        info['alpha'] = config['alpha']
        
    elif distribution == 'factor_model':
        returns, true_cov, factor_loadings, idio_var = generate_factor_model_returns(
            N=N, T=T,
            K=config['K'],
            daily_vol=config['daily_vol'],
            seed=seed
        )
        info['K'] = config['K']
        info['factor_loadings'] = factor_loadings
        info['idio_var'] = idio_var
    
    return returns, true_cov, info


def get_all_synthetic_datasets(N=100, T=1200, seed=42):
    """
    Generate all synthetic datasets.
    
    Parameters:
        N: Number of assets
        T: Number of observations
        seed: Random seed
    
    Returns:
        datasets: Dict of {name: (returns_df, true_cov, info)}
    """
    datasets = {}
    
    for dist_type in SYNTHETIC_TYPES.keys():
        returns, true_cov, info = generate_synthetic_returns(
            distribution=dist_type,
            N=N,
            T=T,
            seed=seed
        )
        datasets[info['name']] = (returns, true_cov, info)
    
    return datasets


# ============================================================================
# Validation and Diagnostics
# ============================================================================

def validate_synthetic_data(returns, true_cov, info):
    """
    Validate synthetic data properties.
    
    Parameters:
        returns: DataFrame of returns
        true_cov: True covariance matrix
        info: Info dict from generator
    
    Returns:
        validation: Dict of validation results
    """
    T, N = returns.shape
    
    # Sample statistics
    sample_mean = returns.mean().mean()
    sample_std = returns.std().mean()
    sample_cov = returns.cov().values
    
    # Kurtosis (excess kurtosis, Gaussian = 0)
    kurtosis = returns.kurtosis().mean()
    
    # Covariance estimation error
    frob_error = np.linalg.norm(sample_cov - true_cov, 'fro') / np.linalg.norm(true_cov, 'fro')
    
    validation = {
        'T': T,
        'N': N,
        'sample_mean': sample_mean,
        'sample_std': sample_std,
        'annualized_vol': sample_std * np.sqrt(252),
        'excess_kurtosis': kurtosis,
        'frobenius_error': frob_error,
        'true_cov_condition': np.linalg.cond(true_cov),
    }
    
    return validation


def print_synthetic_summary(returns, true_cov, info):
    """Print summary of synthetic data."""
    val = validate_synthetic_data(returns, true_cov, info)
    
    print(f"\n{'='*60}")
    print(f"Synthetic Data: {info['name']}")
    print(f"{'='*60}")
    print(f"Description: {info['description']}")
    print(f"Dimensions: T={val['T']} observations, N={val['N']} assets")
    print(f"Sample mean: {val['sample_mean']:.6f}")
    print(f"Sample daily std: {val['sample_std']:.4f}")
    print(f"Annualized volatility: {val['annualized_vol']*100:.1f}%")
    print(f"Excess kurtosis: {val['excess_kurtosis']:.2f} (Gaussian=0)")
    print(f"SCM Frobenius error vs true: {val['frobenius_error']*100:.1f}%")
    print(f"True Σ condition number: {val['true_cov_condition']:.1f}")
    print(f"{'='*60}")


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("SYNTHETIC DATA GENERATION TEST")
    print("="*70)
    
    for dist_type in SYNTHETIC_TYPES.keys():
        returns, true_cov, info = generate_synthetic_returns(
            distribution=dist_type,
            N=100,
            T=1200,
            seed=42
        )
        print_synthetic_summary(returns, true_cov, info)
