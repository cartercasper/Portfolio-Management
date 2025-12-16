
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt
import requests

#########################################
# SCM
#########################################
# Calculate sample covariance matrix (SCM)
def sample_cov(daily_returns):
    
    # parameters:
        # daily_returns is matrix of daily returns
    # returns
        # sample covariance matrix (SCM)
    
    SCM = daily_returns.cov()

    return SCM


#########################################
# Shrinkage target
#########################################

def shrinkage_target(SCM):
    """
    Ledoit-Wolf (2004) shrinkage target.
    
    Following Ledoit and Wolf (2004), we use the "sample variance and mean
    covariance" target, defined as:
        F = diag(Σ_SCM) + σ̄(11^T - I)
    
    where:
        - diag(Σ_SCM) contains the sample variances of the assets
        - I is the identity matrix
        - σ̄ is the average of all off-diagonal elements of Σ_SCM
    
    This target preserves individual asset variances while imposing a
    homogeneous structure on all pairwise covariances.
    
    Parameters:
        SCM: Sample covariance matrix (pandas DataFrame)
    
    Returns:
        F: Shrinkage target (pandas DataFrame)
    """
    N = len(SCM)
    
    # Calculate average off-diagonal covariance
    off_diag_sum = SCM.values.sum() - np.trace(SCM.values)
    num_off_diag = N * (N - 1)
    mean_cov = off_diag_sum / num_off_diag
    
    # Build F: all elements = mean_cov, then replace diagonal with sample variances
    # This implements: F = diag(SCM) + σ̄(11^T - I)
    F = pd.DataFrame(
        np.full((N, N), mean_cov), 
        index=SCM.index, 
        columns=SCM.columns
    )
    np.fill_diagonal(F.values, np.diag(SCM))
    
    return F


#############################################
# Marchenko-Pastur/Clipped Eigenvalues
#############################################

# Marchenko-Pastur (MP) estimation
# IMPORTANT: assume IID returns in order to apply theory 

def MP_est(returns, SCM):

    # PARAMETERS:
      #   1. MATRIX OF RETURNS
      #   2. SAMPLE COVARIANCE MATRIX (SCM)
    # RETURNS:
      # 1. MARCHENKO-PASTUR CLIPPED EIGENVALUE COVARIANCE MATRIX 
    # NOTE:
        # USES THE MEAN OF MP BOUNDS AS CONSTANT FOR REPLACEMENT... MAY NEED TO CHANGE/UPDATE

    #calculate c = (#of stocks)/(#days of returns)
    c = returns.shape[1]/returns.shape[0]

    # calculate sigma2 = average of return variance
    sigma2 = np.trace(SCM)/len(np.diag(SCM))

    # Calculate MP eigenvalue bounds
    bounds = [sigma2*( (1-np.sqrt(c) )**2 ), sigma2*( (1+np.sqrt(c) )**2 )]


    # calculate eigenvalue decomposition of SCM
    eigvals, eigvecs = np.linalg.eigh(SCM)

    # calculate eigenvalues within MP bounds
    MP_eigs = (eigvals >= bounds[0]) & (eigvals <= bounds[1])

    # calculate the constant to replace the MP eigenvalues by (the mean of the eigenvalues within MP bounds)
    replace_constant = eigvals[MP_eigs].mean()

    # copy eigenvalues, replace MP bounds by constant above
    clip_eigs = eigvals.copy()
    clip_eigs[MP_eigs] = replace_constant

    # reformulate covariance matrix by eigvecs @ clip_eigs @ transpose(eigvecs)
    SMP = eigvecs @ np.diag(clip_eigs) @ np.transpose(eigvecs)

    return SMP


#############################################
# Tyler's M-Estimator
#############################################

def tyler_m_estimator(returns, SCM, max_iter=100, tol=1e-6, verbose=False):
    """
    Tyler's M-estimator (Tyler, 1987) - robust, distribution-free scatter estimator.
    
    Tyler's estimator is defined implicitly as the solution to:
        Σ_Tyler = (M/N) Σᵢ (xᵢ xᵢᵀ) / (xᵢᵀ Σ_Tyler⁻¹ xᵢ)
    
    subject to tr(Σ_Tyler) = M (normalization constraint).
    
    Since Tyler's estimator is scale-invariant (estimates shape, not magnitude),
    we rescale it to match the SCM's overall variance level:
        Σ̃_Tyler = (tr(Σ_SCM) / tr(Σ_Tyler)) × Σ_Tyler
    
    This ensures Σ̃_Tyler is on the same scale as Σ_SCM, F, and Σ_MP.
    
    Parameters:
        returns: Daily returns matrix (pandas DataFrame or numpy array)
        SCM: Sample covariance matrix (for rescaling)
        max_iter: Maximum iterations for fixed-point algorithm
        tol: Convergence tolerance
        verbose: If True, print convergence diagnostics
    
    Returns:
        Tyler_scaled: Rescaled Tyler's M-estimator (numpy array)
        If verbose=True, returns (Tyler_scaled, convergence_info) tuple
    
    Reference:
        Tyler, D. E. (1987). A distribution-free M-estimator of multivariate scatter.
    """
    # Convert to numpy if DataFrame
    if isinstance(returns, pd.DataFrame):
        X = returns.values
    else:
        X = returns
    
    if isinstance(SCM, pd.DataFrame):
        SCM_arr = SCM.values
    else:
        SCM_arr = SCM
    
    N, M = X.shape  # N = number of observations, M = number of assets
    
    # Center the data (mean-centered observations)
    X_centered = X - X.mean(axis=0)
    
    # Initialize with identity matrix (normalized)
    Sigma = np.eye(M)
    
    # Track convergence
    converged = False
    final_diff = None
    iterations_used = 0
    
    # Fixed-point iteration
    for iteration in range(max_iter):
        Sigma_old = Sigma.copy()
        iterations_used = iteration + 1
        
        try:
            # Compute inverse of current estimate
            Sigma_inv = np.linalg.inv(Sigma)
            
            # Compute weighted outer products
            Sigma_new = np.zeros((M, M))
            for i in range(N):
                x = X_centered[i, :]
                # Mahalanobis-like weight
                weight = x @ Sigma_inv @ x
                if weight > 1e-10:  # Avoid division by zero
                    Sigma_new += np.outer(x, x) / weight
            
            # Normalize: multiply by M/N and enforce trace constraint
            Sigma_new = (M / N) * Sigma_new
            
            # Enforce trace normalization: tr(Σ) = M
            Sigma = Sigma_new * (M / np.trace(Sigma_new))
            
            # Check convergence
            diff = np.linalg.norm(Sigma - Sigma_old, 'fro') / np.linalg.norm(Sigma_old, 'fro')
            final_diff = diff
            if diff < tol:
                converged = True
                break
                
        except np.linalg.LinAlgError:
            # If inversion fails, return scaled identity as fallback
            if verbose:
                return SCM_arr, {
                    'converged': False,
                    'iterations': iterations_used,
                    'max_iter': max_iter,
                    'final_diff': final_diff,
                    'tolerance': tol,
                    'error': 'LinAlgError - matrix inversion failed',
                    'N': N, 'M': M, 'ratio': N/M,
                    'trace_SCM': np.trace(SCM_arr),
                    'trace_Tyler': None,
                    'cond_SCM': np.linalg.cond(SCM_arr) if N > M else np.inf,
                    'cond_Tyler': np.inf,
                }
            return SCM_arr
    
    # Rescale to match SCM trace (Equation 4 from paper)
    # Σ̃_Tyler = (tr(Σ_SCM) / tr(Σ_Tyler)) × Σ_Tyler
    trace_SCM = np.trace(SCM_arr)
    trace_Tyler = np.trace(Sigma)
    
    if trace_Tyler > 1e-10:
        Tyler_scaled = (trace_SCM / trace_Tyler) * Sigma
    else:
        Tyler_scaled = SCM_arr  # Fallback
    
    # Compute condition number for diagnostics
    try:
        cond_tyler = np.linalg.cond(Tyler_scaled)
        cond_scm = np.linalg.cond(SCM_arr)
    except:
        cond_tyler = np.inf
        cond_scm = np.inf
    
    if verbose:
        convergence_info = {
            'converged': converged,
            'iterations': iterations_used,
            'max_iter': max_iter,
            'final_diff': final_diff,
            'tolerance': tol,
            'N': N,  # observations
            'M': M,  # assets
            'ratio': N / M,  # should be > 1 for Tyler to work well
            'trace_SCM': trace_SCM,
            'trace_Tyler': trace_Tyler,
            'cond_SCM': cond_scm,
            'cond_Tyler': cond_tyler,
        }
        return Tyler_scaled, convergence_info
    
    return Tyler_scaled



