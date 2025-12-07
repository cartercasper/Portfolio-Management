
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

# Calculate Shrinkage Target = ST :sample variances on diagonal, all off diagonal elements are the same (mean sample covariance)

def shrinkage_target(SCM):

    #calculates shrinkage target (ST) based on sample covariance matrix SCM
        
    N = len(SCM)
    off_diag_sum = SCM.values.sum() - np.trace(SCM.values)
    num_off_diag = N*(N-1)
    mean_cov = off_diag_sum / num_off_diag

    # ST = shrinkage target
    ST = pd.DataFrame(np.full((N, N), mean_cov), index=SCM.index, columns=SCM.columns)
    np.fill_diagonal(ST.values, np.diag(SCM))

    return ST


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


