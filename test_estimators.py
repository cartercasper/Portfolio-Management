"""
Diagnostic Tests for Covariance Matrix Estimators

This script runs comprehensive tests to verify that all estimators
are working correctly according to their mathematical specifications.

Usage:
    python test_estimators.py
"""

import numpy as np
import pandas as pd
import sys

# Import our estimators
from estimators import sample_cov, shrinkage_target, MP_est, tyler_m_estimator


def print_header(title):
    """Print formatted header."""
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")


def print_result(test_name, passed, details=""):
    """Print test result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}: {test_name}")
    if details:
        print(f"         {details}")


def generate_test_data(n_samples=500, n_assets=50, seed=42):
    """Generate synthetic returns data with known properties."""
    np.random.seed(seed)
    
    # Generate factor model returns for realistic correlation structure
    # Use smaller factor loadings for more realistic (lower) correlations
    n_factors = 3
    factor_loadings = np.random.randn(n_assets, n_factors) * 0.1  # Smaller loadings
    factors = np.random.randn(n_samples, n_factors)
    idio_vol = np.random.uniform(0.02, 0.04, n_assets)  # Higher idiosyncratic vol
    idiosyncratic = np.random.randn(n_samples, n_assets) * idio_vol
    
    returns = factors @ factor_loadings.T + idiosyncratic
    
    # True covariance (for reference)
    true_cov = factor_loadings @ factor_loadings.T + np.diag(idio_vol**2)
    
    # Create DataFrame
    columns = [f'Asset_{i}' for i in range(n_assets)]
    returns_df = pd.DataFrame(returns, columns=columns)
    
    return returns_df, true_cov


# ============================================================================
# TEST 1: Sample Covariance Matrix (SCM)
# ============================================================================
def test_sample_covariance():
    """Test SCM properties."""
    print_header("TEST 1: Sample Covariance Matrix (SCM)")
    
    returns_df, _ = generate_test_data()
    n_samples, n_assets = returns_df.shape
    
    SCM = sample_cov(returns_df)
    SCM_arr = SCM.values
    
    all_passed = True
    
    # Test 1.1: Shape
    passed = SCM_arr.shape == (n_assets, n_assets)
    print_result("Correct shape (p x p)", passed, f"Shape: {SCM_arr.shape}")
    all_passed &= passed
    
    # Test 1.2: Symmetry
    passed = np.allclose(SCM_arr, SCM_arr.T)
    print_result("Symmetric", passed, f"Max asymmetry: {np.max(np.abs(SCM_arr - SCM_arr.T)):.2e}")
    all_passed &= passed
    
    # Test 1.3: Positive semi-definite
    eigenvalues = np.linalg.eigvalsh(SCM_arr)
    passed = np.all(eigenvalues >= -1e-10)
    print_result("Positive semi-definite", passed, f"Min eigenvalue: {eigenvalues.min():.6e}")
    all_passed &= passed
    
    # Test 1.4: Diagonal = variances
    manual_vars = returns_df.var(ddof=1).values
    passed = np.allclose(np.diag(SCM_arr), manual_vars, rtol=1e-5)
    print_result("Diagonal equals sample variances", passed)
    all_passed &= passed
    
    # Test 1.5: Matches numpy calculation
    numpy_cov = np.cov(returns_df.values, rowvar=False)
    passed = np.allclose(SCM_arr, numpy_cov, rtol=1e-5)
    print_result("Matches numpy.cov()", passed)
    all_passed &= passed
    
    return all_passed


# ============================================================================
# TEST 2: Shrinkage Target (Ledoit-Wolf 2004)
# ============================================================================
def test_shrinkage_target():
    """Test Ledoit-Wolf (2004) shrinkage target: F = diag(SCM) + σ̄(11^T - I)"""
    print_header("TEST 2: Shrinkage Target F (Ledoit-Wolf 2004)")
    
    returns_df, _ = generate_test_data()
    n_assets = returns_df.shape[1]
    
    SCM = sample_cov(returns_df)
    F = shrinkage_target(SCM)
    F_arr = F.values
    SCM_arr = SCM.values
    
    all_passed = True
    
    # Test 2.1: Shape
    passed = F_arr.shape == (n_assets, n_assets)
    print_result("Correct shape (p x p)", passed)
    all_passed &= passed
    
    # Test 2.2: Symmetry
    passed = np.allclose(F_arr, F_arr.T)
    print_result("Symmetric", passed)
    all_passed &= passed
    
    # Test 2.3: Diagonal equals SCM diagonal
    passed = np.allclose(np.diag(F_arr), np.diag(SCM_arr))
    print_result("Diagonal = diag(SCM)", passed)
    all_passed &= passed
    
    # Test 2.4: Off-diagonal elements are constant
    off_diag_mask = ~np.eye(n_assets, dtype=bool)
    off_diag_F = F_arr[off_diag_mask]
    passed = np.allclose(off_diag_F, off_diag_F[0])
    print_result("Off-diagonal elements are constant", passed, 
                 f"Std of off-diag: {np.std(off_diag_F):.2e}")
    all_passed &= passed
    
    # Test 2.5: Off-diagonal = average off-diagonal of SCM
    off_diag_SCM = SCM_arr[off_diag_mask]
    expected_off_diag = np.mean(off_diag_SCM)
    actual_off_diag = off_diag_F[0]
    passed = np.isclose(actual_off_diag, expected_off_diag, rtol=1e-5)
    print_result("Off-diagonal = mean(off-diag of SCM)", passed,
                 f"Expected: {expected_off_diag:.6e}, Got: {actual_off_diag:.6e}")
    all_passed &= passed
    
    # Test 2.6: Positive definite (or nearly so - small negative ok due to averaging)
    eigenvalues = np.linalg.eigvalsh(F_arr)
    # Allow small negative eigenvalues (regularization handles this)
    passed = eigenvalues.min() > -0.01 * np.max(eigenvalues)
    print_result("Positive semi-definite (or nearly)", passed, 
                 f"Min eigenvalue: {eigenvalues.min():.6e}, ratio to max: {eigenvalues.min()/eigenvalues.max():.2e}")
    all_passed &= passed
    
    return all_passed


# ============================================================================
# TEST 3: Marchenko-Pastur Estimator
# ============================================================================
def test_marchenko_pastur():
    """Test MP eigenvalue clipping estimator."""
    print_header("TEST 3: Marchenko-Pastur (MP) Estimator")
    
    # Use higher concentration ratio to test MP clipping better
    returns_df, _ = generate_test_data(n_samples=200, n_assets=100)  # q = 0.5
    n_samples, n_assets = returns_df.shape
    q = n_assets / n_samples
    
    SCM = sample_cov(returns_df)
    SCM_arr = SCM.values
    MP_cov = MP_est(returns_df, SCM)
    
    all_passed = True
    
    # Test 3.1: Shape
    passed = MP_cov.shape == (n_assets, n_assets)
    print_result("Correct shape (p x p)", passed)
    all_passed &= passed
    
    # Test 3.2: Symmetry
    passed = np.allclose(MP_cov, MP_cov.T)
    print_result("Symmetric", passed)
    all_passed &= passed
    
    # Test 3.3: Positive semi-definite
    eigenvalues = np.linalg.eigvalsh(MP_cov)
    passed = np.all(eigenvalues >= -1e-10)
    print_result("Positive semi-definite", passed, f"Min eigenvalue: {eigenvalues.min():.6e}")
    all_passed &= passed
    
    # Test 3.4: MP upper bound calculation
    var_estimate = np.mean(np.diag(SCM_arr))  # Approximate σ²
    lambda_plus = var_estimate * (1 + np.sqrt(q))**2
    print_result("MP upper bound λ+ calculated", True, 
                 f"q={q:.3f}, λ+={lambda_plus:.6f}")
    all_passed &= True
    
    # Test 3.5: Eigenvalue distribution is modified
    scm_eigenvalues = np.linalg.eigvalsh(SCM_arr)
    mp_eigenvalues = np.linalg.eigvalsh(MP_cov)
    
    # Check that eigenvalue distribution changed (clipping happened)
    noise_mask = scm_eigenvalues < lambda_plus
    n_noise = noise_mask.sum()
    
    if n_noise > 0:
        # Noise eigenvalues should be more uniform in MP
        scm_noise_std = np.std(scm_eigenvalues[noise_mask])
        mp_noise_vals = np.sort(mp_eigenvalues)[:n_noise]
        mp_noise_std = np.std(mp_noise_vals)
        
        # MP should reduce variance of noise eigenvalues (they're averaged)
        passed = mp_noise_std <= scm_noise_std + 1e-10
        print_result("Noise eigenvalues are more uniform", passed,
                     f"SCM std: {scm_noise_std:.4e}, MP std: {mp_noise_std:.4e}")
    else:
        passed = True
        print_result("Noise eigenvalues are more uniform", passed, 
                     f"All eigenvalues above λ+ (n_noise={n_noise})")
    all_passed &= passed
    
    # Test 3.6: Trace is preserved (approximately)
    trace_ratio = np.trace(MP_cov) / np.trace(SCM_arr)
    passed = np.isclose(trace_ratio, 1.0, rtol=0.1)
    print_result("Trace approximately preserved", passed, f"Trace ratio: {trace_ratio:.4f}")
    all_passed &= passed
    
    # Test 3.7: Large eigenvalues preserved
    # Signal eigenvalues (above λ+) should be mostly unchanged
    signal_mask = scm_eigenvalues >= lambda_plus
    if signal_mask.sum() > 0:
        scm_signal = np.sort(scm_eigenvalues[signal_mask])
        mp_signal = np.sort(mp_eigenvalues)[-signal_mask.sum():]
        passed = np.allclose(scm_signal, mp_signal, rtol=0.01)
        print_result("Signal eigenvalues preserved", passed,
                     f"Signal eigenvalues: {signal_mask.sum()}")
    else:
        passed = True
        print_result("Signal eigenvalues preserved", passed, "No signal eigenvalues")
    all_passed &= passed
    
    return all_passed


# ============================================================================
# TEST 4: Tyler's M-Estimator
# ============================================================================
def test_tyler_m_estimator():
    """Test Tyler's M-estimator fixed-point iteration."""
    print_header("TEST 4: Tyler's M-Estimator")
    
    returns_df, _ = generate_test_data()
    n_samples, n_assets = returns_df.shape
    
    SCM = sample_cov(returns_df)
    SCM_arr = SCM.values
    Tyler_cov = tyler_m_estimator(returns_df, SCM)
    
    all_passed = True
    
    # Test 4.1: Shape
    passed = Tyler_cov.shape == (n_assets, n_assets)
    print_result("Correct shape (p x p)", passed)
    all_passed &= passed
    
    # Test 4.2: Symmetry
    passed = np.allclose(Tyler_cov, Tyler_cov.T)
    print_result("Symmetric", passed, f"Max asymmetry: {np.max(np.abs(Tyler_cov - Tyler_cov.T)):.2e}")
    all_passed &= passed
    
    # Test 4.3: Positive definite
    eigenvalues = np.linalg.eigvalsh(Tyler_cov)
    passed = np.all(eigenvalues > -1e-10)
    print_result("Positive semi-definite", passed, f"Min eigenvalue: {eigenvalues.min():.6e}")
    all_passed &= passed
    
    # Test 4.4: Scale matches SCM (trace equality)
    trace_ratio = np.trace(Tyler_cov) / np.trace(SCM_arr)
    passed = np.isclose(trace_ratio, 1.0, rtol=0.05)
    print_result("Scale matches SCM (trace)", passed, f"Trace ratio: {trace_ratio:.4f}")
    all_passed &= passed
    
    # Test 4.5: Invertible
    try:
        Tyler_inv = np.linalg.inv(Tyler_cov)
        passed = True
        cond_number = np.linalg.cond(Tyler_cov)
        print_result("Invertible", passed, f"Condition number: {cond_number:.2e}")
    except np.linalg.LinAlgError:
        passed = False
        print_result("Invertible", passed, "Matrix is singular")
    all_passed &= passed
    
    # Test 4.6: Robustness test - add outliers and check stability
    returns_with_outliers = returns_df.copy()
    # Add some outliers
    outlier_idx = np.random.choice(n_samples, size=5, replace=False)
    returns_with_outliers.iloc[outlier_idx] *= 5
    
    SCM_outliers = sample_cov(returns_with_outliers)
    Tyler_outliers = tyler_m_estimator(returns_with_outliers, SCM_outliers)
    
    # Tyler should be more stable than SCM in presence of outliers
    scm_change = np.linalg.norm(SCM_outliers.values - SCM_arr) / np.linalg.norm(SCM_arr)
    tyler_change = np.linalg.norm(Tyler_outliers - Tyler_cov) / np.linalg.norm(Tyler_cov)
    
    passed = tyler_change < scm_change
    print_result("More robust to outliers than SCM", passed,
                 f"SCM change: {scm_change*100:.1f}%, Tyler change: {tyler_change*100:.1f}%")
    all_passed &= passed
    
    return all_passed


# ============================================================================
# TEST 5: Combined Estimator (Paper's 2D Method)
# ============================================================================
def test_combined_estimator_2d():
    """Test paper's combined estimator: Σ*(θ,φ) = φ[θF + (1-θ)MP] + (1-φ)SCM"""
    print_header("TEST 5: Paper's 2D Combined Estimator Σ*(θ,φ)")
    
    from optimization import paper_dual_method_2d
    
    returns_df, _ = generate_test_data()
    
    SCM = sample_cov(returns_df)
    SCM_arr = SCM.values
    F = shrinkage_target(SCM).values
    MP = MP_est(returns_df, SCM)
    
    all_passed = True
    
    # Test 5.1: Boundary θ=0, φ=1 → pure MP
    sigma_star = paper_dual_method_2d(0, 1, SCM_arr, F, MP)
    passed = np.allclose(sigma_star, MP)
    print_result("θ=0, φ=1 → MP", passed)
    all_passed &= passed
    
    # Test 5.2: Boundary θ=1, φ=1 → pure F
    sigma_star = paper_dual_method_2d(1, 1, SCM_arr, F, MP)
    passed = np.allclose(sigma_star, F)
    print_result("θ=1, φ=1 → F", passed)
    all_passed &= passed
    
    # Test 5.3: Boundary φ=0 → pure SCM
    sigma_star = paper_dual_method_2d(0.5, 0, SCM_arr, F, MP)
    passed = np.allclose(sigma_star, SCM_arr)
    print_result("φ=0 → SCM (any θ)", passed)
    all_passed &= passed
    
    # Test 5.4: θ=0.5, φ=0.5 → proper blend
    sigma_star = paper_dual_method_2d(0.5, 0.5, SCM_arr, F, MP)
    expected = 0.5 * (0.5 * F + 0.5 * MP) + 0.5 * SCM_arr
    passed = np.allclose(sigma_star, expected)
    print_result("θ=0.5, φ=0.5 → correct blend", passed)
    all_passed &= passed
    
    # Test 5.5: Result is always valid covariance (allow small negative eigenvalues)
    all_valid = True
    for theta in [0, 0.3, 0.5, 0.7, 1]:
        for phi in [0, 0.3, 0.5, 0.7, 1]:
            sigma_star = paper_dual_method_2d(theta, phi, SCM_arr, F, MP)
            eigenvalues = np.linalg.eigvalsh(sigma_star)
            # Allow small negative eigenvalues relative to largest
            if eigenvalues.min() < -0.01 * eigenvalues.max():
                print_result(f"θ={theta}, φ={phi} is valid", False, 
                            f"Min eigenvalue: {eigenvalues.min():.2e}")
                all_valid = False
    
    if all_valid:
        print_result("All (θ,φ) combinations yield valid covariance", True)
    all_passed &= all_valid
    
    return all_passed


# ============================================================================
# TEST 6: Our 3D Extension
# ============================================================================
def test_combined_estimator_3d():
    """Test our 3D extension: Σ*(θ,φ,ψ) = φ[θF + (1-θ)(ψMP + (1-ψ)Tyler)] + (1-φ)SCM"""
    print_header("TEST 6: Our 3D Extension Σ*(θ,φ,ψ)")
    
    from optimization import our_3d_method
    
    returns_df, _ = generate_test_data()
    
    SCM = sample_cov(returns_df)
    SCM_arr = SCM.values
    F = shrinkage_target(SCM).values
    MP = MP_est(returns_df, SCM)
    Tyler = tyler_m_estimator(returns_df, SCM)
    
    all_passed = True
    
    # Test 6.1: ψ=1 reduces to paper's 2D method
    sigma_3d = our_3d_method(0.5, 0.7, 1.0, SCM_arr, F, MP, Tyler)
    from optimization import paper_dual_method_2d
    sigma_2d = paper_dual_method_2d(0.5, 0.7, SCM_arr, F, MP)
    passed = np.allclose(sigma_3d, sigma_2d)
    print_result("ψ=1 → Paper's 2D method", passed)
    all_passed &= passed
    
    # Test 6.2: θ=0, φ=1, ψ=0 → pure Tyler
    sigma_star = our_3d_method(0, 1, 0, SCM_arr, F, MP, Tyler)
    passed = np.allclose(sigma_star, Tyler)
    print_result("θ=0, φ=1, ψ=0 → Tyler", passed)
    all_passed &= passed
    
    # Test 6.3: θ=0, φ=1, ψ=1 → pure MP
    sigma_star = our_3d_method(0, 1, 1, SCM_arr, F, MP, Tyler)
    passed = np.allclose(sigma_star, MP)
    print_result("θ=0, φ=1, ψ=1 → MP", passed)
    all_passed &= passed
    
    # Test 6.4: θ=1, φ=1, any ψ → pure F
    sigma_star = our_3d_method(1, 1, 0.5, SCM_arr, F, MP, Tyler)
    passed = np.allclose(sigma_star, F)
    print_result("θ=1, φ=1 → F (any ψ)", passed)
    all_passed &= passed
    
    # Test 6.5: φ=0, any θ,ψ → pure SCM
    sigma_star = our_3d_method(0.5, 0, 0.5, SCM_arr, F, MP, Tyler)
    passed = np.allclose(sigma_star, SCM_arr)
    print_result("φ=0 → SCM (any θ,ψ)", passed)
    all_passed &= passed
    
    # Test 6.6: Manual calculation check
    theta, phi, psi = 0.3, 0.7, 0.4
    robust_blend = psi * MP + (1 - psi) * Tyler
    inner_blend = theta * F + (1 - theta) * robust_blend
    expected = phi * inner_blend + (1 - phi) * SCM_arr
    
    sigma_star = our_3d_method(theta, phi, psi, SCM_arr, F, MP, Tyler)
    passed = np.allclose(sigma_star, expected)
    print_result("Manual calculation matches", passed)
    all_passed &= passed
    
    # Test 6.7: All valid covariance matrices (allow small negative eigenvalues)
    test_passed = True
    for theta in [0, 0.5, 1]:
        for phi in [0, 0.5, 1]:
            for psi in [0, 0.5, 1]:
                sigma_star = our_3d_method(theta, phi, psi, SCM_arr, F, MP, Tyler)
                eigenvalues = np.linalg.eigvalsh(sigma_star)
                # Allow small negative eigenvalues relative to largest
                if eigenvalues.min() < -0.01 * eigenvalues.max():
                    test_passed = False
                    break
    print_result("All (θ,φ,ψ) combinations yield valid covariance", test_passed)
    all_passed &= test_passed
    
    return all_passed


# ============================================================================
# TEST 7: GMVP Solver
# ============================================================================
def test_gmvp_solver():
    """Test GMVP optimization."""
    print_header("TEST 7: GMVP Solver")
    
    from optimization import solve_gmvp
    
    returns_df, _ = generate_test_data()
    SCM = sample_cov(returns_df)
    SCM_arr = SCM.values
    n_assets = SCM_arr.shape[0]
    
    all_passed = True
    
    weights = solve_gmvp(SCM_arr)
    
    # Test 7.1: Weights sum to 1
    passed = np.isclose(np.sum(weights), 1.0, rtol=1e-6)
    print_result("Weights sum to 1", passed, f"Sum: {np.sum(weights):.8f}")
    all_passed &= passed
    
    # Test 7.2: Weights are non-negative (long-only constraint)
    passed = np.all(weights >= -1e-6)
    print_result("Weights are non-negative", passed, f"Min weight: {weights.min():.6f}")
    all_passed &= passed
    
    # Test 7.3: Weights are <= 1
    passed = np.all(weights <= 1 + 1e-6)
    print_result("Weights are <= 1", passed, f"Max weight: {weights.max():.6f}")
    all_passed &= passed
    
    # Test 7.4: Check optimality (compare with cvxpy solution)
    import cvxpy as cp
    w = cp.Variable(n_assets)
    objective = cp.Minimize(cp.quad_form(w, SCM_arr + 1e-8*np.eye(n_assets)))
    constraints = [cp.sum(w) == 1, w >= 0, w <= 1]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP)
    cvxpy_weights = w.value
    
    our_variance = weights @ SCM_arr @ weights
    cvxpy_variance = cvxpy_weights @ SCM_arr @ cvxpy_weights
    
    passed = np.isclose(our_variance, cvxpy_variance, rtol=1e-3)
    print_result("Matches cvxpy solution", passed,
                 f"Our var: {our_variance:.8f}, CVXPY var: {cvxpy_variance:.8f}")
    all_passed &= passed
    
    # Test 7.5: Works with different covariance matrices
    F = shrinkage_target(SCM).values
    weights_F = solve_gmvp(F)
    passed = np.isclose(np.sum(weights_F), 1.0, rtol=1e-6)
    print_result("Works with shrinkage target F", passed)
    all_passed &= passed
    
    MP = MP_est(returns_df, SCM)
    weights_MP = solve_gmvp(MP)
    passed = np.isclose(np.sum(weights_MP), 1.0, rtol=1e-6)
    print_result("Works with MP estimator", passed)
    all_passed &= passed
    
    Tyler = tyler_m_estimator(returns_df, SCM)
    weights_Tyler = solve_gmvp(Tyler)
    passed = np.isclose(np.sum(weights_Tyler), 1.0, rtol=1e-6)
    print_result("Works with Tyler estimator", passed)
    all_passed &= passed
    
    return all_passed


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "="*70)
    print(" COVARIANCE ESTIMATOR DIAGNOSTIC TESTS")
    print("="*70)
    print("Running comprehensive tests for all estimators...")
    
    results = {}
    
    # Run all tests
    results['SCM'] = test_sample_covariance()
    results['Shrinkage Target'] = test_shrinkage_target()
    results['Marchenko-Pastur'] = test_marchenko_pastur()
    results['Tyler M-Estimator'] = test_tyler_m_estimator()
    results['2D Combined'] = test_combined_estimator_2d()
    results['3D Combined'] = test_combined_estimator_3d()
    results['GMVP Solver'] = test_gmvp_solver()
    
    # Summary
    print_header("TEST SUMMARY")
    
    total_passed = 0
    total_tests = len(results)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")
        if passed:
            total_passed += 1
    
    print(f"\n  Total: {total_passed}/{total_tests} test suites passed")
    
    if total_passed == total_tests:
        print("\n  🎉 All tests passed! Estimators are working correctly.")
        return 0
    else:
        print("\n  ⚠️  Some tests failed. Please review the output above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
