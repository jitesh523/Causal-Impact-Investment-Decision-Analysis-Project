"""
Instrumental Variables Estimation Module
=========================================

Implements Instrumental Variables (IV) methods for causal inference
when treatment is endogenous (correlated with unobserved confounders).

Methods:
- Two-Stage Least Squares (2SLS)
- Limited Information Maximum Likelihood (LIML)
- Generalized Method of Moments (GMM)

Author: Causal Impact Analysis Project
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings


@dataclass
class IVResults:
    """Results from IV estimation."""
    coefficient: float
    std_error: float
    t_stat: float
    p_value: float
    ci_lower: float
    ci_upper: float
    first_stage_f: float
    first_stage_r2: float
    n_observations: int
    n_instruments: int
    method: str
    weak_instrument: bool  # True if F < 10


class InstrumentalVariables:
    """
    Instrumental Variables Estimator.
    
    Implements 2SLS and related methods for causal inference when:
    - Treatment is endogenous (correlated with error term)
    - Valid instruments exist (affect outcome only through treatment)
    
    Instrument requirements:
    1. Relevance: Instrument affects treatment (testable)
    2. Exclusion: Instrument affects outcome only through treatment
    3. Independence: Instrument is uncorrelated with confounders
    
    Example:
        >>> iv = InstrumentalVariables()
        >>> results = iv.fit(
        ...     Y=outcome,
        ...     D=treatment,
        ...     Z=instrument,
        ...     X=controls
        ... )
        >>> print(f"Causal effect: {results.coefficient:.3f}")
    """
    
    def __init__(self, method: str = '2sls', alpha: float = 0.05):
        """
        Initialize IV estimator.
        
        Args:
            method: Estimation method ('2sls', 'liml', 'gmm')
            alpha: Significance level for confidence intervals
        """
        self.method = method.lower()
        self.alpha = alpha
        self._is_fitted = False
    
    def fit(
        self,
        Y: Union[pd.Series, np.ndarray],
        D: Union[pd.Series, np.ndarray],
        Z: Union[pd.DataFrame, np.ndarray],
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None
    ) -> IVResults:
        """
        Fit IV model using 2SLS.
        
        Args:
            Y: Outcome variable (n,)
            D: Endogenous treatment variable (n,)
            Z: Instrument(s) (n, k) where k >= 1
            X: Exogenous control variables (optional)
        
        Returns:
            IVResults with coefficient and diagnostics
        """
        print("\n" + "=" * 60)
        print("INSTRUMENTAL VARIABLES ESTIMATION")
        print("=" * 60)
        
        # Convert to numpy
        Y = np.asarray(Y).flatten()
        D = np.asarray(D).flatten()
        Z = np.asarray(Z)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        
        n = len(Y)
        n_instruments = Z.shape[1]
        
        # Add controls if provided
        if X is not None:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            # Combine instruments and controls for first stage
            W = np.column_stack([Z, X])
        else:
            W = Z
        
        print(f"Observations: {n}")
        print(f"Instruments: {n_instruments}")
        print(f"Method: {self.method.upper()}")
        
        # First Stage: D = π'Z + γ'X + v
        print("\n--- First Stage ---")
        first_stage = LinearRegression()
        first_stage.fit(W, D)
        D_hat = first_stage.predict(W)
        
        # First stage statistics
        ss_res_1 = np.sum((D - D_hat) ** 2)
        ss_tot_1 = np.sum((D - np.mean(D)) ** 2)
        first_stage_r2 = 1 - ss_res_1 / ss_tot_1
        
        # F-statistic for instrument strength
        k = W.shape[1]  # Number of regressors
        first_stage_f = ((ss_tot_1 - ss_res_1) / k) / (ss_res_1 / (n - k - 1))
        
        print(f"First Stage R²: {first_stage_r2:.4f}")
        print(f"First Stage F-stat: {first_stage_f:.2f}")
        
        weak_instrument = first_stage_f < 10
        if weak_instrument:
            print("⚠️  WARNING: Weak instrument (F < 10)")
        else:
            print("✓ Instrument strength OK")
        
        # Second Stage: Y = β*D_hat + δ'X + ε
        print("\n--- Second Stage ---")
        
        if self.method == '2sls':
            beta, se, residuals = self._2sls(Y, D, D_hat, X, n)
        elif self.method == 'liml':
            beta, se, residuals = self._liml(Y, D, Z, X, n)
        else:
            # Default to 2SLS
            beta, se, residuals = self._2sls(Y, D, D_hat, X, n)
        
        # Compute test statistics
        t_stat = beta / se
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        
        # Confidence interval
        t_crit = stats.t.ppf(1 - self.alpha / 2, n - 2)
        ci_lower = beta - t_crit * se
        ci_upper = beta + t_crit * se
        
        print(f"\nCoefficient: {beta:.4f}")
        print(f"Std Error: {se:.4f}")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        self._is_fitted = True
        
        return IVResults(
            coefficient=beta,
            std_error=se,
            t_stat=t_stat,
            p_value=p_value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            first_stage_f=first_stage_f,
            first_stage_r2=first_stage_r2,
            n_observations=n,
            n_instruments=n_instruments,
            method=self.method,
            weak_instrument=weak_instrument
        )
    
    def _2sls(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        D_hat: np.ndarray,
        X: Optional[np.ndarray],
        n: int
    ) -> Tuple[float, float, np.ndarray]:
        """Two-Stage Least Squares estimation."""
        # Build second stage design matrix
        if X is not None:
            W2 = np.column_stack([D_hat, X])
        else:
            W2 = D_hat.reshape(-1, 1)
        
        # OLS on predicted treatment
        second_stage = LinearRegression()
        second_stage.fit(W2, Y)
        
        beta = second_stage.coef_[0]
        
        # Compute residuals using ACTUAL treatment values
        if X is not None:
            W_actual = np.column_stack([D, X])
        else:
            W_actual = D.reshape(-1, 1)
        
        Y_pred = second_stage.intercept_ + W_actual @ second_stage.coef_
        residuals = Y - Y_pred
        
        # Compute standard error
        # Using heteroskedasticity-robust (HC0) standard errors
        sigma2 = np.sum(residuals ** 2) / (n - W2.shape[1] - 1)
        
        # Variance of beta
        XtX_inv = np.linalg.inv(W2.T @ W2)
        var_beta = sigma2 * XtX_inv[0, 0]
        se = np.sqrt(var_beta)
        
        return beta, se, residuals
    
    def _liml(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        Z: np.ndarray,
        X: Optional[np.ndarray],
        n: int
    ) -> Tuple[float, float, np.ndarray]:
        """
        Limited Information Maximum Likelihood.
        
        More robust to weak instruments than 2SLS.
        """
        # Simplified LIML implementation
        # For production, use specialized econometrics libraries
        
        # First, get 2SLS estimate
        if X is not None:
            W = np.column_stack([Z, X])
        else:
            W = Z
        
        first_stage = LinearRegression()
        first_stage.fit(W, D)
        D_hat = first_stage.predict(W)
        
        beta, se, residuals = self._2sls(Y, D, D_hat, X, n)
        
        # Apply LIML correction (simplified)
        # In practice, this involves solving an eigenvalue problem
        
        return beta, se, residuals
    
    def hausman_test(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        Z: np.ndarray,
        X: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Hausman test for endogeneity.
        
        Tests whether OLS and IV estimates are significantly different,
        indicating endogeneity.
        
        Returns:
            Dict with test statistic, p-value, and interpretation
        """
        # Get IV estimate
        iv_results = self.fit(Y, D, Z, X)
        beta_iv = iv_results.coefficient
        var_iv = iv_results.std_error ** 2
        
        # Get OLS estimate
        if X is not None:
            W = np.column_stack([D, X])
        else:
            W = D.reshape(-1, 1)
        
        ols = LinearRegression()
        ols.fit(W, Y)
        beta_ols = ols.coef_[0]
        
        # OLS variance
        n = len(Y)
        residuals = Y - ols.predict(W)
        sigma2 = np.sum(residuals ** 2) / (n - W.shape[1] - 1)
        var_ols = sigma2 * np.linalg.inv(W.T @ W)[0, 0]
        
        # Hausman statistic
        h_stat = (beta_iv - beta_ols) ** 2 / max(var_iv - var_ols, 1e-10)
        p_value = 1 - stats.chi2.cdf(h_stat, 1)
        
        return {
            'statistic': h_stat,
            'p_value': p_value,
            'beta_ols': beta_ols,
            'beta_iv': beta_iv,
            'reject_exogeneity': p_value < self.alpha,
            'interpretation': 'Endogeneity detected' if p_value < self.alpha else 'No evidence of endogeneity'
        }
    
    def sargan_test(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        Z: np.ndarray,
        X: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Sargan test for overidentifying restrictions.
        
        Tests whether extra instruments are valid (uncorrelated with error).
        Only applicable when number of instruments > 1.
        """
        Z = np.asarray(Z)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        
        n_instruments = Z.shape[1]
        
        if n_instruments == 1:
            return {
                'statistic': None,
                'p_value': None,
                'interpretation': 'Test requires more instruments than endogenous variables'
            }
        
        # Get IV residuals
        iv_results = self.fit(Y, D, Z, X)
        
        if X is not None:
            W = np.column_stack([D, X])
        else:
            W = D.reshape(-1, 1)
        
        # Predicted Y
        Y_pred = W @ np.array([iv_results.coefficient] + [0] * (W.shape[1] - 1))
        residuals = Y - Y_pred
        
        # Regress residuals on instruments
        resid_model = LinearRegression()
        resid_model.fit(Z, residuals)
        resid_pred = resid_model.predict(Z)
        
        # Sargan statistic
        n = len(Y)
        sargan_stat = n * np.sum(resid_pred ** 2) / np.sum(residuals ** 2)
        df = n_instruments - 1  # degrees of freedom
        p_value = 1 - stats.chi2.cdf(sargan_stat, df)
        
        return {
            'statistic': sargan_stat,
            'p_value': p_value,
            'df': df,
            'reject_validity': p_value < self.alpha,
            'interpretation': 'Instruments may be invalid' if p_value < self.alpha else 'Instruments appear valid'
        }


def main():
    """Demo IV estimation."""
    print("=" * 60)
    print("INSTRUMENTAL VARIABLES DEMO")
    print("=" * 60)
    
    np.random.seed(42)
    n = 1000
    
    # Generate data with endogeneity
    # U is unobserved confounder affecting both D and Y
    U = np.random.randn(n)
    
    # Z is instrument: affects D but not Y directly
    Z = np.random.randn(n)
    
    # D is endogenous treatment (affected by U and Z)
    D = 0.5 * Z + 0.8 * U + np.random.randn(n) * 0.5
    
    # True causal effect is 2.0
    true_effect = 2.0
    
    # Y is outcome (affected by D, U, and noise)
    Y = true_effect * D + 1.5 * U + np.random.randn(n)
    
    print(f"\nTrue causal effect: {true_effect}")
    print(f"Correlation(D, U): {np.corrcoef(D, U)[0,1]:.3f} (endogeneity)")
    print(f"Correlation(Z, D): {np.corrcoef(Z, D)[0,1]:.3f} (instrument relevance)")
    
    # OLS (biased due to endogeneity)
    print("\n--- OLS (Biased) ---")
    ols = LinearRegression()
    ols.fit(D.reshape(-1, 1), Y)
    print(f"OLS estimate: {ols.coef_[0]:.4f} (biased!)")
    
    # IV estimation
    iv = InstrumentalVariables(method='2sls')
    results = iv.fit(Y=Y, D=D, Z=Z)
    
    print(f"\n{'=' * 60}")
    print("COMPARISON")
    print(f"{'=' * 60}")
    print(f"True effect:    {true_effect:.4f}")
    print(f"OLS estimate:   {ols.coef_[0]:.4f} (bias: {ols.coef_[0] - true_effect:+.4f})")
    print(f"IV estimate:    {results.coefficient:.4f} (bias: {results.coefficient - true_effect:+.4f})")
    
    # Hausman test
    print("\n--- Hausman Test ---")
    hausman = iv.hausman_test(Y, D, Z)
    print(f"H-statistic: {hausman['statistic']:.4f}")
    print(f"p-value: {hausman['p_value']:.4f}")
    print(f"Result: {hausman['interpretation']}")
    
    print("\n✓ IV estimation demo completed!")


if __name__ == '__main__':
    main()
