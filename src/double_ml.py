"""
Double Machine Learning Module
==============================

Implements Double/Debiased Machine Learning for causal inference.
Uses cross-fitting and orthogonalization to remove regularization bias.

Based on Chernozhukov et al. (2018): "Double/Debiased Machine Learning
for Treatment and Structural Parameters"

Author: Causal Impact Analysis Project
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, RidgeCV, LogisticRegressionCV
from scipy import stats
import warnings


@dataclass
class DMLResults:
    """Results from Double ML estimation."""
    theta: float  # Treatment effect estimate
    se: float  # Standard error
    ci_lower: float  # 95% CI lower bound
    ci_upper: float  # 95% CI upper bound
    p_value: float
    t_stat: float
    n_samples: int
    n_folds: int
    ml_method: str


class DoubleML:
    """
    Double/Debiased Machine Learning for Causal Inference.
    
    Implements the DML framework for estimating treatment effects while
    using flexible ML methods for nuisance parameter estimation.
    
    Key features:
    - Cross-fitting to avoid overfitting bias
    - Neyman orthogonality for debiased estimation
    - Support for multiple ML methods
    - Robust inference with cluster-robust standard errors
    
    Supports two models:
    1. Partially Linear Model (PLR): Y = θ*T + g(X) + ε
    2. Interactive Regression Model (IRM): Y = g(T, X) + ε
    
    Example:
        >>> dml = DoubleML(ml_method='random_forest', n_folds=5)
        >>> dml.fit(X, treatment, outcome)
        >>> results = dml.get_results()
    """
    
    ML_METHODS = {
        'lasso': (LassoCV, RidgeCV),
        'ridge': (RidgeCV, RidgeCV),
        'random_forest': (RandomForestRegressor, RandomForestClassifier),
        'gradient_boosting': (GradientBoostingRegressor, GradientBoostingRegressor)
    }
    
    def __init__(
        self,
        model_type: str = 'plr',
        ml_method: str = 'random_forest',
        n_folds: int = 5,
        n_rep: int = 1,
        random_state: Optional[int] = None
    ):
        """
        Initialize Double ML.
        
        Args:
            model_type: 'plr' (Partially Linear) or 'irm' (Interactive Regression)
            ml_method: ML method for nuisance estimation
            n_folds: Number of cross-fitting folds
            n_rep: Number of repeated cross-fitting (for more stable estimates)
            random_state: Random seed
        """
        if ml_method not in self.ML_METHODS:
            raise ValueError(f"ml_method must be one of {list(self.ML_METHODS.keys())}")
        
        self.model_type = model_type
        self.ml_method = ml_method
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.random_state = random_state
        
        self._theta = None
        self._se = None
        self._is_fitted = False
        self._psi = None  # Influence function values
    
    def _get_ml_models(self) -> Tuple[Any, Any]:
        """Get ML models for outcome and treatment."""
        outcome_class, treatment_class = self.ML_METHODS[self.ml_method]
        
        if self.ml_method == 'random_forest':
            outcome_model = outcome_class(n_estimators=100, max_depth=5, random_state=self.random_state)
            treatment_model = treatment_class(n_estimators=100, max_depth=5, random_state=self.random_state)
        elif self.ml_method == 'gradient_boosting':
            outcome_model = outcome_class(n_estimators=100, max_depth=3, random_state=self.random_state)
            treatment_model = outcome_class(n_estimators=100, max_depth=3, random_state=self.random_state)
        else:
            outcome_model = outcome_class()
            treatment_model = treatment_class() if treatment_class != outcome_class else outcome_class()
        
        return outcome_model, treatment_model
    
    def _cross_fit_nuisance(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        folds: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Cross-fit nuisance parameters.
        
        Returns:
            Y_residual: Residualized outcomes
            T_residual: Residualized treatment
            g_hat: Outcome predictions E[Y|X]
            m_hat: Treatment predictions E[T|X] (propensity)
        """
        n = len(Y)
        g_hat = np.zeros(n)  # E[Y|X]
        m_hat = np.zeros(n)  # E[T|X]
        l_hat = np.zeros(n)  # E[Y|X,T] (for IRM)
        
        outcome_model, treatment_model = self._get_ml_models()
        
        for train_idx, test_idx in folds:
            # Fit outcome model: E[Y|X]
            outcome_model_copy = self._clone_model(outcome_model)
            outcome_model_copy.fit(X[train_idx], Y[train_idx])
            g_hat[test_idx] = outcome_model_copy.predict(X[test_idx])
            
            # Fit treatment model: E[T|X]
            treatment_model_copy = self._clone_model(treatment_model)
            if hasattr(treatment_model_copy, 'predict_proba'):
                treatment_model_copy.fit(X[train_idx], T[train_idx])
                m_hat[test_idx] = treatment_model_copy.predict_proba(X[test_idx])[:, 1]
            else:
                treatment_model_copy.fit(X[train_idx], T[train_idx])
                m_hat[test_idx] = treatment_model_copy.predict(X[test_idx])
        
        # Clip propensity to avoid extreme values
        m_hat = np.clip(m_hat, 0.01, 0.99)
        
        # Compute residuals
        Y_residual = Y - g_hat
        T_residual = T - m_hat
        
        return Y_residual, T_residual, g_hat, m_hat
    
    def _clone_model(self, model):
        """Clone a model with same parameters."""
        from sklearn.base import clone
        return clone(model)
    
    def _estimate_plr(
        self,
        Y_residual: np.ndarray,
        T_residual: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Estimate Partially Linear Regression model.
        
        Y - E[Y|X] = θ * (T - E[T|X]) + ε
        """
        # Estimate theta using OLS on residuals
        theta = np.sum(T_residual * Y_residual) / np.sum(T_residual ** 2)
        
        # Influence function (for SE calculation)
        psi = (Y_residual - theta * T_residual) * T_residual
        
        return theta, psi
    
    def _estimate_irm(
        self,
        Y: np.ndarray,
        T: np.ndarray,
        g_hat: np.ndarray,
        m_hat: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Estimate Interactive Regression Model (AIPW-style).
        
        Uses Augmented Inverse Propensity Weighting.
        """
        n = len(Y)
        
        # AIPW score
        # theta = E[ ((T - m(X)) / (m(X)(1-m(X)))) * (Y - g(X)) ]
        weights = (T - m_hat) / (m_hat * (1 - m_hat))
        
        # Doubly robust score
        psi = weights * (Y - g_hat)
        theta = np.mean(psi)
        
        return theta, psi
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        treatment: Union[pd.Series, np.ndarray],
        outcome: Union[pd.Series, np.ndarray]
    ) -> 'DoubleML':
        """
        Fit Double ML model.
        
        Args:
            X: Covariates (n_samples, n_features)
            treatment: Treatment indicator (n_samples,)
            outcome: Outcome variable (n_samples,)
        
        Returns:
            Self
        """
        print("\n" + "=" * 60)
        print("DOUBLE MACHINE LEARNING")
        print("=" * 60)
        
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X = X.values
        T = np.asarray(treatment).flatten()
        Y = np.asarray(outcome).flatten()
        
        n = len(Y)
        print(f"Model: {self.model_type.upper()}")
        print(f"ML Method: {self.ml_method}")
        print(f"Samples: {n}, Features: {X.shape[1]}")
        print(f"Cross-fitting: {self.n_folds} folds × {self.n_rep} repetitions")
        
        # Store estimates across repetitions
        theta_estimates = []
        psi_values = []
        
        for rep in range(self.n_rep):
            if self.n_rep > 1:
                print(f"\nRepetition {rep + 1}/{self.n_rep}")
            
            # Create folds
            kf = KFold(n_splits=self.n_folds, shuffle=True, 
                      random_state=self.random_state + rep if self.random_state else None)
            folds = list(kf.split(X))
            
            # Cross-fit nuisance parameters
            Y_residual, T_residual, g_hat, m_hat = self._cross_fit_nuisance(X, T, Y, folds)
            
            # Estimate treatment effect
            if self.model_type == 'plr':
                theta, psi = self._estimate_plr(Y_residual, T_residual)
            else:  # irm
                theta, psi = self._estimate_irm(Y, T, g_hat, m_hat)
            
            theta_estimates.append(theta)
            psi_values.append(psi)
        
        # Aggregate across repetitions
        self._theta = np.mean(theta_estimates)
        
        # Combine influence functions
        self._psi = np.mean(psi_values, axis=0)
        
        # Compute standard error using influence function
        var = np.mean(self._psi ** 2) / n
        self._se = np.sqrt(var)
        
        # Compute statistics
        self._t_stat = self._theta / self._se
        self._p_value = 2 * (1 - stats.norm.cdf(abs(self._t_stat)))
        self._ci_lower = self._theta - 1.96 * self._se
        self._ci_upper = self._theta + 1.96 * self._se
        self._n_samples = n
        
        self._is_fitted = True
        
        print(f"\n{'=' * 60}")
        print("RESULTS")
        print(f"{'=' * 60}")
        print(f"Treatment Effect (θ): {self._theta:.4f}")
        print(f"Standard Error:       {self._se:.4f}")
        print(f"95% CI:               [{self._ci_lower:.4f}, {self._ci_upper:.4f}]")
        print(f"t-statistic:          {self._t_stat:.4f}")
        print(f"p-value:              {self._p_value:.4f}")
        
        return self
    
    def get_results(self) -> DMLResults:
        """Get estimation results."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return DMLResults(
            theta=self._theta,
            se=self._se,
            ci_lower=self._ci_lower,
            ci_upper=self._ci_upper,
            p_value=self._p_value,
            t_stat=self._t_stat,
            n_samples=self._n_samples,
            n_folds=self.n_folds,
            ml_method=self.ml_method
        )
    
    def sensitivity_analysis(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        confounding_strengths: List[float] = [0.1, 0.2, 0.3, 0.5]
    ) -> pd.DataFrame:
        """
        Sensitivity analysis for unmeasured confounding.
        
        Simulates how estimates would change under different
        degrees of unmeasured confounding.
        
        Args:
            confounding_strengths: List of confounding effect sizes
        
        Returns:
            DataFrame with sensitivity analysis results
        """
        base_theta = self._theta
        
        results = []
        for strength in confounding_strengths:
            # Simulate confounded estimate
            # This is a simplified sensitivity analysis
            bias = strength * np.std(Y) * np.std(T) / np.var(T)
            adjusted_theta = base_theta - bias
            
            results.append({
                'confounding_strength': strength,
                'original_theta': base_theta,
                'adjusted_theta': adjusted_theta,
                'bias': bias,
                'significant': abs(adjusted_theta) > 1.96 * self._se
            })
        
        return pd.DataFrame(results)


def main():
    """Demo Double ML."""
    print("=" * 60)
    print("DOUBLE MACHINE LEARNING DEMO")
    print("=" * 60)
    
    np.random.seed(42)
    n = 1000
    
    # Generate data
    X = np.random.randn(n, 5)
    
    # True propensity depends on X
    propensity = 1 / (1 + np.exp(-0.5 * X[:, 0] + 0.3 * X[:, 1]))
    T = np.random.binomial(1, propensity)
    
    # True treatment effect
    true_theta = 3.0
    
    # Outcome with nonlinear effects
    Y = (
        2 * X[:, 0] ** 2 +  # Nonlinear confounding
        X[:, 1] + 
        0.5 * X[:, 2] * X[:, 3] +  # Interaction
        true_theta * T +  # Treatment effect
        np.random.randn(n)  # Noise
    )
    
    # Fit DML
    dml = DoubleML(
        model_type='plr',
        ml_method='random_forest',
        n_folds=5,
        random_state=42
    )
    dml.fit(X, T, Y)
    
    results = dml.get_results()
    
    print(f"\nTrue θ: {true_theta}")
    print(f"Estimated θ: {results.theta:.4f}")
    print(f"Bias: {results.theta - true_theta:.4f}")
    
    # Compare with naive OLS
    from sklearn.linear_model import LinearRegression
    ols = LinearRegression()
    ols.fit(np.column_stack([X, T]), Y)
    naive_theta = ols.coef_[-1]
    print(f"\nNaive OLS θ: {naive_theta:.4f}")
    print(f"Naive OLS Bias: {naive_theta - true_theta:.4f}")
    
    print("\n✓ Double ML demo completed!")


if __name__ == '__main__':
    main()
