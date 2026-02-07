"""
Causal Forest Module
====================

Implements Causal Forest for estimating heterogeneous treatment effects (HTE).
Uses tree-based methods to discover how treatment effects vary across subgroups.

Based on Generalized Random Forests (GRF) methodology from:
Athey, Tibshirani, and Wager (2019).

Author: Causal Impact Analysis Project
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.tree import DecisionTreeRegressor
import warnings


@dataclass
class CausalForestResults:
    """Results from Causal Forest estimation."""
    ate: float  # Average Treatment Effect
    ate_se: float  # Standard error of ATE
    cate: np.ndarray  # Conditional Average Treatment Effects
    cate_se: np.ndarray  # Standard errors of CATEs
    feature_importance: Dict[str, float]
    variable_importance: pd.DataFrame
    n_treated: int
    n_control: int


class CausalForest:
    """
    Causal Forest for Heterogeneous Treatment Effect Estimation.
    
    Implements a simplified version of the Generalized Random Forest algorithm
    for estimating conditional average treatment effects (CATEs).
    
    Uses orthogonalization (double ML) to reduce bias:
    1. Estimate propensity score P(T=1|X)
    2. Estimate outcome model E[Y|X]
    3. Train forest on residualized outcomes
    
    Example:
        >>> cf = CausalForest(n_estimators=100)
        >>> cf.fit(X, treatment, outcome)
        >>> cates = cf.predict(X_new)
        >>> results = cf.get_results()
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 5,
        max_features: str = 'sqrt',
        honesty: bool = True,
        honesty_fraction: float = 0.5,
        random_state: Optional[int] = None
    ):
        """
        Initialize Causal Forest.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth (None for unlimited)
            min_samples_leaf: Minimum samples per leaf
            max_features: Features to consider at each split
            honesty: Use honest estimation (separate train/estimation samples)
            honesty_fraction: Fraction of data for honest estimation
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.honesty = honesty
        self.honesty_fraction = honesty_fraction
        self.random_state = random_state
        
        self._propensity_model = None
        self._outcome_model = None
        self._forest = None
        self._feature_names = None
        self._is_fitted = False
        
        self.cate_ = None
        self.cate_se_ = None
        self.ate_ = None
        self.ate_se_ = None
    
    def _estimate_propensity(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Estimate propensity scores using cross-fitting."""
        self._propensity_model = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=self.random_state
        )
        
        # Cross-fitted propensity scores to avoid overfitting
        propensity = cross_val_predict(
            self._propensity_model,
            X, T,
            cv=3,
            method='predict_proba'
        )[:, 1]
        
        # Clip to avoid extreme weights
        propensity = np.clip(propensity, 0.01, 0.99)
        
        # Fit final model for prediction
        self._propensity_model.fit(X, T)
        
        return propensity
    
    def _estimate_outcome(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Estimate outcome model using cross-fitting."""
        self._outcome_model = RandomForestRegressor(
            n_estimators=50,
            max_depth=5,
            random_state=self.random_state
        )
        
        # Cross-fitted outcome predictions
        outcome_hat = cross_val_predict(
            self._outcome_model,
            X, Y,
            cv=3
        )
        
        # Fit final model
        self._outcome_model.fit(X, Y)
        
        return outcome_hat
    
    def _build_causal_tree(
        self,
        X: np.ndarray,
        pseudo_outcomes: np.ndarray,
        weights: np.ndarray,
        indices: np.ndarray
    ) -> DecisionTreeRegressor:
        """Build a single causal tree with weighted samples."""
        tree = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state
        )
        
        # Bootstrap sample
        n = len(indices)
        boot_indices = np.random.choice(indices, size=n, replace=True)
        
        if self.honesty:
            # Split into structure and estimation samples
            n_structure = int(n * (1 - self.honesty_fraction))
            structure_idx = boot_indices[:n_structure]
            estimation_idx = boot_indices[n_structure:]
            
            # Fit tree structure on first half
            tree.fit(
                X[structure_idx],
                pseudo_outcomes[structure_idx],
                sample_weight=weights[structure_idx]
            )
            
            # Re-estimate leaf values using second half (honest estimation)
            # This is a simplified version - full GRF uses local centering
            leaf_ids = tree.apply(X[estimation_idx])
            unique_leaves = np.unique(leaf_ids)
            
            for leaf in unique_leaves:
                mask = leaf_ids == leaf
                if mask.sum() > 0:
                    # Update leaf value with honest estimate
                    leaf_value = np.average(
                        pseudo_outcomes[estimation_idx][mask],
                        weights=weights[estimation_idx][mask]
                    )
                    tree.tree_.value[leaf, 0, 0] = leaf_value
        else:
            tree.fit(
                X[boot_indices],
                pseudo_outcomes[boot_indices],
                sample_weight=weights[boot_indices]
            )
        
        return tree
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        treatment: Union[pd.Series, np.ndarray],
        outcome: Union[pd.Series, np.ndarray]
    ) -> 'CausalForest':
        """
        Fit Causal Forest to data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            treatment: Binary treatment indicator (n_samples,)
            outcome: Outcome variable (n_samples,)
        
        Returns:
            Self
        """
        print("\n" + "=" * 60)
        print("CAUSAL FOREST ESTIMATION")
        print("=" * 60)
        
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
            X = X.values
        else:
            self._feature_names = [f'X{i}' for i in range(X.shape[1])]
        
        T = np.asarray(treatment).flatten()
        Y = np.asarray(outcome).flatten()
        
        n_samples = len(Y)
        n_treated = T.sum()
        n_control = n_samples - n_treated
        
        print(f"Samples: {n_samples} (treated: {n_treated}, control: {n_control})")
        
        # Step 1: Estimate nuisance functions
        print("\n1. Estimating propensity scores...")
        propensity = self._estimate_propensity(X, T)
        
        print("2. Estimating outcome model...")
        outcome_hat = self._estimate_outcome(X, Y)
        
        # Step 2: Compute pseudo-outcomes (doubly robust formula)
        print("3. Computing pseudo-outcomes...")
        
        # Inverse propensity weights
        weights = T / propensity + (1 - T) / (1 - propensity)
        
        # Pseudo-outcome for CATE estimation
        # tau(x) = E[Y(1) - Y(0) | X=x]
        pseudo_outcomes = (
            (T - propensity) / (propensity * (1 - propensity)) * 
            (Y - outcome_hat)
        )
        
        # Step 3: Build forest on pseudo-outcomes
        print(f"4. Building causal forest ({self.n_estimators} trees)...")
        
        self._forest = []
        indices = np.arange(n_samples)
        
        np.random.seed(self.random_state)
        for i in range(self.n_estimators):
            tree = self._build_causal_tree(X, pseudo_outcomes, np.ones(n_samples), indices)
            self._forest.append(tree)
        
        # Step 4: Compute CATEs and ATE
        print("5. Computing treatment effects...")
        
        self.cate_, self.cate_se_ = self._predict_with_se(X)
        self.ate_ = np.mean(self.cate_)
        self.ate_se_ = np.std(self.cate_) / np.sqrt(n_samples)
        
        print(f"\n✓ Average Treatment Effect (ATE): {self.ate_:.4f} (SE: {self.ate_se_:.4f})")
        
        self._is_fitted = True
        self._n_treated = n_treated
        self._n_control = n_control
        
        return self
    
    def _predict_with_se(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions with standard errors from forest."""
        predictions = np.zeros((len(X), self.n_estimators))
        
        for i, tree in enumerate(self._forest):
            predictions[:, i] = tree.predict(X)
        
        cate = np.mean(predictions, axis=1)
        cate_se = np.std(predictions, axis=1) / np.sqrt(self.n_estimators)
        
        return cate, cate_se
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        return_se: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict CATEs for new observations.
        
        Args:
            X: Feature matrix
            return_se: Return standard errors
        
        Returns:
            CATEs (and optionally standard errors)
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        cate, cate_se = self._predict_with_se(X)
        
        if return_se:
            return cate, cate_se
        return cate
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance for treatment effect heterogeneity."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Aggregate importance across trees
        importances = np.zeros(len(self._feature_names))
        
        for tree in self._forest:
            importances += tree.feature_importances_
        
        importances /= self.n_estimators
        
        df = pd.DataFrame({
            'feature': self._feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return df
    
    def get_heterogeneity_groups(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        n_groups: int = 4
    ) -> pd.DataFrame:
        """
        Identify groups with different treatment effects.
        
        Args:
            X: Feature matrix
            n_groups: Number of groups to create
        
        Returns:
            DataFrame with group assignments and statistics
        """
        cates = self.predict(X)
        
        # Create groups based on CATE quantiles
        quantiles = np.percentile(cates, np.linspace(0, 100, n_groups + 1))
        group_labels = np.digitize(cates, quantiles[1:-1])
        
        results = []
        for g in range(n_groups):
            mask = group_labels == g
            results.append({
                'group': g + 1,
                'n_samples': mask.sum(),
                'mean_cate': cates[mask].mean(),
                'std_cate': cates[mask].std(),
                'min_cate': cates[mask].min(),
                'max_cate': cates[mask].max()
            })
        
        return pd.DataFrame(results)
    
    def get_results(self) -> CausalForestResults:
        """Get comprehensive results."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        importance_df = self.get_feature_importance()
        
        return CausalForestResults(
            ate=self.ate_,
            ate_se=self.ate_se_,
            cate=self.cate_,
            cate_se=self.cate_se_,
            feature_importance=dict(zip(importance_df['feature'], importance_df['importance'])),
            variable_importance=importance_df,
            n_treated=self._n_treated,
            n_control=self._n_control
        )
    
    def plot_cate_distribution(self, save_path: Optional[str] = None):
        """Plot distribution of CATEs."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogram
        axes[0].hist(self.cate_, bins=30, edgecolor='black', alpha=0.7, color='#2E86AB')
        axes[0].axvline(self.ate_, color='red', linestyle='--', linewidth=2, label=f'ATE = {self.ate_:.3f}')
        axes[0].axvline(0, color='gray', linestyle=':', linewidth=1)
        axes[0].set_xlabel('CATE')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Treatment Effects')
        axes[0].legend()
        
        # Feature importance
        importance_df = self.get_feature_importance().head(10)
        axes[1].barh(importance_df['feature'], importance_df['importance'], color='#06A77D')
        axes[1].set_xlabel('Importance')
        axes[1].set_title('Top 10 Features for Effect Heterogeneity')
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
        return fig


def main():
    """Demo Causal Forest."""
    print("=" * 60)
    print("CAUSAL FOREST DEMO")
    print("=" * 60)
    
    np.random.seed(42)
    n = 1000
    
    # Generate data with heterogeneous treatment effects
    X = np.random.randn(n, 5)
    X_df = pd.DataFrame(X, columns=['age', 'income', 'education', 'engagement', 'tenure'])
    
    # Treatment depends on features
    propensity = 1 / (1 + np.exp(-0.5 * X[:, 0] + 0.3 * X[:, 1]))
    T = np.random.binomial(1, propensity)
    
    # Heterogeneous treatment effect: CATE varies with age and income
    true_cate = 5 + 3 * X[:, 0] - 2 * X[:, 1]  # Effect varies by age (+) and income (-)
    
    # Outcome
    Y = 10 + 2 * X[:, 0] + X[:, 1] + T * true_cate + np.random.randn(n) * 2
    
    # Fit Causal Forest
    cf = CausalForest(n_estimators=50, random_state=42)
    cf.fit(X_df, T, Y)
    
    # Get results
    results = cf.get_results()
    
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"True ATE:      {np.mean(true_cate):.4f}")
    print(f"Estimated ATE: {results.ate:.4f} (SE: {results.ate_se:.4f})")
    
    # Heterogeneity groups
    print("\nTreatment Effect Heterogeneity:")
    groups = cf.get_heterogeneity_groups(X_df, n_groups=4)
    print(groups.to_string(index=False))
    
    # Feature importance
    print("\nFeature Importance for Heterogeneity:")
    print(results.variable_importance.to_string(index=False))
    
    # Correlation with true CATE
    correlation = np.corrcoef(true_cate, results.cate)[0, 1]
    print(f"\nCorrelation with true CATE: {correlation:.4f}")
    
    print("\n✓ Causal Forest demo completed!")


if __name__ == '__main__':
    main()
