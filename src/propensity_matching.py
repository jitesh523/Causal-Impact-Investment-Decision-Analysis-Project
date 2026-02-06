"""
Propensity Score Matching Module
================================

Implements propensity score matching for creating balanced treatment/control groups
in observational studies. Uses logistic regression to estimate propensity scores
and nearest-neighbor matching with optional caliper.

Author: Causal Impact Analysis Project
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
import warnings

warnings.filterwarnings('ignore')


class PropensityMatcher:
    """
    Propensity Score Matching for causal inference.
    
    Uses logistic regression to estimate propensity scores (probability of
    treatment given covariates) and performs nearest-neighbor matching to
    create balanced treatment and control groups.
    
    Example:
        >>> matcher = PropensityMatcher(data, treatment_col='treated', 
        ...                              covariates=['age', 'income', 'region'])
        >>> matcher.fit()
        >>> matched_data = matcher.match(caliper=0.1)
        >>> matcher.plot_balance()
    """
    
    def __init__(
        self, 
        data: pd.DataFrame, 
        treatment_col: str,
        covariates: List[str],
        outcome_col: Optional[str] = None
    ):
        """
        Initialize PropensityMatcher.
        
        Args:
            data: DataFrame containing treatment, covariates, and optionally outcome
            treatment_col: Name of binary treatment column (0/1)
            covariates: List of covariate column names for propensity model
            outcome_col: Optional name of outcome column for effect estimation
        """
        self.data = data.copy()
        self.treatment_col = treatment_col
        self.covariates = covariates
        self.outcome_col = outcome_col
        
        self.propensity_scores = None
        self.matched_data = None
        self.matched_pairs = None
        self.model = None
        self.scaler = None
        self.balance_stats = None
        
        # Validate inputs
        self._validate_inputs()
    
    def _validate_inputs(self):
        """Validate input data and columns."""
        if self.treatment_col not in self.data.columns:
            raise ValueError(f"Treatment column '{self.treatment_col}' not found")
        
        for cov in self.covariates:
            if cov not in self.data.columns:
                raise ValueError(f"Covariate '{cov}' not found in data")
        
        treatment_vals = self.data[self.treatment_col].unique()
        if not set(treatment_vals).issubset({0, 1}):
            raise ValueError("Treatment column must be binary (0/1)")
    
    def fit(self, C: float = 1.0) -> 'PropensityMatcher':
        """
        Fit propensity score model using logistic regression.
        
        Args:
            C: Regularization strength (inverse of lambda). Lower = more regularization.
        
        Returns:
            self for method chaining
        """
        print("Fitting propensity score model...")
        
        # Prepare features
        X = self.data[self.covariates].copy()
        y = self.data[self.treatment_col].values
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit logistic regression
        self.model = LogisticRegression(C=C, max_iter=1000, random_state=42)
        self.model.fit(X_scaled, y)
        
        # Get propensity scores (probability of treatment)
        self.propensity_scores = self.model.predict_proba(X_scaled)[:, 1]
        self.data['propensity_score'] = self.propensity_scores
        
        print(f"✓ Model fitted. AUC: {self._calculate_auc():.4f}")
        print(f"  Treated: {(y == 1).sum()}, Control: {(y == 0).sum()}")
        
        return self
    
    def _calculate_auc(self) -> float:
        """Calculate AUC for propensity model."""
        from sklearn.metrics import roc_auc_score
        y = self.data[self.treatment_col].values
        return roc_auc_score(y, self.propensity_scores)
    
    def match(
        self, 
        method: str = 'nearest',
        caliper: Optional[float] = None,
        replacement: bool = False,
        ratio: int = 1
    ) -> pd.DataFrame:
        """
        Match treatment and control units based on propensity scores.
        
        Args:
            method: Matching method ('nearest' for nearest-neighbor)
            caliper: Maximum allowed difference in propensity scores (None = no limit)
            replacement: Whether to allow matching with replacement
            ratio: Number of control units to match per treated unit
        
        Returns:
            DataFrame containing matched pairs
        """
        if self.propensity_scores is None:
            raise ValueError("Must call fit() before match()")
        
        print(f"\nMatching (method={method}, caliper={caliper}, ratio={ratio})...")
        
        # Separate treatment and control
        treated_mask = self.data[self.treatment_col] == 1
        treated = self.data[treated_mask].copy()
        control = self.data[~treated_mask].copy()
        
        treated_scores = treated['propensity_score'].values
        control_scores = control['propensity_score'].values
        
        # Calculate pairwise distances
        distances = cdist(
            treated_scores.reshape(-1, 1),
            control_scores.reshape(-1, 1),
            metric='euclidean'
        )
        
        # Perform matching
        matched_treated_idx = []
        matched_control_idx = []
        used_controls = set()
        
        for i, t_score in enumerate(treated_scores):
            # Get distances to all controls
            dists = distances[i, :]
            
            # Sort by distance
            sorted_idx = np.argsort(dists)
            
            matches_found = 0
            for j in sorted_idx:
                if matches_found >= ratio:
                    break
                
                # Skip if already used (unless with replacement)
                if not replacement and j in used_controls:
                    continue
                
                # Check caliper constraint
                if caliper is not None and dists[j] > caliper:
                    break
                
                matched_treated_idx.append(treated.index[i])
                matched_control_idx.append(control.index[j])
                
                if not replacement:
                    used_controls.add(j)
                
                matches_found += 1
        
        # Create matched dataset
        matched_treated = self.data.loc[matched_treated_idx].copy()
        matched_treated['match_id'] = range(len(matched_treated))
        matched_treated['matched_group'] = 'treated'
        
        matched_control = self.data.loc[matched_control_idx].copy()
        matched_control['match_id'] = range(len(matched_control))
        matched_control['matched_group'] = 'control'
        
        self.matched_data = pd.concat([matched_treated, matched_control], ignore_index=True)
        self.matched_pairs = list(zip(matched_treated_idx, matched_control_idx))
        
        n_matched = len(matched_treated)
        n_treated = len(treated)
        print(f"✓ Matched {n_matched}/{n_treated} treated units ({100*n_matched/n_treated:.1f}%)")
        
        return self.matched_data
    
    def calculate_balance(self) -> pd.DataFrame:
        """
        Calculate balance diagnostics (standardized mean differences).
        
        Returns:
            DataFrame with balance statistics for each covariate
        """
        if self.matched_data is None:
            raise ValueError("Must call match() before calculate_balance()")
        
        balance_stats = []
        
        for cov in self.covariates:
            # Before matching
            treated_before = self.data[self.data[self.treatment_col] == 1][cov]
            control_before = self.data[self.data[self.treatment_col] == 0][cov]
            
            smd_before = self._standardized_mean_diff(treated_before, control_before)
            
            # After matching
            treated_after = self.matched_data[
                self.matched_data[self.treatment_col] == 1
            ][cov]
            control_after = self.matched_data[
                self.matched_data[self.treatment_col] == 0
            ][cov]
            
            smd_after = self._standardized_mean_diff(treated_after, control_after)
            
            balance_stats.append({
                'covariate': cov,
                'smd_before': smd_before,
                'smd_after': smd_after,
                'improvement': abs(smd_before) - abs(smd_after)
            })
        
        self.balance_stats = pd.DataFrame(balance_stats)
        return self.balance_stats
    
    def _standardized_mean_diff(self, treated: pd.Series, control: pd.Series) -> float:
        """Calculate standardized mean difference."""
        mean_diff = treated.mean() - control.mean()
        pooled_std = np.sqrt((treated.var() + control.var()) / 2)
        return mean_diff / pooled_std if pooled_std > 0 else 0
    
    def estimate_ate(self) -> Dict[str, float]:
        """
        Estimate Average Treatment Effect on the Treated (ATT).
        
        Returns:
            Dictionary with treatment effect estimates
        """
        if self.matched_data is None:
            raise ValueError("Must call match() before estimate_ate()")
        
        if self.outcome_col is None:
            raise ValueError("outcome_col must be specified to estimate treatment effect")
        
        treated_outcomes = self.matched_data[
            self.matched_data[self.treatment_col] == 1
        ][self.outcome_col]
        
        control_outcomes = self.matched_data[
            self.matched_data[self.treatment_col] == 0
        ][self.outcome_col]
        
        # Calculate ATT
        att = treated_outcomes.mean() - control_outcomes.mean()
        
        # Standard error (assuming independent samples)
        se = np.sqrt(
            treated_outcomes.var() / len(treated_outcomes) +
            control_outcomes.var() / len(control_outcomes)
        )
        
        # T-test
        t_stat, p_value = stats.ttest_ind(treated_outcomes, control_outcomes)
        
        # Confidence interval
        ci_lower = att - 1.96 * se
        ci_upper = att + 1.96 * se
        
        results = {
            'att': att,
            'std_error': se,
            't_statistic': t_stat,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_treated': len(treated_outcomes),
            'n_control': len(control_outcomes)
        }
        
        print("\n" + "=" * 60)
        print("AVERAGE TREATMENT EFFECT ON TREATED (ATT)")
        print("=" * 60)
        print(f"ATT Estimate: {att:.4f}")
        print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print("✓ Effect is statistically significant at α=0.05")
        else:
            print("⚠ Effect is NOT statistically significant at α=0.05")
        
        return results
    
    def plot_propensity_distribution(self, save_path: Optional[str] = None):
        """
        Plot propensity score distributions for treated and control groups.
        
        Args:
            save_path: Optional path to save the figure
        """
        if self.propensity_scores is None:
            raise ValueError("Must call fit() before plotting")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        treated_mask = self.data[self.treatment_col] == 1
        
        # Before matching
        ax1 = axes[0]
        ax1.hist(
            self.data[treated_mask]['propensity_score'],
            bins=30, alpha=0.7, label='Treated', color='#2E86AB', density=True
        )
        ax1.hist(
            self.data[~treated_mask]['propensity_score'],
            bins=30, alpha=0.7, label='Control', color='#F18F01', density=True
        )
        ax1.set_xlabel('Propensity Score', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title('Before Matching', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # After matching (if available)
        ax2 = axes[1]
        if self.matched_data is not None:
            matched_treated = self.matched_data[
                self.matched_data[self.treatment_col] == 1
            ]['propensity_score']
            matched_control = self.matched_data[
                self.matched_data[self.treatment_col] == 0
            ]['propensity_score']
            
            ax2.hist(matched_treated, bins=30, alpha=0.7, label='Treated', 
                    color='#2E86AB', density=True)
            ax2.hist(matched_control, bins=30, alpha=0.7, label='Control',
                    color='#F18F01', density=True)
            ax2.set_title('After Matching', fontsize=14, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'Run match() first', ha='center', va='center',
                    fontsize=14, transform=ax2.transAxes)
            ax2.set_title('After Matching (Not Yet Performed)', fontsize=14)
        
        ax2.set_xlabel('Propensity Score', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot: {save_path}")
        
        plt.close()
        return fig
    
    def plot_balance(self, save_path: Optional[str] = None):
        """
        Plot covariate balance before and after matching.
        
        Args:
            save_path: Optional path to save the figure
        """
        if self.balance_stats is None:
            self.calculate_balance()
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(self.covariates) * 0.5)))
        
        y_pos = np.arange(len(self.covariates))
        
        # Plot SMD before and after
        ax.barh(y_pos - 0.2, self.balance_stats['smd_before'].abs(), 
                height=0.4, label='Before Matching', color='#E63946', alpha=0.8)
        ax.barh(y_pos + 0.2, self.balance_stats['smd_after'].abs(),
                height=0.4, label='After Matching', color='#06A77D', alpha=0.8)
        
        # Add threshold line
        ax.axvline(x=0.1, color='black', linestyle='--', linewidth=2, 
                   label='Balance Threshold (0.1)')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(self.balance_stats['covariate'])
        ax.set_xlabel('Absolute Standardized Mean Difference', fontsize=12)
        ax.set_title('Covariate Balance: Before vs After Matching', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot: {save_path}")
        
        plt.close()
        return fig
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of matching results.
        
        Returns:
            Dictionary containing all key results
        """
        summary = {
            'n_original_treated': (self.data[self.treatment_col] == 1).sum(),
            'n_original_control': (self.data[self.treatment_col] == 0).sum(),
            'propensity_auc': self._calculate_auc() if self.model else None,
        }
        
        if self.matched_data is not None:
            summary['n_matched_treated'] = (
                self.matched_data[self.treatment_col] == 1
            ).sum()
            summary['n_matched_control'] = (
                self.matched_data[self.treatment_col] == 0
            ).sum()
            summary['match_rate'] = (
                summary['n_matched_treated'] / summary['n_original_treated']
            )
        
        if self.balance_stats is not None:
            summary['mean_smd_before'] = self.balance_stats['smd_before'].abs().mean()
            summary['mean_smd_after'] = self.balance_stats['smd_after'].abs().mean()
            summary['balanced_covariates'] = (
                self.balance_stats['smd_after'].abs() < 0.1
            ).sum()
        
        return summary


def main():
    """Demo propensity score matching with sample data."""
    print("=" * 70)
    print("PROPENSITY SCORE MATCHING DEMO")
    print("=" * 70)
    
    # Generate sample data
    np.random.seed(42)
    n = 1000
    
    # Covariates
    age = np.random.normal(45, 15, n)
    income = np.random.lognormal(10, 0.5, n)
    education = np.random.choice([1, 2, 3, 4], n, p=[0.2, 0.3, 0.3, 0.2])
    
    # Treatment assignment (correlated with covariates)
    propensity = 1 / (1 + np.exp(-(0.05 * age + 0.00001 * income + 0.3 * education - 5)))
    treatment = (np.random.random(n) < propensity).astype(int)
    
    # Outcome (treatment effect = 100, confounded by covariates)
    outcome = 500 + 0.5 * age + 0.001 * income + 50 * education + 100 * treatment + np.random.normal(0, 50, n)
    
    data = pd.DataFrame({
        'age': age,
        'income': income,
        'education': education,
        'treatment': treatment,
        'outcome': outcome
    })
    
    print(f"\nSample data: {len(data)} observations")
    print(f"Treated: {treatment.sum()}, Control: {(1-treatment).sum()}")
    
    # Run propensity matching
    matcher = PropensityMatcher(
        data=data,
        treatment_col='treatment',
        covariates=['age', 'income', 'education'],
        outcome_col='outcome'
    )
    
    matcher.fit()
    matcher.match(caliper=0.1)
    matcher.calculate_balance()
    matcher.estimate_ate()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    summary = matcher.get_summary()
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    print("\n✓ Propensity Score Matching completed successfully!")


if __name__ == '__main__':
    main()
