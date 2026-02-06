"""
Regression Discontinuity Design (RDD) Module
=============================================

Implements regression discontinuity design for causal inference when
treatment is assigned based on a threshold/cutoff. Includes sharp and
fuzzy RDD, bandwidth selection, and manipulation testing.

Author: Causal Impact Analysis Project
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class RDDResults:
    """Container for RDD estimation results."""
    treatment_effect: float
    std_error: float
    t_statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    bandwidth: float
    n_left: int
    n_right: int
    design_type: str


class RegressionDiscontinuity:
    """
    Regression Discontinuity Design estimator.
    
    RDD exploits situations where treatment is assigned based on whether
    a running variable exceeds a threshold. By comparing outcomes just
    above and just below the cutoff, we can estimate causal effects.
    
    Example:
        >>> rdd = RegressionDiscontinuity(
        ...     data=df,
        ...     outcome='test_score',
        ...     running_var='age',
        ...     cutoff=18,
        ...     bandwidth=2
        ... )
        >>> results = rdd.estimate()
        >>> rdd.plot_discontinuity()
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        outcome: str,
        running_var: str,
        cutoff: float,
        treatment: Optional[str] = None,
        bandwidth: Optional[float] = None,
        kernel: str = 'triangular'
    ):
        """
        Initialize RDD estimator.
        
        Args:
            data: DataFrame containing outcome and running variable
            outcome: Name of outcome variable column
            running_var: Name of running/forcing variable column
            cutoff: Treatment threshold value
            treatment: Optional treatment indicator (for fuzzy RDD)
            bandwidth: Bandwidth for local regression (None = auto-select)
            kernel: Kernel type ('triangular', 'uniform', 'epanechnikov')
        """
        self.data = data.copy()
        self.outcome = outcome
        self.running_var = running_var
        self.cutoff = cutoff
        self.treatment = treatment
        self.bandwidth = bandwidth
        self.kernel = kernel
        
        self.results = None
        self.design_type = 'fuzzy' if treatment else 'sharp'
        
        # Center running variable at cutoff
        self.data['running_centered'] = self.data[running_var] - cutoff
        self.data['above_cutoff'] = (self.data[running_var] >= cutoff).astype(int)
        
        self._validate_inputs()
    
    def _validate_inputs(self):
        """Validate input data."""
        if self.outcome not in self.data.columns:
            raise ValueError(f"Outcome '{self.outcome}' not found")
        if self.running_var not in self.data.columns:
            raise ValueError(f"Running variable '{self.running_var}' not found")
        
        # Check we have data on both sides of cutoff
        n_below = (self.data[self.running_var] < self.cutoff).sum()
        n_above = (self.data[self.running_var] >= self.cutoff).sum()
        
        if n_below < 10 or n_above < 10:
            raise ValueError(f"Insufficient data around cutoff (below: {n_below}, above: {n_above})")
    
    def _kernel_weights(self, x: np.ndarray, h: float) -> np.ndarray:
        """Calculate kernel weights for observations."""
        u = x / h
        
        if self.kernel == 'triangular':
            weights = np.maximum(0, 1 - np.abs(u))
        elif self.kernel == 'uniform':
            weights = (np.abs(u) <= 1).astype(float)
        elif self.kernel == 'epanechnikov':
            weights = np.maximum(0, 0.75 * (1 - u**2))
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
        
        return weights
    
    def select_bandwidth(self, method: str = 'ik') -> float:
        """
        Select optimal bandwidth using IK (Imbens-Kalyanaraman) method.
        
        Args:
            method: Bandwidth selection method ('ik' or 'rot')
        
        Returns:
            Optimal bandwidth
        """
        print("Selecting optimal bandwidth...")
        
        y = self.data[self.outcome].values
        x = self.data['running_centered'].values
        
        if method == 'rot':
            # Rule of thumb: 1.06 * std * n^(-1/5)
            h = 1.06 * np.std(x) * (len(x) ** (-0.2))
        else:
            # Simplified IK-style bandwidth
            # Based on minimizing MSE of local linear estimator
            
            # Pilot bandwidth
            h_pilot = 1.84 * np.std(x) * (len(x) ** (-0.2))
            
            # Estimate variance
            left_mask = (x < 0) & (x >= -h_pilot)
            right_mask = (x >= 0) & (x <= h_pilot)
            
            if left_mask.sum() > 5 and right_mask.sum() > 5:
                var_left = np.var(y[left_mask])
                var_right = np.var(y[right_mask])
                var_pooled = (var_left + var_right) / 2
            else:
                var_pooled = np.var(y)
            
            # Regularized curvature estimate
            h_curvature = 3.0 * np.std(x) * (len(x) ** (-0.2))
            
            # Use weighted local quadratic to estimate curvature
            def estimate_curvature(side_mask, h):
                subset_x = x[side_mask]
                subset_y = y[side_mask]
                if len(subset_x) < 10:
                    return 0.1
                weights = self._kernel_weights(subset_x, h)
                X = np.column_stack([np.ones_like(subset_x), subset_x, subset_x**2])
                try:
                    W = np.diag(weights)
                    beta = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ subset_y, rcond=None)[0]
                    return max(abs(beta[2]), 0.01)
                except:
                    return 0.1
            
            curv_left = estimate_curvature(x < 0, h_curvature)
            curv_right = estimate_curvature(x >= 0, h_curvature)
            curv = max(curv_left, curv_right)
            
            # IK formula (simplified)
            C_k = 3.4375  # Triangular kernel constant
            n = len(x)
            h = C_k * ((var_pooled / curv) ** 0.2) * (n ** (-0.2))
            
            # Bound bandwidth
            h = max(h, np.std(x) * 0.1)
            h = min(h, np.std(x) * 2.0)
        
        print(f"✓ Selected bandwidth: {h:.4f}")
        return h
    
    def estimate(self, polynomial_order: int = 1) -> RDDResults:
        """
        Estimate the treatment effect at the discontinuity.
        
        Uses local polynomial regression on each side of the cutoff.
        
        Args:
            polynomial_order: Order of local polynomial (1 = linear, 2 = quadratic)
        
        Returns:
            RDDResults object
        """
        print("\n" + "=" * 70)
        print(f"REGRESSION DISCONTINUITY DESIGN ({self.design_type.upper()})")
        print("=" * 70)
        print(f"Cutoff: {self.cutoff}")
        print(f"Polynomial order: {polynomial_order}")
        
        # Select bandwidth if not specified
        if self.bandwidth is None:
            self.bandwidth = self.select_bandwidth()
        else:
            print(f"Using specified bandwidth: {self.bandwidth:.4f}")
        
        h = self.bandwidth
        x = self.data['running_centered'].values
        y = self.data[self.outcome].values
        d = self.data['above_cutoff'].values
        
        # Filter to bandwidth
        in_bandwidth = np.abs(x) <= h
        x_bw = x[in_bandwidth]
        y_bw = y[in_bandwidth]
        d_bw = d[in_bandwidth]
        
        n_left = (x_bw < 0).sum()
        n_right = (x_bw >= 0).sum()
        
        print(f"Observations in bandwidth: {len(x_bw)} (left: {n_left}, right: {n_right})")
        
        # Calculate kernel weights
        weights = self._kernel_weights(x_bw, h)
        
        if self.design_type == 'sharp':
            # Sharp RDD: simple jump at cutoff
            # Model: Y = α + τ*D + β*X + γ*D*X + ε
            
            # Build design matrix
            X_left = x_bw * (1 - d_bw)  # Slope left of cutoff
            X_right = x_bw * d_bw  # Slope right of cutoff
            
            if polynomial_order == 1:
                X_design = np.column_stack([
                    np.ones_like(x_bw), d_bw, X_left, X_right
                ])
            else:
                X_design = np.column_stack([
                    np.ones_like(x_bw), d_bw, 
                    X_left, X_right,
                    (X_left ** 2), (X_right ** 2)
                ])
            
            # Weighted least squares
            W = np.diag(weights)
            XtWX = X_design.T @ W @ X_design
            XtWy = X_design.T @ W @ y_bw
            
            try:
                beta = np.linalg.solve(XtWX, XtWy)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(XtWX, XtWy, rcond=None)[0]
            
            tau = beta[1]  # Treatment effect is coefficient on D
            
            # Calculate standard errors
            residuals = y_bw - X_design @ beta
            sigma2 = np.sum(weights * residuals**2) / (np.sum(weights) - len(beta))
            
            try:
                var_beta = sigma2 * np.linalg.inv(XtWX)
                se_tau = np.sqrt(var_beta[1, 1])
            except:
                se_tau = np.std(residuals) / np.sqrt(len(residuals))
        
        else:
            # Fuzzy RDD: 2SLS with treatment probability jump at cutoff
            treatment_data = self.data[self.treatment].values[in_bandwidth]
            
            # First stage: T on D and controls
            X_first = np.column_stack([np.ones_like(x_bw), d_bw, x_bw])
            W = np.diag(weights)
            beta_first = np.linalg.lstsq(
                X_first.T @ W @ X_first,
                X_first.T @ W @ treatment_data,
                rcond=None
            )[0]
            T_hat = X_first @ beta_first
            
            # Second stage: Y on T_hat
            X_second = np.column_stack([np.ones_like(x_bw), T_hat, x_bw])
            beta_second = np.linalg.lstsq(
                X_second.T @ W @ X_second,
                X_second.T @ W @ y_bw,
                rcond=None
            )[0]
            
            tau = beta_second[1]  # Treatment effect
            
            # SE calculation (simplified)
            residuals = y_bw - X_second @ beta_second
            sigma2 = np.sum(weights * residuals**2) / (np.sum(weights) - 3)
            se_tau = np.sqrt(sigma2 / np.sum(weights * (T_hat - T_hat.mean())**2))
        
        # Inference
        t_stat = tau / se_tau
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(x_bw) - 4))
        ci_lower = tau - 1.96 * se_tau
        ci_upper = tau + 1.96 * se_tau
        
        self.results = RDDResults(
            treatment_effect=tau,
            std_error=se_tau,
            t_statistic=t_stat,
            p_value=p_value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            bandwidth=h,
            n_left=n_left,
            n_right=n_right,
            design_type=self.design_type
        )
        
        # Print results
        print("\n" + "-" * 50)
        print("RESULTS")
        print("-" * 50)
        print(f"Treatment Effect: {tau:.4f}")
        print(f"Standard Error: {se_tau:.4f}")
        print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print("\n✓ Effect is statistically significant at α=0.05")
        else:
            print("\n⚠ Effect is NOT statistically significant at α=0.05")
        
        return self.results
    
    def mccrary_test(self, bins: int = 50) -> Dict[str, float]:
        """
        McCrary density test for manipulation at the cutoff.
        
        Tests whether there's bunching of observations just above/below
        the cutoff, which would suggest manipulation of the running variable.
        
        Args:
            bins: Number of bins for histogram
        
        Returns:
            Dictionary with test results
        """
        print("\n" + "=" * 70)
        print("MCCRARY DENSITY TEST")
        print("=" * 70)
        print("Testing for manipulation of running variable at cutoff...")
        
        x = self.data['running_centered'].values
        
        # Create histogram
        hist, bin_edges = np.histogram(x, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Find bins around cutoff
        left_bins = bin_centers < 0
        right_bins = bin_centers >= 0
        
        # Estimate density on each side near cutoff
        near_cutoff = np.abs(bin_centers) < np.std(x) / 2
        
        density_left = np.mean(hist[left_bins & near_cutoff])
        density_right = np.mean(hist[right_bins & near_cutoff])
        
        # Log difference in density
        if density_left > 0 and density_right > 0:
            log_diff = np.log(density_right) - np.log(density_left)
        else:
            log_diff = 0
        
        # Simple t-test for density difference
        left_counts = hist[left_bins & near_cutoff]
        right_counts = hist[right_bins & near_cutoff]
        
        if len(left_counts) > 1 and len(right_counts) > 1:
            t_stat, p_value = stats.ttest_ind(right_counts, left_counts)
        else:
            t_stat, p_value = 0, 1
        
        results = {
            'density_left': density_left,
            'density_right': density_right,
            'log_difference': log_diff,
            't_statistic': t_stat,
            'p_value': p_value,
            'manipulation_detected': p_value < 0.05
        }
        
        if results['manipulation_detected']:
            print("⚠ WARNING: Evidence of manipulation detected!")
            print("  The density is discontinuous at the cutoff.")
        else:
            print("✓ No evidence of manipulation")
            print("  Density appears continuous at the cutoff.")
        
        print(f"\np-value: {p_value:.4f}")
        
        return results
    
    def sensitivity_analysis(
        self,
        bandwidths: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Sensitivity analysis across different bandwidths.
        
        Args:
            bandwidths: List of bandwidths to test (None = auto-generate)
        
        Returns:
            DataFrame with results for each bandwidth
        """
        print("\n" + "=" * 70)
        print("BANDWIDTH SENSITIVITY ANALYSIS")
        print("=" * 70)
        
        if bandwidths is None:
            # Generate range around optimal bandwidth
            h_opt = self.bandwidth or self.select_bandwidth()
            bandwidths = [h_opt * m for m in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]]
        
        results_list = []
        original_bw = self.bandwidth
        
        for h in bandwidths:
            self.bandwidth = h
            result = self.estimate()
            
            results_list.append({
                'bandwidth': h,
                'treatment_effect': result.treatment_effect,
                'std_error': result.std_error,
                'p_value': result.p_value,
                'n_left': result.n_left,
                'n_right': result.n_right
            })
        
        self.bandwidth = original_bw
        
        sensitivity_df = pd.DataFrame(results_list)
        print("\nSensitivity Results:")
        print(sensitivity_df.to_string(index=False))
        
        return sensitivity_df
    
    def plot_discontinuity(
        self,
        nbins: int = 20,
        save_path: Optional[str] = None
    ):
        """
        Plot the regression discontinuity.
        
        Args:
            nbins: Number of bins for scatter plot
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        
        x = self.data['running_centered'].values
        y = self.data[self.outcome].values
        
        # Bin the data for cleaner visualization
        bin_edges = np.linspace(x.min(), x.max(), nbins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_means = np.zeros(nbins)
        
        for i in range(nbins):
            mask = (x >= bin_edges[i]) & (x < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_means[i] = y[mask].mean()
        
        # Separate left and right
        left_mask = bin_centers < 0
        right_mask = bin_centers >= 0
        
        # Plot binned means
        ax.scatter(bin_centers[left_mask], bin_means[left_mask], 
                  s=100, color='#2E86AB', alpha=0.8, label='Control', zorder=3)
        ax.scatter(bin_centers[right_mask], bin_means[right_mask],
                  s=100, color='#F18F01', alpha=0.8, label='Treatment', zorder=3)
        
        # Fit local linear regression on each side
        h = self.bandwidth or np.std(x) * 0.5
        
        for side, color, label in [('left', '#2E86AB', 'Left fit'), ('right', '#F18F01', 'Right fit')]:
            if side == 'left':
                mask = (x < 0) & (x >= -h)
            else:
                mask = (x >= 0) & (x <= h)
            
            if mask.sum() > 5:
                X_fit = sm.add_constant(x[mask])
                model = sm.OLS(y[mask], X_fit).fit()
                
                x_pred = np.linspace(x[mask].min(), x[mask].max(), 100)
                X_pred = sm.add_constant(x_pred)
                y_pred = model.predict(X_pred)
                
                ax.plot(x_pred, y_pred, '-', color=color, linewidth=2.5, label=label)
        
        # Mark cutoff
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7,
                  label='Cutoff')
        
        # Shade bandwidth region
        if self.bandwidth:
            ax.axvspan(-self.bandwidth, self.bandwidth, alpha=0.1, color='gray',
                      label=f'Bandwidth (h={self.bandwidth:.2f})')
        
        # Annotate treatment effect
        if self.results:
            ax.annotate(
                f'τ = {self.results.treatment_effect:.2f}',
                xy=(0.02, 0.95),
                xycoords='axes fraction',
                fontsize=14,
                fontweight='bold',
                color='red',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
        
        ax.set_xlabel(f'{self.running_var} (centered at cutoff)', 
                     fontsize=12, fontweight='bold')
        ax.set_ylabel(self.outcome, fontsize=12, fontweight='bold')
        ax.set_title('Regression Discontinuity Design', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot: {save_path}")
        
        plt.close()
        return fig
    
    def plot_density(self, save_path: Optional[str] = None):
        """
        Plot density of running variable around cutoff.
        
        Args:
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = self.data['running_centered'].values
        
        # Histogram with different colors on each side
        bins = 40
        
        left_data = x[x < 0]
        right_data = x[x >= 0]
        
        ax.hist(left_data, bins=bins//2, alpha=0.7, color='#2E86AB', 
               label='Below Cutoff', density=True)
        ax.hist(right_data, bins=bins//2, alpha=0.7, color='#F18F01',
               label='Above Cutoff', density=True)
        
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2,
                  label='Cutoff')
        
        ax.set_xlabel(f'{self.running_var} (centered at cutoff)',
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_title('Running Variable Density (McCrary Test)',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot: {save_path}")
        
        plt.close()
        return fig
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of RDD analysis."""
        if self.results is None:
            raise ValueError("Must run estimate() first")
        
        return {
            'design_type': self.results.design_type,
            'cutoff': self.cutoff,
            'bandwidth': self.results.bandwidth,
            'treatment_effect': self.results.treatment_effect,
            'std_error': self.results.std_error,
            'p_value': self.results.p_value,
            'ci_lower': self.results.ci_lower,
            'ci_upper': self.results.ci_upper,
            'n_left': self.results.n_left,
            'n_right': self.results.n_right,
            'significant': self.results.p_value < 0.05
        }


def main():
    """Demo RDD with sample data."""
    print("=" * 70)
    print("REGRESSION DISCONTINUITY DESIGN DEMO")
    print("=" * 70)
    
    # Generate sample data with discontinuity
    np.random.seed(42)
    n = 500
    
    # Running variable (e.g., test score)
    running = np.random.uniform(0, 100, n)
    cutoff = 50
    
    # Treatment assigned at cutoff
    treated = (running >= cutoff).astype(int)
    
    # Outcome with treatment effect of 10 at cutoff + noise
    # Also has continuous relationship with running variable
    outcome = (
        20 +  # Intercept
        0.5 * running +  # Slope
        10 * treated +  # Treatment effect
        np.random.normal(0, 5, n)  # Noise
    )
    
    data = pd.DataFrame({
        'score': running,
        'outcome': outcome,
        'treated': treated
    })
    
    print(f"\nSample data: {n} observations")
    print(f"Cutoff: {cutoff}")
    print(f"Treated: {treated.sum()}, Control: {(1-treated).sum()}")
    
    # Run RDD
    rdd = RegressionDiscontinuity(
        data=data,
        outcome='outcome',
        running_var='score',
        cutoff=cutoff
    )
    
    results = rdd.estimate()
    rdd.mccrary_test()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    summary = rdd.get_summary()
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    print("\n✓ Regression Discontinuity Design completed successfully!")


if __name__ == '__main__':
    main()
