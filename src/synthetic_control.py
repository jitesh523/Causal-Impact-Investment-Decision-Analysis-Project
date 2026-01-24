"""
Synthetic Control Method Implementation
Alternative Causal Inference Approach

This module implements the Synthetic Control Method (SCM) as an alternative
to BSTS for estimating causal effects. SCM constructs a weighted combination
of control units to create a "synthetic" version of the treated unit.

Reference: Abadie, A., Diamond, A., & Hainmueller, J. (2010)
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')


class SyntheticControl:
    """Synthetic Control Method for Causal Inference"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.weights = None
        self.synthetic_outcome = None
        self.results = None
        
    def prepare_data(self, treated_series: pd.Series, control_matrix: pd.DataFrame):
        """
        Prepare data for synthetic control estimation
        
        Args:
            treated_series: Time series of treated unit outcomes
            control_matrix: DataFrame where each column is a potential control unit
        """
        self.treated = treated_series.values
        self.controls = control_matrix.values
        self.dates = treated_series.index
        
        # Find intervention index
        intervention_date = pd.to_datetime(self.config['dates']['intervention_date'])
        self.intervention_idx = np.searchsorted(self.dates, intervention_date)
        
        # Split into pre and post periods
        self.pre_treated = self.treated[:self.intervention_idx]
        self.pre_controls = self.controls[:self.intervention_idx]
        self.post_treated = self.treated[self.intervention_idx:]
        self.post_controls = self.controls[self.intervention_idx:]
        
        print(f"Pre-intervention periods: {len(self.pre_treated)}")
        print(f"Post-intervention periods: {len(self.post_treated)}")
        print(f"Number of control units: {self.controls.shape[1]}")
        
        return self
    
    def _objective(self, weights: np.ndarray) -> float:
        """
        Objective function: minimize pre-intervention RMSE
        between treated and synthetic control
        """
        synthetic = self.pre_controls @ weights
        rmse = np.sqrt(np.mean((self.pre_treated - synthetic) ** 2))
        return rmse
    
    def _constraint_sum(self, weights: np.ndarray) -> float:
        """Weights must sum to 1"""
        return np.sum(weights) - 1.0
    
    def fit(self) -> 'SyntheticControl':
        """
        Estimate optimal weights for synthetic control
        Uses constrained optimization to find weights that minimize
        pre-intervention prediction error
        """
        print("\n" + "=" * 80)
        print("FITTING SYNTHETIC CONTROL MODEL")
        print("=" * 80)
        
        n_controls = self.controls.shape[1]
        
        # Initial weights (equal)
        w0 = np.ones(n_controls) / n_controls
        
        # Constraints: weights sum to 1, weights >= 0
        constraints = {'type': 'eq', 'fun': self._constraint_sum}
        bounds = [(0, 1) for _ in range(n_controls)]
        
        # Optimize
        result = minimize(
            self._objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )
        
        if not result.success:
            print(f"⚠️ Optimization warning: {result.message}")
        
        self.weights = result.x
        
        # Calculate synthetic control for entire period
        self.synthetic_outcome = self.controls @ self.weights
        
        # Pre-intervention fit statistics
        pre_synthetic = self.pre_controls @ self.weights
        pre_rmse = np.sqrt(np.mean((self.pre_treated - pre_synthetic) ** 2))
        pre_mape = np.mean(np.abs((self.pre_treated - pre_synthetic) / self.pre_treated)) * 100
        
        print(f"\n✓ Model fitted successfully!")
        print(f"  Pre-intervention RMSE: {pre_rmse:.4f}")
        print(f"  Pre-intervention MAPE: {pre_mape:.2f}%")
        print(f"\nOptimal Weights (non-zero):")
        for i, w in enumerate(self.weights):
            if w > 0.01:
                print(f"  Control {i}: {w:.4f}")
        
        return self
    
    def estimate_effect(self) -> dict:
        """Calculate causal effect estimates"""
        print("\n" + "=" * 80)
        print("ESTIMATING CAUSAL EFFECT")
        print("=" * 80)
        
        # Post-intervention effects
        post_synthetic = self.post_controls @ self.weights
        effects = self.post_treated - post_synthetic
        
        # Cumulative effect
        cumulative_effect = np.sum(effects)
        avg_effect = np.mean(effects)
        
        # Relative effect
        rel_effect_pct = (cumulative_effect / np.sum(post_synthetic)) * 100
        
        # Simple significance test using pre-period placebo
        # (in-place permutation test)
        pre_synthetic = self.pre_controls @ self.weights
        pre_errors = self.pre_treated - pre_synthetic
        pre_std = np.std(pre_errors)
        
        # Pseudo p-value (how many SDs is the effect from 0)
        z_score = avg_effect / pre_std if pre_std > 0 else 0
        
        self.results = {
            'segment': 'synthetic_control',
            'average_effect': avg_effect,
            'cumulative_effect': cumulative_effect,
            'cumulative_effect_pct': rel_effect_pct,
            'pre_period_rmse': np.sqrt(np.mean(pre_errors ** 2)),
            'post_period_rmse': np.sqrt(np.mean(effects ** 2)),
            'z_score': z_score,
            'method': 'Synthetic Control'
        }
        
        print(f"\n📊 RESULTS:")
        print(f"  Average daily effect: ${avg_effect:,.2f}")
        print(f"  Cumulative effect: ${cumulative_effect:,.2f}")
        print(f"  Relative effect: {rel_effect_pct:.2f}%")
        print(f"  Z-score: {z_score:.2f}")
        
        if abs(z_score) > 2:
            print("\n✅ Effect appears statistically significant (|z| > 2)")
        else:
            print("\n⚠️ Effect may not be statistically significant (|z| <= 2)")
        
        return self.results
    
    def plot_results(self, save_path: str = None):
        """Generate visualization of synthetic control results"""
        print("\nGenerating Synthetic Control visualization...")
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Plot 1: Treated vs Synthetic
        ax1 = axes[0]
        ax1.plot(self.dates, self.treated, 'o-', color='#2E86AB', 
                label='Observed (Treated)', linewidth=2, markersize=3)
        ax1.plot(self.dates, self.synthetic_outcome, '--', color='#E63946',
                label='Synthetic Control', linewidth=2)
        
        intervention_date = self.dates[self.intervention_idx]
        ax1.axvline(intervention_date, color='gray', linestyle=':', linewidth=2.5,
                   label='Intervention', alpha=0.7)
        ax1.fill_between(self.dates[self.intervention_idx:], 
                        self.treated[self.intervention_idx:],
                        self.synthetic_outcome[self.intervention_idx:],
                        alpha=0.3, color='#06A77D', label='Causal Effect')
        
        ax1.set_ylabel('Revenue ($)', fontsize=12, fontweight='bold')
        ax1.set_title('Synthetic Control Method: Observed vs. Synthetic', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Gap (Treatment Effect)
        ax2 = axes[1]
        gap = self.treated - self.synthetic_outcome
        ax2.plot(self.dates, gap, 'o-', color='#06A77D', linewidth=2, markersize=3)
        ax2.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax2.axvline(intervention_date, color='gray', linestyle=':', linewidth=2.5, alpha=0.7)
        ax2.fill_between(self.dates, 0, gap, 
                        where=(gap > 0), color='#06A77D', alpha=0.3)
        ax2.fill_between(self.dates, 0, gap, 
                        where=(gap < 0), color='#E63946', alpha=0.3)
        
        ax2.set_ylabel('Treatment Effect ($)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.set_title('Gap: Observed - Synthetic (Causal Effect)', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot: {save_path}")
        
        plt.close()
        return fig


def main():
    """Run Synthetic Control analysis"""
    from data_pipeline import DataPipeline
    
    print("=" * 80)
    print("SYNTHETIC CONTROL METHOD")
    print("Alternative Causal Inference Approach")
    print("=" * 80)
    
    # Load data
    pipeline = DataPipeline('config.yaml')
    pipeline.load_data().clean_data().create_time_series()
    
    # Get time series data
    ts_data = pipeline.time_series_data
    
    # Prepare treated and control series
    # Treated: revenue from treatment group
    # Control: we'll use multiple lagged versions or segments as controls
    
    treated = ts_data[ts_data['treatment_exposed'] == 1].groupby('date')['revenue_usd'].sum()
    
    # For demonstration, create synthetic controls from:
    # 1. Control group revenue
    # 2. Impressions (scaled)
    # 3. Clicks (scaled)
    control_rev = ts_data[ts_data['treatment_exposed'] == 0].groupby('date')['revenue_usd'].sum()
    control_imp = ts_data[ts_data['treatment_exposed'] == 0].groupby('date')['impressions'].sum() / 100
    control_clk = ts_data[ts_data['treatment_exposed'] == 0].groupby('date')['clicks'].sum() * 10
    
    # Align indices
    common_idx = treated.index.intersection(control_rev.index)
    treated = treated.loc[common_idx]
    
    control_matrix = pd.DataFrame({
        'control_revenue': control_rev.loc[common_idx],
        'control_impressions': control_imp.loc[common_idx],
        'control_clicks': control_clk.loc[common_idx]
    })
    
    # Run Synthetic Control
    sc = SyntheticControl('config.yaml')
    sc.prepare_data(treated, control_matrix)
    sc.fit()
    results = sc.estimate_effect()
    
    # Generate plot
    output_dir = Path('reports/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    sc.plot_results(save_path=output_dir / 'synthetic_control_analysis.png')
    
    # Compare with BSTS results
    print("\n" + "=" * 80)
    print("COMPARISON WITH BSTS METHOD")
    print("=" * 80)
    
    bsts_metrics_path = Path('data/processed/impact_metrics.csv')
    if bsts_metrics_path.exists():
        bsts_df = pd.read_csv(bsts_metrics_path)
        bsts_agg = bsts_df[bsts_df['segment'] == 'aggregated'].iloc[0]
        
        print(f"\n{'Metric':<25} {'BSTS':<20} {'Synthetic Control':<20}")
        print("-" * 65)
        print(f"{'Cumulative Effect':<25} ${bsts_agg['cumulative_effect']:>15,.2f} ${results['cumulative_effect']:>15,.2f}")
        print(f"{'Relative Effect (%)':<25} {bsts_agg['cumulative_effect_pct']:>15.2f}% {results['cumulative_effect_pct']:>15.2f}%")
    
    print("\n✅ Synthetic Control analysis completed!")
    
    return sc, results


if __name__ == '__main__':
    main()
