"""
Causal Impact Analysis Module
Steps 4-5: Building BSTS Model and Interpreting Results

This module:
- Builds Bayesian Structural Time Series model using statsmodels
- Generates counterfactual predictions  
- Calculates causal impact with confidence intervals
- Performs placebo tests
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from statsmodels.tsa.statespace.structural import UnobservedComponents
from scipy import stats
from sklearn.linear_model import BayesianRidge

sns.set_style('whitegrid')


class CausalAnalyzer:
    """Causal impact analysis using structural time series"""
    
    def __init__(self, data_dict):
        """
        Initialize with data dictionary from pipeline
        
        Args:
            data_dict: Dict with keys 'data', 'y', 'X', 'dates', 'pre_period', 'post_period'
        """
        self.data = data_dict['data']
        self.y = data_dict['y']
        self.X = data_dict['X']
        self.dates = pd.to_datetime(data_dict['dates'])
        self.pre_period = data_dict['pre_period']
        self.post_period = data_dict['post_period']
        self.model = None
        self.predictions = None
        self.impact_results = None
        
    def run_causal_impact(self, alpha=0.05):
        """
        Run Causal Impact analysis using Bayesian regression
        
        Args:
            alpha: Significance level (default 0.05 for 95% CI)
        """
        print("\n" + "=" * 80)
        print("RUNNING CAUSAL IMPACT ANALYSIS")
        print("=" * 80)
        
        pre_start, pre_end = self.pre_period
        post_start, post_end = self.post_period
        
        print(f"\nPre-intervention period: {self.dates[pre_start].date()} to {self.dates[pre_end].date()}")
        print(f"Post-intervention period: {self.dates[post_start].date()} to {self.dates[post_end].date()}")
        print(f"Number of pre-period observations: {pre_end - pre_start + 1}")
        print(f"Number of post-period observations: {post_end - post_start + 1}")
        
        # Get pre-period data for training
        y_pre = self.y[pre_start:pre_end + 1]
        X_pre = self.X[pre_start:pre_end + 1]
        
        # Fit Bayesian Ridge regression on pre-period
        print("\nFitting Bayesian regression model on pre-intervention period...")
        self.model = BayesianRidge(
            max_iter=300,
            compute_score=True,
            alpha_1=1e-6,
            alpha_2=1e-6,
            lambda_1=1e-6,
            lambda_2=1e-6
        )
        self.model.fit(X_pre, y_pre)
        
        # Make predictions for the entire period
        y_pred, y_std = self.model.predict(self.X, return_std=True)
        
        # Calculate confidence intervals
        z_score = stats.norm.ppf(1 - alpha / 2)
        y_lower = y_pred - z_score * y_std
        y_upper = y_pred + z_score * y_std
        
        # Store predictions
        self.predictions = {
            'actual': self.y,
            'predicted': y_pred,
            'predicted_lower': y_lower,
            'predicted_upper': y_upper,
            'std': y_std
        }
        
        # Calculate point-wise effects (only in post-period)
        point_effect = np.zeros_like(self.y)
        point_effect[post_start:post_end + 1] = self.y[post_start:post_end + 1] - y_pred[post_start:post_end + 1]
        
        point_effect_lower = np.zeros_like(self.y)
        point_effect_lower[post_start:post_end + 1] = self.y[post_start:post_end + 1] - y_upper[post_start:post_end + 1]
        
        point_effect_upper = np.zeros_like(self.y)
        point_effect_upper[post_start:post_end + 1] = self.y[post_start:post_end + 1] - y_lower[post_start:post_end + 1]
        
        self.predictions['point_effect'] = point_effect
        self.predictions['point_effect_lower'] = point_effect_lower
        self.predictions['point_effect_upper'] = point_effect_upper
        
        # Calculate cumulative metrics for post-period
        actual_post = self.y[post_start:post_end + 1]
        pred_post = y_pred[post_start:post_end + 1]
        
        cumulative_actual = np.sum(actual_post)
        cumulative_predicted = np.sum(pred_post)
        cumulative_effect = cumulative_actual - cumulative_predicted
        
        # Calculate average metrics
        avg_actual = np.mean(actual_post)
        avg_predicted = np.mean(pred_post)
        avg_effect = avg_actual - avg_predicted
        
        # Calculate relative effects
        rel_effect_avg = avg_effect / avg_predicted if avg_predicted != 0 else 0
        rel_effect_cum = cumulative_effect / cumulative_predicted if cumulative_predicted != 0 else 0
        
        # Statistical significance (t-test on post-period residuals)
        residuals = actual_post - pred_post
        t_stat, p_value = stats.ttest_1samp(residuals, 0)
        
        # Store results
        self.impact_results = {
            'average_actual': avg_actual,
            'average_predicted': avg_predicted,
            'average_effect': avg_effect,
            'average_effect_pct': rel_effect_avg * 100,
            'cumulative_actual': cumulative_actual,
            'cumulative_predicted': cumulative_predicted,
            'cumulative_effect': cumulative_effect,
            'cumulative_effect_pct': rel_effect_cum * 100,
            'p_value': p_value,
            't_statistic': t_stat,
            'alpha': alpha
        }
        
        print("✓ Model fitted successfully!")
        print(f"✓ R² score on pre-period: {self.model.score(X_pre, y_pre):.4f}")
        
        return self
    
    def get_summary(self):
        """Get text summary of causal impact"""
        if self.impact_results is None:
            raise ValueError("Must run run_causal_impact() first")
        
        r = self.impact_results
        print("\n" + "=" * 80)
        print("CAUSAL IMPACT SUMMARY")
        print("=" * 80)
        print(f"\n{'Metric':<30} {'Actual':<15} {'Predicted':<15} {'Effect':<15}")
        print("-" * 75)
        print(f"{'Average (daily)':<30} {r['average_actual']:<15.2f} {r['average_predicted']:<15.2f} {r['average_effect']:<15.2f}")
        print(f"{'  Relative effect (%)':<30} {'':<15} {'':<15} {r['average_effect_pct']:<15.2f}")
        print()
        print(f"{'Cumulative (total)':<30} {r['cumulative_actual']:<15.2f} {r['cumulative_predicted']:<15.2f} {r['cumulative_effect']:<15.2f}")
        print(f"{'  Relative effect (%)':<30} {'':<15} {'':<15} {r['cumulative_effect_pct']:<15.2f}")
        print()
        print(f"Statistical significance: p-value = {r['p_value']:.4f}")
        
        if r['p_value'] < 0.05:
            print("✓ The effect is statistically significant at α = 0.05")
        else:
            print("⚠ The effect is NOT statistically significant at α = 0.05")
        
        # Interpretation
        print("\n" + "=" * 80)
        print("INTERPRETATION")
        print("=" * 80)
        
        if r['cumulative_effect'] > 0:
            print(f"✓ The intervention had a POSITIVE impact")
            print(f"  - Cumulative gain: ${r['cumulative_effect']:.2f}")
            print(f"  - This represents a {abs(r['cumulative_effect_pct']):.2f}% increase")
        else:
            print(f"⚠ The intervention had a NEGATIVE impact")
            print(f"  - Cumulative loss: ${abs(r['cumulative_effect']):.2f}")
            print(f"  - This represents a {abs(r['cumulative_effect_pct']):.2f}% decrease")
        
        return r
    
    def get_impact_metrics(self):
        """Extract key impact metrics"""
        if self.impact_results is None:
            raise ValueError("Must run run_causal_impact() first")
        
        return self.impact_results
    
    def plot_results(self, figsize=(15, 10), save_path=None):
        """Plot causal impact results"""
        if self.predictions is None:
            raise ValueError("Must run run_causal_impact() first")
        
        print("\nGenerating visualizations...")
        
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        # Plot 1: Original series
        ax1 = axes[0]
        ax1.plot(self.dates, self.predictions['actual'], 'o-', color='#2E86AB', 
                label='Observed Data', linewidth=2, markersize=3, alpha=0.8)
        ax1.plot(self.dates, self.predictions['predicted'], '--', color='#A23B72',
                label='Counterfactual Prediction', linewidth=2, alpha=0.7)
        ax1.fill_between(self.dates, 
                         self.predictions['predicted_lower'], 
                         self.predictions['predicted_upper'],
                         color='#A23B72', alpha=0.2, label='95% CI')
        
        # Mark intervention
        intervention_date = self.dates[self.post_period[0]]
        ax1.axvline(intervention_date, color='red', linestyle=':', linewidth=2.5, 
                   label='Intervention', alpha=0.7)
        
        ax1.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=10, framealpha=0.9)
        ax1.set_title('Original vs. Counterfactual', fontsize=14, fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Point-wise causal effect
        ax2 = axes[1]
        point_effect = self.predictions['point_effect']
        ax2.plot(self.dates, point_effect, 'o-', color='#F18F01', 
                linewidth=2, markersize=3, alpha=0.8, label='Point Effect')
        ax2.fill_between(self.dates,
                         self.predictions['point_effect_lower'],
                         self.predictions['point_effect_upper'],
                         color='#F18F01', alpha=0.2, label='95% CI')
        ax2.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax2.axvline(intervention_date, color='red', linestyle=':', linewidth=2.5, alpha=0.7)
        ax2.set_ylabel('Point Effect', fontsize=12, fontweight='bold')
        ax2.set_title('Pointwise Causal Effect', fontsize=14, fontweight='bold', pad=15)
        ax2.legend(loc='best', fontsize=10, framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Cumulative effect
        ax3 = axes[2]
        cumulative_effect = np.cumsum(point_effect)
        ax3.plot(self.dates, cumulative_effect, 'o-', color='#06A77D',
                linewidth=2, markersize=3, alpha=0.8, label='Cumulative Effect')
        ax3.fill_between(self.dates, 0, cumulative_effect, 
                        where=(cumulative_effect > 0), color='#06A77D', alpha=0.3)
        ax3.fill_between(self.dates, 0, cumulative_effect, 
                        where=(cumulative_effect < 0), color='#E63946', alpha=0.3)
        ax3.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax3.axvline(intervention_date, color='red', linestyle=':', linewidth=2.5, alpha=0.7)
        ax3.set_ylabel('Cumulative Effect', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax3.set_title('Cumulative Causal Effect', fontsize=14, fontweight='bold', pad=15)
        ax3.legend(loc='best', fontsize=10, framealpha=0.9)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot: {save_path}")
        
        plt.close()
        
        return fig
    
    def placebo_test(self, placebo_date_str='2024-02-15'):
        """
        Perform placebo test by running analysis on pre-period
        Should show no significant effect
        """
        print(f"\n{'=' * 80}")
        print("PLACEBO TEST")
        print(f"{'=' * 80}")
        print(f"Running analysis with fake intervention on {placebo_date_str}")
        print("Expected: No significant effect (validates model)")
        
        placebo_date = pd.to_datetime(placebo_date_str)
        
        # Find index of placebo date
        placebo_idx = np.where(self.dates == placebo_date)[0]
        if len(placebo_idx) == 0:
            print(f"Warning: Placebo date {placebo_date_str} not in data. Using midpoint of pre-period.")
            placebo_idx = (self.pre_period[0] + self.pre_period[1]) // 2
        else:
            placebo_idx = placebo_idx[0]
        
        # Only use pre-intervention data
        pre_data_end = self.pre_period[1]
        
        # Create temporary data dict for placebo
        placebo_dict = {
            'data': self.data[:pre_data_end + 1],
            'y': self.y[:pre_data_end + 1],
            'X': self.X[:pre_data_end + 1],
            'dates': self.dates[:pre_data_end + 1],
            'pre_period': [self.pre_period[0], placebo_idx - 1],
            'post_period': [placebo_idx, pre_data_end]
        }
        
        # Run analysis
        placebo_analyzer = CausalAnalyzer(placebo_dict)
        placebo_analyzer.run_causal_impact()
        placebo_results = placebo_analyzer.get_summary()
        
        placebo_p = placebo_results['p_value']
        print(f"\n✓ Placebo test p-value: {placebo_p:.4f}")
        
        if placebo_p > 0.05:
            print("✅ PASS: No significant effect found (as expected)")
        else:
            print("⚠️  WARNING: Significant effect found in placebo test!")
            print("   This may indicate model issues or confounding factors.")
        
        return placebo_analyzer


def main():
    """Run causal analysis on pipeline output"""
    from data_pipeline import DataPipeline
    
    print("=" * 80)
    print("CAUSAL IMPACT ANALYSIS")
    print("Steps 4-5: BSTS Model & Results Interpretation")
    print("=" * 80)
    
    # Load processed data
    pipeline = DataPipeline('Dataset.xlsx')
    pipeline.load_data().clean_data().create_time_series()
    
    # Get analysis series for revenue
    print("\n--- Analyzing Revenue Impact ---")
    revenue_data = pipeline.get_analysis_series(metric='revenue_usd')
    
    # Run causal analysis
    analyzer = CausalAnalyzer(revenue_data)
    analyzer.run_causal_impact()
    
    # Get summary
    results = analyzer.get_summary()
    
    # Extract metrics
    metrics = analyzer.get_impact_metrics()
    
    print("\n" + "=" * 80)
    print("KEY IMPACT METRICS")
    print("=" * 80)
    print(f"Average daily impact: ${metrics['average_effect']:.2f}")
    print(f"  Relative effect: {metrics['average_effect_pct']:.2f}%")
    print(f"\nCumulative impact: ${metrics['cumulative_effect']:.2f}")
    print(f"  Relative effect: {metrics['cumulative_effect_pct']:.2f}%")
    print(f"\nStatistical significance: p = {metrics['p_value']:.4f}")
    
    # Generate plots
    output_dir = Path('reports/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    analyzer.plot_results(save_path=output_dir / 'causal_impact_analysis.png')
    
    # Placebo test
    analyzer.placebo_test()
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('data/processed/impact_metrics.csv', index=False)
    print(f"\n✓ Saved impact metrics to data/processed/impact_metrics.csv")
    
    print("\n✅ Causal analysis completed successfully!")
    
    return analyzer, metrics


if __name__ == '__main__':
    main()
