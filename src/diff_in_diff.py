"""
Difference-in-Differences (DiD) Module
=======================================

Implements the Difference-in-Differences estimator for causal inference
with panel data or repeated cross-sections. Includes parallel trends testing,
event study analysis, and heterogeneous treatment effects.

Author: Causal Impact Analysis Project
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class DiDResults:
    """Container for DiD estimation results."""
    ate: float
    std_error: float
    t_statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    r_squared: float
    n_obs: int
    model_summary: str


class DifferenceInDifferences:
    """
    Difference-in-Differences estimator for causal inference.
    
    The DiD approach estimates causal effects by comparing the change in outcomes
    over time between a treatment group and a control group. Key assumption:
    parallel trends (both groups would have evolved similarly absent treatment).
    
    Example:
        >>> did = DifferenceInDifferences(
        ...     data=df,
        ...     outcome='revenue',
        ...     treatment='treated',
        ...     time='post_period',
        ...     unit_id='store_id'
        ... )
        >>> results = did.estimate()
        >>> did.plot_parallel_trends()
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        time: str,
        unit_id: Optional[str] = None,
        time_var: Optional[str] = None,
        covariates: Optional[List[str]] = None
    ):
        """
        Initialize DiD estimator.
        
        Args:
            data: Panel or repeated cross-section data
            outcome: Name of outcome variable column
            treatment: Name of treatment group indicator (0/1)
            time: Name of post-treatment period indicator (0/1)
            unit_id: Optional unit identifier for panel data (enables fixed effects)
            time_var: Optional continuous time variable for event study
            covariates: Optional list of control variables
        """
        self.data = data.copy()
        self.outcome = outcome
        self.treatment = treatment
        self.time = time
        self.unit_id = unit_id
        self.time_var = time_var
        self.covariates = covariates or []
        
        self.results = None
        self.model = None
        self.event_study_results = None
        
        self._validate_inputs()
    
    def _validate_inputs(self):
        """Validate input data and columns."""
        required_cols = [self.outcome, self.treatment, self.time]
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in data")
        
        # Check treatment is binary
        if not set(self.data[self.treatment].unique()).issubset({0, 1}):
            raise ValueError("Treatment column must be binary (0/1)")
        
        # Check time is binary
        if not set(self.data[self.time].unique()).issubset({0, 1}):
            raise ValueError("Time period column must be binary (0/1)")
    
    def estimate(self, robust_se: bool = True) -> DiDResults:
        """
        Estimate the DiD treatment effect using OLS.
        
        The model estimated is:
        Y = β₀ + β₁*Treatment + β₂*Post + β₃*(Treatment×Post) + ε
        
        where β₃ is the DiD estimator (ATT).
        
        Args:
            robust_se: Use heteroskedasticity-robust standard errors
        
        Returns:
            DiDResults object containing estimation results
        """
        print("\n" + "=" * 70)
        print("DIFFERENCE-IN-DIFFERENCES ESTIMATION")
        print("=" * 70)
        
        # Create interaction term
        self.data['did_interaction'] = (
            self.data[self.treatment] * self.data[self.time]
        )
        
        # Build formula
        formula = f"{self.outcome} ~ {self.treatment} + {self.time} + did_interaction"
        
        # Add covariates if specified
        if self.covariates:
            formula += " + " + " + ".join(self.covariates)
        
        # Fit model
        if self.unit_id and self.unit_id in self.data.columns:
            # Panel data with unit fixed effects
            formula += f" + C({self.unit_id})"
            print("Using unit fixed effects...")
        
        print(f"Formula: {formula}")
        
        self.model = smf.ols(formula, data=self.data).fit(
            cov_type='HC3' if robust_se else 'nonrobust'
        )
        
        # Extract DiD coefficient
        ate = self.model.params['did_interaction']
        std_error = self.model.bse['did_interaction']
        t_stat = self.model.tvalues['did_interaction']
        p_value = self.model.pvalues['did_interaction']
        
        # Confidence interval
        ci = self.model.conf_int().loc['did_interaction']
        ci_lower, ci_upper = ci[0], ci[1]
        
        self.results = DiDResults(
            ate=ate,
            std_error=std_error,
            t_statistic=t_stat,
            p_value=p_value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            r_squared=self.model.rsquared,
            n_obs=int(self.model.nobs),
            model_summary=str(self.model.summary())
        )
        
        # Print results
        print("\n" + "-" * 50)
        print("RESULTS")
        print("-" * 50)
        print(f"DiD Estimate (ATT): {ate:.4f}")
        print(f"Standard Error: {std_error:.4f}")
        print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        print(f"R²: {self.model.rsquared:.4f}")
        
        if p_value < 0.05:
            print("\n✓ Effect is statistically significant at α=0.05")
        else:
            print("\n⚠ Effect is NOT statistically significant at α=0.05")
        
        return self.results
    
    def test_parallel_trends(
        self,
        time_periods: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Test the parallel trends assumption.
        
        Uses pre-treatment data to test whether treatment and control groups
        evolved similarly before the intervention.
        
        Args:
            time_periods: Optional list of pre-treatment time periods
        
        Returns:
            Dictionary with test results
        """
        print("\n" + "=" * 70)
        print("PARALLEL TRENDS TEST")
        print("=" * 70)
        
        # Get pre-treatment data
        pre_data = self.data[self.data[self.time] == 0].copy()
        
        if self.time_var is None:
            print("⚠ No time variable specified. Using simple pre-period comparison.")
            
            # Simple test: compare pre-treatment trends
            treated_mean = pre_data[pre_data[self.treatment] == 1][self.outcome].mean()
            control_mean = pre_data[pre_data[self.treatment] == 0][self.outcome].mean()
            
            t_stat, p_value = stats.ttest_ind(
                pre_data[pre_data[self.treatment] == 1][self.outcome],
                pre_data[pre_data[self.treatment] == 0][self.outcome]
            )
            
            results = {
                'test_type': 'simple_comparison',
                'treated_mean': treated_mean,
                'control_mean': control_mean,
                'difference': treated_mean - control_mean,
                't_statistic': t_stat,
                'p_value': p_value,
                'parallel_trends_valid': p_value > 0.1  # Not rejecting null
            }
        else:
            # Regression-based test with time trends
            pre_data['trend_interaction'] = (
                pre_data[self.treatment] * pre_data[self.time_var]
            )
            
            formula = f"{self.outcome} ~ {self.treatment} + {self.time_var} + trend_interaction"
            model = smf.ols(formula, data=pre_data).fit()
            
            results = {
                'test_type': 'trend_comparison',
                'interaction_coef': model.params['trend_interaction'],
                'interaction_se': model.bse['trend_interaction'],
                'p_value': model.pvalues['trend_interaction'],
                'parallel_trends_valid': model.pvalues['trend_interaction'] > 0.1
            }
        
        if results['parallel_trends_valid']:
            print("✓ Parallel trends assumption appears VALID")
            print("  (No statistically significant difference in pre-trends)")
        else:
            print("⚠ WARNING: Parallel trends assumption may be VIOLATED")
            print("  (Treatment and control groups had different pre-trends)")
        
        print(f"\np-value: {results['p_value']:.4f}")
        
        return results
    
    def event_study(
        self,
        time_var: str,
        reference_period: int = -1
    ) -> pd.DataFrame:
        """
        Perform event study analysis.
        
        Estimates treatment effects for each time period relative to treatment,
        allowing visualization of dynamic treatment effects.
        
        Args:
            time_var: Name of relative time period variable
            reference_period: Which period to use as reference (default: -1)
        
        Returns:
            DataFrame with period-specific treatment effects
        """
        print("\n" + "=" * 70)
        print("EVENT STUDY ANALYSIS")
        print("=" * 70)
        
        if time_var not in self.data.columns:
            raise ValueError(f"Time variable '{time_var}' not found")
        
        # Get unique periods
        periods = sorted(self.data[time_var].unique())
        print(f"Periods: {periods}")
        print(f"Reference period: {reference_period}")
        
        # Create dummies for each period × treatment interaction
        event_data = self.data.copy()
        
        results_list = []
        
        for period in periods:
            if period == reference_period:
                continue
            
            # Create interaction for this period
            event_data[f'period_{period}'] = (
                (event_data[time_var] == period).astype(int) *
                event_data[self.treatment]
            )
        
        # Build formula with all period interactions
        period_terms = [f'period_{p}' for p in periods if p != reference_period]
        formula = f"{self.outcome} ~ {self.treatment} + C({time_var}) + " + " + ".join(period_terms)
        
        model = smf.ols(formula, data=event_data).fit(cov_type='HC3')
        
        for period in periods:
            if period == reference_period:
                results_list.append({
                    'period': period,
                    'coefficient': 0.0,
                    'std_error': 0.0,
                    'ci_lower': 0.0,
                    'ci_upper': 0.0,
                    'p_value': 1.0
                })
            else:
                var_name = f'period_{period}'
                coef = model.params[var_name]
                se = model.bse[var_name]
                ci = model.conf_int().loc[var_name]
                
                results_list.append({
                    'period': period,
                    'coefficient': coef,
                    'std_error': se,
                    'ci_lower': ci[0],
                    'ci_upper': ci[1],
                    'p_value': model.pvalues[var_name]
                })
        
        self.event_study_results = pd.DataFrame(results_list)
        
        print("\nEvent Study Coefficients:")
        print(self.event_study_results.to_string(index=False))
        
        return self.event_study_results
    
    def heterogeneous_effects(
        self,
        by: str
    ) -> pd.DataFrame:
        """
        Estimate heterogeneous treatment effects by subgroup.
        
        Args:
            by: Column name to stratify by
        
        Returns:
            DataFrame with subgroup-specific treatment effects
        """
        print(f"\n" + "=" * 70)
        print(f"HETEROGENEOUS EFFECTS BY: {by}")
        print("=" * 70)
        
        if by not in self.data.columns:
            raise ValueError(f"Stratification variable '{by}' not found")
        
        results_list = []
        
        for group in self.data[by].unique():
            subset = self.data[self.data[by] == group].copy()
            
            # Create interaction
            subset['did_interaction'] = subset[self.treatment] * subset[self.time]
            
            formula = f"{self.outcome} ~ {self.treatment} + {self.time} + did_interaction"
            model = smf.ols(formula, data=subset).fit(cov_type='HC3')
            
            results_list.append({
                'group': group,
                'ate': model.params['did_interaction'],
                'std_error': model.bse['did_interaction'],
                'p_value': model.pvalues['did_interaction'],
                'n_obs': int(model.nobs)
            })
        
        het_results = pd.DataFrame(results_list)
        print(het_results.to_string(index=False))
        
        return het_results
    
    def plot_parallel_trends(self, save_path: Optional[str] = None):
        """
        Visualize parallel trends and treatment effect.
        
        Args:
            save_path: Optional path to save figure
        """
        if self.time_var is None:
            # Simple 2-period plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Calculate means by group and period
            means = self.data.groupby([self.treatment, self.time])[self.outcome].mean()
            
            treated_pre = means.get((1, 0), np.nan)
            treated_post = means.get((1, 1), np.nan)
            control_pre = means.get((0, 0), np.nan)
            control_post = means.get((0, 1), np.nan)
            
            ax.plot([0, 1], [control_pre, control_post], 'o-', 
                   color='#F18F01', linewidth=2, markersize=10, label='Control')
            ax.plot([0, 1], [treated_pre, treated_post], 'o-',
                   color='#2E86AB', linewidth=2, markersize=10, label='Treated')
            
            # Add counterfactual
            counterfactual = treated_pre + (control_post - control_pre)
            ax.plot([1], [counterfactual], 's', color='#2E86AB', 
                   markersize=10, alpha=0.5, label='Counterfactual')
            ax.plot([0, 1], [treated_pre, counterfactual], '--',
                   color='#2E86AB', alpha=0.5, linewidth=2)
            
            # Annotate DiD effect
            if self.results:
                mid_y = (treated_post + counterfactual) / 2
                ax.annotate('', xy=(1.05, treated_post), xytext=(1.05, counterfactual),
                           arrowprops=dict(arrowstyle='<->', color='red', lw=2))
                ax.text(1.1, mid_y, f'DiD = {self.results.ate:.2f}',
                       fontsize=12, fontweight='bold', color='red', va='center')
            
            ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=2, 
                      label='Treatment', alpha=0.7)
            
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Pre-Treatment', 'Post-Treatment'])
            ax.set_xlabel('Period', fontsize=12, fontweight='bold')
            ax.set_ylabel(self.outcome, fontsize=12, fontweight='bold')
            ax.set_title('Difference-in-Differences Analysis', 
                        fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
        else:
            # Multi-period plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            means = self.data.groupby([self.treatment, self.time_var])[self.outcome].mean().reset_index()
            
            control = means[means[self.treatment] == 0]
            treated = means[means[self.treatment] == 1]
            
            ax.plot(control[self.time_var], control[self.outcome], 'o-',
                   color='#F18F01', linewidth=2, markersize=8, label='Control')
            ax.plot(treated[self.time_var], treated[self.outcome], 'o-',
                   color='#2E86AB', linewidth=2, markersize=8, label='Treated')
            
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2,
                      label='Treatment Start', alpha=0.7)
            
            ax.set_xlabel('Time Period', fontsize=12, fontweight='bold')
            ax.set_ylabel(self.outcome, fontsize=12, fontweight='bold')
            ax.set_title('Parallel Trends Visualization', fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot: {save_path}")
        
        plt.close()
        return fig
    
    def plot_event_study(self, save_path: Optional[str] = None):
        """
        Plot event study coefficients.
        
        Args:
            save_path: Optional path to save figure
        """
        if self.event_study_results is None:
            raise ValueError("Must run event_study() first")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        periods = self.event_study_results['period']
        coefs = self.event_study_results['coefficient']
        ci_lower = self.event_study_results['ci_lower']
        ci_upper = self.event_study_results['ci_upper']
        
        # Plot coefficients
        ax.scatter(periods, coefs, color='#2E86AB', s=100, zorder=3)
        ax.vlines(periods, ci_lower, ci_upper, color='#2E86AB', linewidth=2, zorder=2)
        ax.plot(periods, coefs, '-', color='#2E86AB', linewidth=1, alpha=0.5)
        
        # Reference lines
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7,
                  label='Treatment Start')
        
        # Shade pre/post periods
        ax.axvspan(ax.get_xlim()[0], 0, alpha=0.1, color='gray', label='Pre-Treatment')
        ax.axvspan(0, ax.get_xlim()[1], alpha=0.1, color='green', label='Post-Treatment')
        
        ax.set_xlabel('Period Relative to Treatment', fontsize=12, fontweight='bold')
        ax.set_ylabel('Treatment Effect', fontsize=12, fontweight='bold')
        ax.set_title('Event Study: Dynamic Treatment Effects', 
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
        """Get comprehensive summary of DiD analysis."""
        if self.results is None:
            raise ValueError("Must run estimate() first")
        
        return {
            'ate': self.results.ate,
            'std_error': self.results.std_error,
            'p_value': self.results.p_value,
            'ci_lower': self.results.ci_lower,
            'ci_upper': self.results.ci_upper,
            'r_squared': self.results.r_squared,
            'n_obs': self.results.n_obs,
            'significant': self.results.p_value < 0.05
        }


def main():
    """Demo DiD estimation with sample data."""
    print("=" * 70)
    print("DIFFERENCE-IN-DIFFERENCES DEMO")
    print("=" * 70)
    
    # Generate sample panel data
    np.random.seed(42)
    n_units = 200
    
    # Create units
    units = []
    for i in range(n_units):
        treated = 1 if i < n_units // 2 else 0
        base_outcome = np.random.normal(100, 20)
        
        # Pre-period
        units.append({
            'unit_id': i,
            'treatment': treated,
            'post': 0,
            'outcome': base_outcome + np.random.normal(0, 10)
        })
        
        # Post-period (treatment effect = 25 for treated)
        effect = 25 if treated else 0
        units.append({
            'unit_id': i,
            'treatment': treated,
            'post': 1,
            'outcome': base_outcome + 5 + effect + np.random.normal(0, 10)  # time effect = 5
        })
    
    data = pd.DataFrame(units)
    
    print(f"\nSample data: {len(data)} observations, {n_units} units")
    print(f"Treated units: {n_units // 2}, Control units: {n_units // 2}")
    
    # Run DiD
    did = DifferenceInDifferences(
        data=data,
        outcome='outcome',
        treatment='treatment',
        time='post',
        unit_id='unit_id'
    )
    
    results = did.estimate()
    did.test_parallel_trends()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    summary = did.get_summary()
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    print("\n✓ Difference-in-Differences completed successfully!")


if __name__ == '__main__':
    main()
