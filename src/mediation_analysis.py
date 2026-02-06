"""
Mediation Analysis Module
=========================

Implements mediation analysis to understand causal pathways.
Uses the Baron-Kenny approach and modern causal inference methods
to decompose total effects into direct and indirect components.

Author: Causal Impact Analysis Project
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class MediationResults:
    """Container for mediation analysis results."""
    total_effect: float
    direct_effect: float
    indirect_effect: float
    proportion_mediated: float
    sobel_statistic: float
    sobel_pvalue: float
    bootstrap_ci_lower: Optional[float]
    bootstrap_ci_upper: Optional[float]


class MediationAnalyzer:
    """
    Mediation Analysis for understanding causal pathways.
    
    Implements the Baron-Kenny approach to decompose total treatment effects
    into direct effects (X → Y) and indirect effects (X → M → Y).
    
    The causal pathway:
        Treatment (X) → Mediator (M) → Outcome (Y)
                   ↘_______________↗
                    (Direct effect)
    
    Example:
        >>> analyzer = MediationAnalyzer(
        ...     data=df,
        ...     treatment='marketing_spend',
        ...     mediator='brand_awareness',
        ...     outcome='sales'
        ... )
        >>> results = analyzer.analyze()
        >>> analyzer.plot_mediation_diagram()
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        treatment: str,
        mediator: str,
        outcome: str,
        covariates: Optional[List[str]] = None
    ):
        """
        Initialize MediationAnalyzer.
        
        Args:
            data: DataFrame containing all variables
            treatment: Name of treatment variable (X)
            mediator: Name of mediating variable (M)
            outcome: Name of outcome variable (Y)
            covariates: Optional list of control variables
        """
        self.data = data.copy()
        self.treatment = treatment
        self.mediator = mediator
        self.outcome = outcome
        self.covariates = covariates or []
        
        self.results = None
        self.models = {}
        
        self._validate_inputs()
    
    def _validate_inputs(self):
        """Validate input data and columns."""
        required = [self.treatment, self.mediator, self.outcome]
        for col in required:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in data")
        
        for cov in self.covariates:
            if cov not in self.data.columns:
                raise ValueError(f"Covariate '{cov}' not found in data")
    
    def _build_formula(self, outcome: str, predictors: List[str]) -> str:
        """Build statsmodels formula."""
        rhs = " + ".join(predictors)
        return f"{outcome} ~ {rhs}"
    
    def analyze(self, bootstrap_n: int = 1000) -> MediationResults:
        """
        Run complete mediation analysis.
        
        Implements the Baron-Kenny 4-step approach:
        1. Total effect: X → Y
        2. X → M relationship
        3. M → Y relationship (controlling for X)
        4. Direct effect: X → Y (controlling for M)
        
        Args:
            bootstrap_n: Number of bootstrap samples for CI
        
        Returns:
            MediationResults object
        """
        print("\n" + "=" * 70)
        print("MEDIATION ANALYSIS")
        print("=" * 70)
        print(f"Treatment: {self.treatment}")
        print(f"Mediator: {self.mediator}")
        print(f"Outcome: {self.outcome}")
        
        controls = self.covariates if self.covariates else []
        
        # Step 1: Total effect (c path)
        # Y = c*X + e
        print("\n--- Step 1: Total Effect (X → Y) ---")
        predictors_total = [self.treatment] + controls
        model_total = smf.ols(
            self._build_formula(self.outcome, predictors_total),
            data=self.data
        ).fit()
        
        total_effect = model_total.params[self.treatment]
        total_se = model_total.bse[self.treatment]
        total_pval = model_total.pvalues[self.treatment]
        
        self.models['total'] = model_total
        print(f"  c (total effect): {total_effect:.4f} (p={total_pval:.4f})")
        
        # Step 2: X → M relationship (a path)
        # M = a*X + e
        print("\n--- Step 2: Treatment → Mediator (a path) ---")
        predictors_a = [self.treatment] + controls
        model_a = smf.ols(
            self._build_formula(self.mediator, predictors_a),
            data=self.data
        ).fit()
        
        a_coef = model_a.params[self.treatment]
        a_se = model_a.bse[self.treatment]
        a_pval = model_a.pvalues[self.treatment]
        
        self.models['a_path'] = model_a
        print(f"  a (X → M): {a_coef:.4f} (p={a_pval:.4f})")
        
        # Step 3 & 4: M → Y and Direct effect (b and c' paths)
        # Y = c'*X + b*M + e
        print("\n--- Step 3 & 4: Full Model (X + M → Y) ---")
        predictors_full = [self.treatment, self.mediator] + controls
        model_full = smf.ols(
            self._build_formula(self.outcome, predictors_full),
            data=self.data
        ).fit()
        
        direct_effect = model_full.params[self.treatment]  # c'
        direct_se = model_full.bse[self.treatment]
        direct_pval = model_full.pvalues[self.treatment]
        
        b_coef = model_full.params[self.mediator]  # b
        b_se = model_full.bse[self.mediator]
        b_pval = model_full.pvalues[self.mediator]
        
        self.models['full'] = model_full
        print(f"  c' (direct effect): {direct_effect:.4f} (p={direct_pval:.4f})")
        print(f"  b (M → Y): {b_coef:.4f} (p={b_pval:.4f})")
        
        # Calculate indirect effect
        indirect_effect = a_coef * b_coef
        print(f"\n--- Indirect Effect (a × b) ---")
        print(f"  Indirect effect: {indirect_effect:.4f}")
        
        # Proportion mediated
        if abs(total_effect) > 0.001:
            proportion_mediated = indirect_effect / total_effect
        else:
            proportion_mediated = 0
        print(f"  Proportion mediated: {proportion_mediated:.2%}")
        
        # Sobel test for significance of indirect effect
        sobel_se = np.sqrt(a_coef**2 * b_se**2 + b_coef**2 * a_se**2)
        sobel_z = indirect_effect / sobel_se
        sobel_pval = 2 * (1 - stats.norm.cdf(abs(sobel_z)))
        
        print(f"\n--- Sobel Test ---")
        print(f"  Z-statistic: {sobel_z:.4f}")
        print(f"  p-value: {sobel_pval:.4f}")
        
        # Bootstrap confidence interval for indirect effect
        print(f"\n--- Bootstrap CI ({bootstrap_n} samples) ---")
        boot_indirect = self._bootstrap_indirect(n_samples=bootstrap_n)
        ci_lower = np.percentile(boot_indirect, 2.5)
        ci_upper = np.percentile(boot_indirect, 97.5)
        print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        self.results = MediationResults(
            total_effect=total_effect,
            direct_effect=direct_effect,
            indirect_effect=indirect_effect,
            proportion_mediated=proportion_mediated,
            sobel_statistic=sobel_z,
            sobel_pvalue=sobel_pval,
            bootstrap_ci_lower=ci_lower,
            bootstrap_ci_upper=ci_upper
        )
        
        # Summary
        print("\n" + "=" * 70)
        print("MEDIATION SUMMARY")
        print("=" * 70)
        print(f"Total Effect (c):     {total_effect:.4f}")
        print(f"Direct Effect (c'):   {direct_effect:.4f}")
        print(f"Indirect Effect (ab): {indirect_effect:.4f}")
        print(f"Proportion Mediated:  {proportion_mediated:.2%}")
        
        # Interpretation
        if sobel_pval < 0.05:
            if proportion_mediated > 0.8:
                print("\n✓ FULL MEDIATION: Most of the effect goes through the mediator")
            elif proportion_mediated > 0.2:
                print("\n✓ PARTIAL MEDIATION: Significant indirect effect exists")
            else:
                print("\n⚠ SMALL MEDIATION: Minor indirect effect")
        else:
            print("\n⚠ NO SIGNIFICANT MEDIATION: Indirect effect not significant")
        
        return self.results
    
    def _bootstrap_indirect(self, n_samples: int = 1000) -> np.ndarray:
        """Bootstrap the indirect effect for confidence intervals."""
        n = len(self.data)
        indirect_effects = []
        controls = self.covariates if self.covariates else []
        
        for _ in range(n_samples):
            # Resample with replacement
            sample = self.data.sample(n=n, replace=True)
            
            try:
                # Fit a path
                predictors_a = [self.treatment] + controls
                model_a = smf.ols(
                    self._build_formula(self.mediator, predictors_a),
                    data=sample
                ).fit()
                a = model_a.params[self.treatment]
                
                # Fit b path
                predictors_full = [self.treatment, self.mediator] + controls
                model_full = smf.ols(
                    self._build_formula(self.outcome, predictors_full),
                    data=sample
                ).fit()
                b = model_full.params[self.mediator]
                
                indirect_effects.append(a * b)
            except:
                pass
        
        return np.array(indirect_effects)
    
    def sensitivity_analysis(self, rho_values: Optional[List[float]] = None) -> pd.DataFrame:
        """
        Sensitivity analysis for unmeasured confounding.
        
        Tests how robust the indirect effect is to potential unmeasured
        confounders between M and Y.
        
        Args:
            rho_values: Correlation values to test (default: range)
        
        Returns:
            DataFrame with sensitivity results
        """
        print("\n" + "=" * 70)
        print("SENSITIVITY ANALYSIS FOR UNMEASURED CONFOUNDING")
        print("=" * 70)
        
        if rho_values is None:
            rho_values = np.arange(-0.5, 0.55, 0.1)
        
        # Get residual variances from models
        m_resid_var = self.models['a_path'].resid.var()
        y_resid_var = self.models['full'].resid.var()
        
        results_list = []
        
        for rho in rho_values:
            # Adjust indirect effect for potential confounding
            adjustment = rho * np.sqrt(m_resid_var * y_resid_var)
            adjusted_indirect = self.results.indirect_effect - adjustment
            
            # Recalculate proportion mediated
            if abs(self.results.total_effect) > 0.001:
                adj_proportion = adjusted_indirect / self.results.total_effect
            else:
                adj_proportion = 0
            
            results_list.append({
                'rho': rho,
                'adjusted_indirect': adjusted_indirect,
                'adjusted_proportion': adj_proportion,
                'still_positive': adjusted_indirect > 0
            })
        
        sensitivity_df = pd.DataFrame(results_list)
        print(sensitivity_df.to_string(index=False))
        
        return sensitivity_df
    
    def plot_mediation_diagram(self, save_path: Optional[str] = None):
        """
        Create visual diagram of mediation model.
        
        Args:
            save_path: Optional path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Positions
        x_pos = 0.15
        m_pos = 0.5
        y_pos = 0.85
        
        # Draw boxes
        box_props = dict(boxstyle='round,pad=0.5', facecolor='lightblue', 
                        edgecolor='navy', linewidth=2)
        
        ax.text(x_pos, 0.5, f"Treatment\n({self.treatment})", 
               fontsize=14, ha='center', va='center', bbox=box_props)
        ax.text(m_pos, 0.8, f"Mediator\n({self.mediator})",
               fontsize=14, ha='center', va='center', bbox=box_props)
        ax.text(y_pos, 0.5, f"Outcome\n({self.outcome})",
               fontsize=14, ha='center', va='center', bbox=box_props)
        
        # Draw arrows with coefficients
        arrow_props = dict(arrowstyle='->', color='navy', lw=2)
        
        if self.results:
            # a path (X → M)
            ax.annotate('', xy=(m_pos - 0.1, 0.75), xytext=(x_pos + 0.1, 0.55),
                       arrowprops=arrow_props)
            a_path = self.models['a_path'].params[self.treatment]
            ax.text((x_pos + m_pos) / 2 - 0.05, 0.7, f"a = {a_path:.3f}",
                   fontsize=12, fontweight='bold', color='darkgreen')
            
            # b path (M → Y)
            ax.annotate('', xy=(y_pos - 0.1, 0.55), xytext=(m_pos + 0.1, 0.75),
                       arrowprops=arrow_props)
            b_path = self.models['full'].params[self.mediator]
            ax.text((m_pos + y_pos) / 2 + 0.05, 0.7, f"b = {b_path:.3f}",
                   fontsize=12, fontweight='bold', color='darkgreen')
            
            # c' path (X → Y direct)
            ax.annotate('', xy=(y_pos - 0.1, 0.5), xytext=(x_pos + 0.1, 0.5),
                       arrowprops=arrow_props)
            ax.text(m_pos, 0.45, f"c' = {self.results.direct_effect:.3f}",
                   fontsize=12, fontweight='bold', color='darkblue', ha='center')
            
            # Add summary box
            summary_text = (
                f"Total Effect (c): {self.results.total_effect:.3f}\n"
                f"Direct Effect (c'): {self.results.direct_effect:.3f}\n"
                f"Indirect Effect (ab): {self.results.indirect_effect:.3f}\n"
                f"Proportion Mediated: {self.results.proportion_mediated:.1%}\n"
                f"Sobel p-value: {self.results.sobel_pvalue:.4f}"
            )
            ax.text(0.5, 0.15, summary_text, fontsize=11, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', 
                            edgecolor='orange', linewidth=2),
                   transform=ax.transAxes)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Mediation Model', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot: {save_path}")
        
        plt.close()
        return fig
    
    def plot_effects(self, save_path: Optional[str] = None):
        """
        Bar chart comparing direct, indirect, and total effects.
        
        Args:
            save_path: Optional path to save figure
        """
        if self.results is None:
            raise ValueError("Must run analyze() first")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        effects = ['Total Effect\n(c)', 'Direct Effect\n(c\')', 'Indirect Effect\n(a×b)']
        values = [
            self.results.total_effect,
            self.results.direct_effect,
            self.results.indirect_effect
        ]
        colors = ['#2E86AB', '#06A77D', '#F18F01']
        
        bars = ax.bar(effects, values, color=colors, edgecolor='black', linewidth=2)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Add bootstrap CI for indirect effect
        if self.results.bootstrap_ci_lower is not None:
            ax.errorbar(2, values[2], 
                       yerr=[[values[2] - self.results.bootstrap_ci_lower],
                             [self.results.bootstrap_ci_upper - values[2]]],
                       fmt='none', color='black', capsize=10, capthick=2)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_ylabel('Effect Size', fontsize=12, fontweight='bold')
        ax.set_title('Decomposition of Treatment Effect', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot: {save_path}")
        
        plt.close()
        return fig
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of mediation analysis."""
        if self.results is None:
            raise ValueError("Must run analyze() first")
        
        return {
            'total_effect': self.results.total_effect,
            'direct_effect': self.results.direct_effect,
            'indirect_effect': self.results.indirect_effect,
            'proportion_mediated': self.results.proportion_mediated,
            'sobel_z': self.results.sobel_statistic,
            'sobel_pvalue': self.results.sobel_pvalue,
            'ci_lower': self.results.bootstrap_ci_lower,
            'ci_upper': self.results.bootstrap_ci_upper,
            'significant_mediation': self.results.sobel_pvalue < 0.05
        }


def main():
    """Demo mediation analysis with sample data."""
    print("=" * 70)
    print("MEDIATION ANALYSIS DEMO")
    print("=" * 70)
    
    # Generate sample data
    np.random.seed(42)
    n = 500
    
    # Treatment variable
    treatment = np.random.normal(0, 1, n)
    
    # Mediator (affected by treatment)
    mediator = 0.6 * treatment + np.random.normal(0, 0.5, n)
    
    # Outcome (affected by both treatment and mediator)
    # Direct effect of treatment = 0.3
    # Effect through mediator = 0.6 * 0.5 = 0.3
    outcome = 0.3 * treatment + 0.5 * mediator + np.random.normal(0, 0.5, n)
    
    data = pd.DataFrame({
        'treatment': treatment,
        'mediator': mediator,
        'outcome': outcome
    })
    
    print(f"\nSample data: {n} observations")
    print("True effects:")
    print("  a (T → M): 0.6")
    print("  b (M → Y): 0.5")
    print("  c' (T → Y direct): 0.3")
    print("  True indirect (ab): 0.3")
    
    # Run mediation analysis
    analyzer = MediationAnalyzer(
        data=data,
        treatment='treatment',
        mediator='mediator',
        outcome='outcome'
    )
    
    results = analyzer.analyze(bootstrap_n=500)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    summary = analyzer.get_summary()
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    print("\n✓ Mediation Analysis completed successfully!")


if __name__ == '__main__':
    main()
