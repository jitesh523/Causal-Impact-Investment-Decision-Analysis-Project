"""
Bayesian A/B Testing Module
Alternative hypothesis testing using Bayesian inference

Computes posterior probability that treatment is better than control
using Beta-Binomial conjugate prior model.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml

sns.set_style('whitegrid')


class BayesianABTest:
    """Bayesian A/B Testing for conversion analysis"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.results = None
        
    def run_test(
        self,
        treatment_successes: int,
        treatment_trials: int,
        control_successes: int,
        control_trials: int,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        n_samples: int = 100000
    ) -> dict:
        """
        Run Bayesian A/B test
        
        Args:
            treatment_successes: Number of conversions in treatment
            treatment_trials: Total users in treatment
            control_successes: Number of conversions in control
            control_trials: Total users in control
            prior_alpha: Beta prior alpha (default: 1 for uniform)
            prior_beta: Beta prior beta (default: 1 for uniform)
            n_samples: Monte Carlo samples for probability estimation
            
        Returns:
            Dictionary with test results
        """
        print("\n" + "=" * 80)
        print("BAYESIAN A/B TEST")
        print("=" * 80)
        
        print(f"\nTreatment: {treatment_successes}/{treatment_trials} conversions")
        print(f"Control: {control_successes}/{control_trials} conversions")
        
        # Posterior parameters (Beta-Binomial conjugate)
        treatment_alpha = prior_alpha + treatment_successes
        treatment_beta = prior_beta + treatment_trials - treatment_successes
        
        control_alpha = prior_alpha + control_successes
        control_beta = prior_beta + control_trials - control_successes
        
        # Posterior distributions
        treatment_dist = stats.beta(treatment_alpha, treatment_beta)
        control_dist = stats.beta(control_alpha, control_beta)
        
        # Point estimates
        treatment_mean = treatment_alpha / (treatment_alpha + treatment_beta)
        control_mean = control_alpha / (control_alpha + control_beta)
        
        # Credible intervals (95%)
        treatment_ci = treatment_dist.ppf([0.025, 0.975])
        control_ci = control_dist.ppf([0.025, 0.975])
        
        # Monte Carlo sampling for P(treatment > control)
        treatment_samples = treatment_dist.rvs(n_samples)
        control_samples = control_dist.rvs(n_samples)
        
        prob_treatment_better = np.mean(treatment_samples > control_samples)
        
        # Expected lift
        lift_samples = (treatment_samples - control_samples) / control_samples
        expected_lift = np.mean(lift_samples)
        lift_ci = np.percentile(lift_samples, [2.5, 97.5])
        
        self.results = {
            'treatment_conversions': treatment_successes,
            'treatment_trials': treatment_trials,
            'treatment_rate': treatment_successes / treatment_trials,
            'treatment_posterior_mean': treatment_mean,
            'treatment_ci_lower': treatment_ci[0],
            'treatment_ci_upper': treatment_ci[1],
            'control_conversions': control_successes,
            'control_trials': control_trials,
            'control_rate': control_successes / control_trials,
            'control_posterior_mean': control_mean,
            'control_ci_lower': control_ci[0],
            'control_ci_upper': control_ci[1],
            'prob_treatment_better': prob_treatment_better,
            'expected_lift': expected_lift,
            'lift_ci_lower': lift_ci[0],
            'lift_ci_upper': lift_ci[1]
        }
        
        # Store distributions for plotting
        self._treatment_dist = treatment_dist
        self._control_dist = control_dist
        
        print(f"\n📊 RESULTS:")
        print(f"  Treatment conversion rate: {treatment_mean:.4f} ({treatment_ci[0]:.4f}, {treatment_ci[1]:.4f})")
        print(f"  Control conversion rate: {control_mean:.4f} ({control_ci[0]:.4f}, {control_ci[1]:.4f})")
        print(f"\n  P(Treatment > Control): {prob_treatment_better:.4f}")
        print(f"  Expected Lift: {expected_lift*100:.2f}% ({lift_ci[0]*100:.2f}%, {lift_ci[1]*100:.2f}%)")
        
        if prob_treatment_better > 0.95:
            print("\n✅ Strong evidence: Treatment is better (P > 0.95)")
        elif prob_treatment_better > 0.90:
            print("\n📈 Moderate evidence: Treatment is likely better (P > 0.90)")
        elif prob_treatment_better > 0.50:
            print("\n⚠️ Weak evidence: Slight favor for treatment")
        else:
            print("\n❌ No evidence: Control may be better")
        
        return self.results
    
    def plot_posteriors(self, save_path: str = None):
        """Plot posterior distributions"""
        if self.results is None:
            print("Run test first!")
            return None
        
        print("\nGenerating posterior distribution plot...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.linspace(0, 1, 1000)
        
        # Plot posteriors
        ax.plot(x, self._treatment_dist.pdf(x), 'b-', lw=2, label='Treatment')
        ax.fill_between(x, self._treatment_dist.pdf(x), alpha=0.3, color='blue')
        
        ax.plot(x, self._control_dist.pdf(x), 'r-', lw=2, label='Control')
        ax.fill_between(x, self._control_dist.pdf(x), alpha=0.3, color='red')
        
        ax.axvline(self.results['treatment_posterior_mean'], color='blue', 
                   linestyle='--', alpha=0.7, label=f"Treatment mean: {self.results['treatment_posterior_mean']:.4f}")
        ax.axvline(self.results['control_posterior_mean'], color='red', 
                   linestyle='--', alpha=0.7, label=f"Control mean: {self.results['control_posterior_mean']:.4f}")
        
        ax.set_xlabel('Conversion Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_title(f'Bayesian A/B Test: P(Treatment > Control) = {self.results["prob_treatment_better"]:.4f}',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot: {save_path}")
        
        plt.close()
        return fig


def main():
    """Run Bayesian A/B test on project data"""
    from data_pipeline import DataPipeline
    
    print("=" * 80)
    print("BAYESIAN A/B TESTING")
    print("=" * 80)
    
    # Load data
    pipeline = DataPipeline('config.yaml')
    pipeline.load_data().clean_data()
    
    df = pipeline.cleaned_data
    
    # Get treatment and control stats
    treatment = df[df['treatment_exposed'] == 1]
    control = df[df['treatment_exposed'] == 0]
    
    treatment_conversions = treatment['conversion'].sum()
    treatment_trials = len(treatment)
    
    control_conversions = control['conversion'].sum()
    control_trials = len(control)
    
    # Run Bayesian test
    ab_test = BayesianABTest('config.yaml')
    results = ab_test.run_test(
        treatment_successes=int(treatment_conversions),
        treatment_trials=int(treatment_trials),
        control_successes=int(control_conversions),
        control_trials=int(control_trials)
    )
    
    # Plot
    output_dir = Path('reports/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    ab_test.plot_posteriors(save_path=output_dir / 'bayesian_ab_test.png')
    
    # Export
    results_df = pd.DataFrame([results])
    results_df.to_csv('data/processed/bayesian_ab_results.csv', index=False)
    print(f"\n✓ Exported results: data/processed/bayesian_ab_results.csv")
    
    print("\n✅ Bayesian A/B testing completed!")
    
    return ab_test, results


if __name__ == '__main__':
    main()
