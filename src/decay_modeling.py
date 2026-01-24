"""
Long-Term Effects / Decay Modeling
Estimate how campaign effects diminish over time

Models exponential decay to calculate:
- Effect half-life
- Decay rate
- Long-term cumulative impact
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import sys

sys.path.append(str(Path(__file__).parent))

from data_pipeline import DataPipeline
from causal_analysis import CausalAnalyzer

sns.set_style('whitegrid')


def exponential_decay(t: np.ndarray, amplitude: float, decay_rate: float, baseline: float) -> np.ndarray:
    """Exponential decay function: A * exp(-λt) + baseline"""
    return amplitude * np.exp(-decay_rate * t) + baseline


class DecayModeler:
    """Model the decay of campaign effects over time"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.effects = None
        self.dates = None
        self.fit_params = None
        self.results = None
        
    def load_effects(self):
        """Load causal effects from analysis"""
        print("Loading causal impact effects...")
        
        pipeline = DataPipeline('config.yaml')
        pipeline.load_data().clean_data().create_time_series()
        
        analysis_data = pipeline.get_analysis_series(metric='revenue_usd')
        
        analyzer = CausalAnalyzer(analysis_data, config=self.config)
        analyzer.run_causal_impact()
        
        # Get post-intervention effects
        post_start = analyzer.post_period[0]
        post_end = analyzer.post_period[1]
        
        self.effects = analyzer.predictions['point_effect'][post_start:post_end + 1]
        self.dates = analyzer.dates[post_start:post_end + 1]
        
        # Days since intervention
        self.days = np.arange(len(self.effects))
        
        print(f"✓ Loaded {len(self.effects)} post-intervention data points")
        
        return self
    
    def fit_decay_model(self):
        """Fit exponential decay model to effects"""
        print("\n" + "=" * 80)
        print("FITTING DECAY MODEL")
        print("=" * 80)
        
        if self.effects is None:
            raise ValueError("Load effects first!")
        
        # Initial guesses
        amplitude_guess = np.max(self.effects) - np.min(self.effects)
        decay_rate_guess = 0.05
        baseline_guess = np.mean(self.effects[-10:]) if len(self.effects) > 10 else 0
        
        try:
            # Fit the model
            popt, pcov = curve_fit(
                exponential_decay,
                self.days,
                self.effects,
                p0=[amplitude_guess, decay_rate_guess, baseline_guess],
                bounds=([0, 0, -np.inf], [np.inf, 1, np.inf]),
                maxfev=5000
            )
            
            self.fit_params = {
                'amplitude': popt[0],
                'decay_rate': popt[1],
                'baseline': popt[2]
            }
            
            # Calculate metrics
            half_life = np.log(2) / popt[1] if popt[1] > 0 else np.inf
            
            # Predicted values
            predicted = exponential_decay(self.days, *popt)
            
            # R-squared
            ss_res = np.sum((self.effects - predicted) ** 2)
            ss_tot = np.sum((self.effects - np.mean(self.effects)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Long-term projection (365 days)
            future_days = np.arange(365)
            future_effects = exponential_decay(future_days, *popt)
            long_term_cumulative = np.sum(future_effects)
            
            self.results = {
                'amplitude': popt[0],
                'decay_rate': popt[1],
                'baseline': popt[2],
                'half_life_days': half_life,
                'r_squared': r_squared,
                'long_term_cumulative_365d': long_term_cumulative,
                'observed_cumulative': np.sum(self.effects)
            }
            
            print(f"\n📊 DECAY MODEL RESULTS:")
            print(f"  Initial Effect Amplitude: ${popt[0]:,.2f}")
            print(f"  Decay Rate (λ): {popt[1]:.4f} per day")
            print(f"  Baseline Effect: ${popt[2]:,.2f}")
            print(f"  Half-Life: {half_life:.1f} days")
            print(f"  Model R²: {r_squared:.4f}")
            print(f"\n  Observed Cumulative (to date): ${np.sum(self.effects):,.2f}")
            print(f"  Projected 1-Year Cumulative: ${long_term_cumulative:,.2f}")
            
        except Exception as e:
            print(f"⚠️ Could not fit decay model: {str(e)}")
            print("Effects may not follow exponential decay pattern")
            
            self.results = {
                'error': str(e),
                'observed_cumulative': np.sum(self.effects)
            }
        
        return self
    
    def plot_decay(self, save_path: str = None):
        """Plot decay model fit"""
        if self.effects is None:
            print("Load effects first!")
            return None
        
        print("\nGenerating decay model visualization...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Effects over time with fitted curve
        ax1 = axes[0]
        ax1.plot(self.days, self.effects, 'o', color='#2E86AB', alpha=0.6, label='Observed Effect')
        
        if self.fit_params:
            # Fitted curve
            x_smooth = np.linspace(0, len(self.days), 100)
            y_smooth = exponential_decay(x_smooth, **self.fit_params)
            ax1.plot(x_smooth, y_smooth, '-', color='#E63946', lw=2, label='Fitted Decay')
            
            # Half-life line
            half_life = self.results['half_life_days']
            if half_life < len(self.days):
                ax1.axvline(half_life, color='gray', linestyle='--', alpha=0.7, 
                           label=f'Half-life: {half_life:.1f} days')
        
        ax1.axhline(0, color='black', linewidth=1)
        ax1.set_xlabel('Days Since Intervention', fontweight='bold')
        ax1.set_ylabel('Daily Effect ($)', fontweight='bold')
        ax1.set_title('Campaign Effect Decay Over Time', fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative effect projection
        ax2 = axes[1]
        
        # Observed cumulative
        observed_cum = np.cumsum(self.effects)
        ax2.plot(self.days, observed_cum, 'o-', color='#2E86AB', lw=2, label='Observed Cumulative')
        
        if self.fit_params:
            # Projected cumulative (365 days)
            future_days = np.arange(365)
            future_effects = exponential_decay(future_days, **self.fit_params)
            projected_cum = np.cumsum(future_effects)
            
            ax2.plot(future_days, projected_cum, '--', color='#06A77D', lw=2, 
                    label='1-Year Projection', alpha=0.7)
            ax2.axvline(len(self.days), color='gray', linestyle=':', alpha=0.7)
        
        ax2.set_xlabel('Days Since Intervention', fontweight='bold')
        ax2.set_ylabel('Cumulative Effect ($)', fontweight='bold')
        ax2.set_title('Cumulative Impact Projection', fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot: {save_path}")
        
        plt.close()
        return fig
    
    def export_results(self, output_path: str = None):
        """Export decay analysis results"""
        if self.results is None:
            print("Fit model first!")
            return None
        
        if output_path is None:
            output_path = Path(self.config['data']['processed_dir']) / 'decay_analysis.csv'
        
        df = pd.DataFrame([self.results])
        df.to_csv(output_path, index=False)
        print(f"✓ Exported results: {output_path}")
        
        return df


def main():
    """Run decay analysis"""
    print("=" * 80)
    print("LONG-TERM EFFECT DECAY MODELING")
    print("=" * 80)
    
    modeler = DecayModeler('config.yaml')
    modeler.load_effects()
    modeler.fit_decay_model()
    
    # Plot
    output_dir = Path('reports/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    modeler.plot_decay(save_path=output_dir / 'decay_analysis.png')
    
    # Export
    modeler.export_results()
    
    print("\n✅ Decay modeling completed!")
    
    return modeler


if __name__ == '__main__':
    main()
