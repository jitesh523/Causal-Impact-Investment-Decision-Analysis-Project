"""
Sensitivity Analysis Module
Test Robustness of Causal Impact Results

This module tests how sensitive the causal impact estimates are to:
1. Changes in intervention date
2. Changes in model parameters
3. Different pre/post period lengths
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import sys

sys.path.append(str(Path(__file__).parent))

from data_pipeline import DataPipeline
from causal_analysis import CausalAnalyzer

sns.set_style('whitegrid')


class SensitivityAnalyzer:
    """Analyze sensitivity of causal impact results to parameter changes"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.pipeline = None
        self.results = []
        
    def load_data(self):
        """Load and prepare data"""
        print("Loading data...")
        self.pipeline = DataPipeline('config.yaml')
        self.pipeline.load_data().clean_data()
        return self
    
    def sensitivity_intervention_date(self, date_offsets: list = None):
        """
        Test sensitivity to intervention date changes
        
        Args:
            date_offsets: List of day offsets from actual intervention date
        """
        if date_offsets is None:
            date_offsets = [-14, -7, 0, 7, 14]  # Two weeks before/after
        
        print("\n" + "=" * 80)
        print("SENSITIVITY ANALYSIS: Intervention Date")
        print("=" * 80)
        
        base_date = pd.to_datetime(self.config['dates']['intervention_date'])
        results = []
        
        for offset in date_offsets:
            test_date = base_date + pd.Timedelta(days=offset)
            test_date_str = test_date.strftime('%Y-%m-%d')
            
            print(f"\nTesting intervention date: {test_date_str} (offset: {offset:+d} days)")
            
            try:
                # Create time series with modified date
                self.pipeline.create_time_series(intervention_date=test_date_str)
                analysis_data = self.pipeline.get_analysis_series(metric='revenue_usd')
                
                # Check if we have valid pre and post periods
                pre_start, pre_end = analysis_data['pre_period']
                post_start, post_end = analysis_data['post_period']
                
                if pre_end - pre_start < 10 or post_end - post_start < 10:
                    print(f"  Skipping: Insufficient data points")
                    continue
                
                # Run analysis
                analyzer = CausalAnalyzer(analysis_data, config=self.config)
                analyzer.run_causal_impact()
                metrics = analyzer.get_impact_metrics()
                
                results.append({
                    'parameter': 'intervention_date',
                    'value': test_date_str,
                    'offset_days': offset,
                    'cumulative_effect': metrics['cumulative_effect'],
                    'p_value': metrics['p_value'],
                    'is_significant': metrics['p_value'] < 0.05
                })
                
                print(f"  Effect: ${metrics['cumulative_effect']:,.2f}, p={metrics['p_value']:.4f}")
                
            except Exception as e:
                print(f"  Error: {str(e)}")
        
        self.results.extend(results)
        return results
    
    def sensitivity_alpha(self, alpha_values: list = None):
        """
        Test sensitivity to significance level (alpha)
        
        Args:
            alpha_values: List of alpha values to test
        """
        if alpha_values is None:
            alpha_values = [0.01, 0.05, 0.10]
        
        print("\n" + "=" * 80)
        print("SENSITIVITY ANALYSIS: Significance Level (Alpha)")
        print("=" * 80)
        
        # Reset to original intervention date
        self.pipeline.create_time_series()
        analysis_data = self.pipeline.get_analysis_series(metric='revenue_usd')
        
        results = []
        
        for alpha in alpha_values:
            print(f"\nTesting alpha = {alpha}")
            
            # Run analysis with different alpha
            config_copy = self.config.copy()
            config_copy['model'] = config_copy.get('model', {})
            config_copy['model']['alpha'] = alpha
            
            analyzer = CausalAnalyzer(analysis_data, config=config_copy)
            analyzer.run_causal_impact()
            metrics = analyzer.get_impact_metrics()
            
            results.append({
                'parameter': 'alpha',
                'value': alpha,
                'cumulative_effect': metrics['cumulative_effect'],
                'p_value': metrics['p_value'],
                'is_significant': metrics['p_value'] < alpha
            })
            
            sig_status = "✅ Significant" if metrics['p_value'] < alpha else "⚠️ Not Significant"
            print(f"  Effect: ${metrics['cumulative_effect']:,.2f}, p={metrics['p_value']:.4f} ({sig_status})")
        
        self.results.extend(results)
        return results
    
    def sensitivity_metric(self, metrics: list = None):
        """
        Test results across different outcome metrics
        
        Args:
            metrics: List of metrics to analyze
        """
        if metrics is None:
            metrics = ['revenue_usd', 'conversion', 'clicks']
        
        print("\n" + "=" * 80)
        print("SENSITIVITY ANALYSIS: Different Metrics")
        print("=" * 80)
        
        self.pipeline.create_time_series()
        results = []
        
        for metric in metrics:
            print(f"\nTesting metric: {metric}")
            
            try:
                analysis_data = self.pipeline.get_analysis_series(metric=metric)
                
                analyzer = CausalAnalyzer(analysis_data, config=self.config)
                analyzer.run_causal_impact()
                impact_metrics = analyzer.get_impact_metrics()
                
                results.append({
                    'parameter': 'metric',
                    'value': metric,
                    'cumulative_effect': impact_metrics['cumulative_effect'],
                    'cumulative_effect_pct': impact_metrics['cumulative_effect_pct'],
                    'p_value': impact_metrics['p_value'],
                    'is_significant': impact_metrics['p_value'] < 0.05
                })
                
                print(f"  Effect: {impact_metrics['cumulative_effect']:,.2f} ({impact_metrics['cumulative_effect_pct']:.2f}%)")
                print(f"  p-value: {impact_metrics['p_value']:.4f}")
                
            except Exception as e:
                print(f"  Error: {str(e)}")
        
        self.results.extend(results)
        return results
    
    def plot_sensitivity(self, save_path: str = None):
        """Generate sensitivity analysis visualization"""
        if not self.results:
            print("No results to plot. Run sensitivity analyses first.")
            return
        
        print("\nGenerating sensitivity plots...")
        
        # Filter results by parameter type
        date_results = [r for r in self.results if r['parameter'] == 'intervention_date']
        
        if date_results:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            offsets = [r['offset_days'] for r in date_results]
            effects = [r['cumulative_effect'] for r in date_results]
            colors = ['#06A77D' if r['is_significant'] else '#E63946' for r in date_results]
            
            bars = ax.bar(offsets, effects, color=colors, edgecolor='black', alpha=0.8)
            
            ax.axhline(0, color='black', linestyle='-', linewidth=1)
            ax.axvline(0, color='gray', linestyle='--', linewidth=2, label='Actual Intervention')
            
            ax.set_xlabel('Days from Actual Intervention', fontsize=12, fontweight='bold')
            ax.set_ylabel('Cumulative Effect ($)', fontsize=12, fontweight='bold')
            ax.set_title('Sensitivity to Intervention Date', fontsize=14, fontweight='bold')
            
            # Legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#06A77D', edgecolor='black', label='Significant (p<0.05)'),
                Patch(facecolor='#E63946', edgecolor='black', label='Not Significant')
            ]
            ax.legend(handles=legend_elements, loc='best')
            
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ Saved plot: {save_path}")
            
            plt.close()
        
        return fig
    
    def export_results(self, output_path: str = None):
        """Export sensitivity analysis results to CSV"""
        if not self.results:
            print("No results to export.")
            return
        
        if output_path is None:
            output_path = Path(self.config['data']['processed_dir']) / 'sensitivity_results.csv'
        
        df = pd.DataFrame(self.results)
        df.to_csv(output_path, index=False)
        print(f"✓ Exported results: {output_path}")
        
        return df


def main():
    """Run comprehensive sensitivity analysis"""
    print("=" * 80)
    print("SENSITIVITY ANALYSIS")
    print("Testing Robustness of Causal Impact Estimates")
    print("=" * 80)
    
    analyzer = SensitivityAnalyzer('config.yaml')
    analyzer.load_data()
    
    # Run sensitivity tests
    analyzer.sensitivity_intervention_date()
    analyzer.sensitivity_alpha()
    analyzer.sensitivity_metric()
    
    # Generate plots
    output_dir = Path('reports/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    analyzer.plot_sensitivity(save_path=output_dir / 'sensitivity_analysis.png')
    
    # Export results
    analyzer.export_results()
    
    # Summary
    print("\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 80)
    
    df = pd.DataFrame(analyzer.results)
    
    # Date sensitivity
    date_results = df[df['parameter'] == 'intervention_date']
    if not date_results.empty:
        print(f"\nIntervention Date Sensitivity:")
        print(f"  Effect range: ${date_results['cumulative_effect'].min():,.2f} to ${date_results['cumulative_effect'].max():,.2f}")
        print(f"  Significant in {date_results['is_significant'].sum()}/{len(date_results)} tests")
    
    # Metric sensitivity
    metric_results = df[df['parameter'] == 'metric']
    if not metric_results.empty:
        print(f"\nMetric Sensitivity:")
        for _, row in metric_results.iterrows():
            sig = "✅" if row['is_significant'] else "⚠️"
            print(f"  {row['value']}: {row['cumulative_effect']:,.2f} ({row['cumulative_effect_pct']:.1f}%) {sig}")
    
    print("\n✅ Sensitivity analysis completed!")
    
    return analyzer


if __name__ == '__main__':
    main()
