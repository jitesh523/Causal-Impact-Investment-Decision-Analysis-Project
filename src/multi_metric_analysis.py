"""
Multi-Metric Analysis Module
Analyze causal impact across multiple KPIs

Supports analysis of:
- Revenue
- Conversions
- Clicks
- CTR
- User counts
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


class MultiMetricAnalyzer:
    """Analyze causal impact across multiple metrics"""
    
    SUPPORTED_METRICS = [
        'revenue_usd',
        'conversion',
        'clicks',
        'impressions',
        'n_users'
    ]
    
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.pipeline = None
        self.results = {}
        
    def load_data(self):
        """Load and prepare data"""
        print("Loading data...")
        self.pipeline = DataPipeline('config.yaml')
        self.pipeline.load_data().clean_data().create_time_series()
        return self
    
    def analyze_metrics(self, metrics: list = None):
        """
        Run causal analysis on multiple metrics
        
        Args:
            metrics: List of metric names to analyze
        """
        if metrics is None:
            metrics = self.SUPPORTED_METRICS
        
        print("\n" + "=" * 80)
        print("MULTI-METRIC CAUSAL ANALYSIS")
        print("=" * 80)
        
        for metric in metrics:
            print(f"\n--- Analyzing: {metric} ---")
            
            try:
                analysis_data = self.pipeline.get_analysis_series(metric=metric)
                
                analyzer = CausalAnalyzer(analysis_data, config=self.config)
                analyzer.run_causal_impact()
                
                impact = analyzer.get_impact_metrics()
                
                self.results[metric] = {
                    'cumulative_effect': impact['cumulative_effect'],
                    'cumulative_effect_pct': impact['cumulative_effect_pct'],
                    'average_effect': impact['average_effect'],
                    'p_value': impact['p_value'],
                    'is_significant': impact['p_value'] < 0.05
                }
                
                print(f"  Effect: {impact['cumulative_effect']:,.2f} ({impact['cumulative_effect_pct']:.2f}%)")
                print(f"  P-value: {impact['p_value']:.4f}")
                
            except Exception as e:
                print(f"  Error: {str(e)}")
                self.results[metric] = None
        
        return self
    
    def get_summary_dataframe(self) -> pd.DataFrame:
        """Get results as a DataFrame"""
        rows = []
        for metric, data in self.results.items():
            if data:
                rows.append({
                    'Metric': metric,
                    'Cumulative Effect': data['cumulative_effect'],
                    'Effect (%)': data['cumulative_effect_pct'],
                    'Avg Daily Effect': data['average_effect'],
                    'P-Value': data['p_value'],
                    'Significant': '✅' if data['is_significant'] else '⚠️'
                })
        
        return pd.DataFrame(rows)
    
    def plot_comparison(self, save_path: str = None):
        """Generate comparison chart across metrics"""
        print("\nGenerating multi-metric comparison chart...")
        
        df = self.get_summary_dataframe()
        if df.empty:
            print("No results to plot")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Absolute effects
        ax1 = axes[0]
        colors = ['#06A77D' if r['is_significant'] else '#E63946' 
                  for m, r in self.results.items() if r]
        bars = ax1.bar(df['Metric'], df['Cumulative Effect'], color=colors, edgecolor='black')
        ax1.axhline(0, color='black', linewidth=1)
        ax1.set_ylabel('Cumulative Effect', fontweight='bold')
        ax1.set_title('Absolute Impact by Metric', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # Relative effects
        ax2 = axes[1]
        bars2 = ax2.bar(df['Metric'], df['Effect (%)'], color=colors, edgecolor='black')
        ax2.axhline(0, color='black', linewidth=1)
        ax2.set_ylabel('Effect (%)', fontweight='bold')
        ax2.set_title('Relative Impact by Metric', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot: {save_path}")
        
        plt.close()
        return fig
    
    def export_results(self, output_path: str = None):
        """Export results to CSV"""
        if output_path is None:
            output_path = Path(self.config['data']['processed_dir']) / 'multi_metric_results.csv'
        
        df = self.get_summary_dataframe()
        df.to_csv(output_path, index=False)
        print(f"✓ Exported results: {output_path}")
        
        return df


def main():
    """Run multi-metric analysis"""
    print("=" * 80)
    print("MULTI-METRIC CAUSAL IMPACT ANALYSIS")
    print("=" * 80)
    
    analyzer = MultiMetricAnalyzer('config.yaml')
    analyzer.load_data()
    analyzer.analyze_metrics()
    
    # Display summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(analyzer.get_summary_dataframe().to_string(index=False))
    
    # Plot and export
    output_dir = Path('reports/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    analyzer.plot_comparison(save_path=output_dir / 'multi_metric_analysis.png')
    analyzer.export_results()
    
    print("\n✅ Multi-metric analysis completed!")
    
    return analyzer


if __name__ == '__main__':
    main()
