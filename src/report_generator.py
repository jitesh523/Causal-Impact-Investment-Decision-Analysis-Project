"""
Automated Report Generator
Generate Professional HTML/PDF Reports

Creates shareable reports with:
- Executive summary
- Key metrics and KPIs
- Visualizations
- Technical methodology
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import yaml
import base64


class ReportGenerator:
    """Generate professional reports from causal impact analysis"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.metrics = None
        self.financial = None
        self.report_date = datetime.now().strftime('%Y-%m-%d %H:%M')
        
    def load_results(self):
        """Load analysis results"""
        # Load impact metrics
        metrics_path = Path(self.config['data']['processed_dir']) / 'impact_metrics.csv'
        if metrics_path.exists():
            self.metrics = pd.read_csv(metrics_path)
            print(f"✓ Loaded impact metrics: {len(self.metrics)} rows")
        
        # Load financial results
        financial_path = Path(self.config['data']['output_dir']) / 'financial_results.csv'
        if financial_path.exists():
            self.financial = pd.read_csv(financial_path)
            print(f"✓ Loaded financial results: {len(self.financial)} rows")
        
        return self
    
    def _image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 for embedding in HTML"""
        try:
            with open(image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except FileNotFoundError:
            return None
    
    def _get_primary_metrics(self) -> dict:
        """Extract primary metrics from aggregated results"""
        if self.metrics is None or self.metrics.empty:
            return {}
        
        # Get aggregated row
        agg = self.metrics[self.metrics['segment'] == 'aggregated']
        if agg.empty:
            agg = self.metrics.iloc[[0]]
        
        return agg.iloc[0].to_dict()
    
    def _get_financial_metrics(self) -> dict:
        """Extract financial metrics"""
        if self.financial is None or self.financial.empty:
            return {}
        
        # Check if segment column exists
        if 'segment' in self.financial.columns:
            agg = self.financial[self.financial['segment'] == 'aggregated']
            if agg.empty:
                agg = self.financial.iloc[[0]]
        else:
            agg = self.financial.iloc[[0]]
        
        return agg.iloc[0].to_dict()
    
    def generate_html(self, output_path: str = None) -> str:
        """Generate HTML report"""
        print("\nGenerating HTML report...")
        
        metrics = self._get_primary_metrics()
        financial = self._get_financial_metrics()
        
        # Get campaign cost
        campaign_cost = self.config['campaign'].get('cost', 0)
        
        # Calculate values with defaults
        cum_effect = metrics.get('cumulative_effect', 0)
        p_value = metrics.get('p_value', 1)
        roi_pct = financial.get('roi_percentage', 0)
        net_profit = financial.get('net_profit', cum_effect - campaign_cost)
        
        # Significance badge
        sig_badge = "✅ Significant" if p_value < 0.05 else "⚠️ Not Significant"
        sig_color = "#06A77D" if p_value < 0.05 else "#E63946"
        
        # Load main plot
        main_plot_path = Path(self.config['data']['figures_dir']) / 'causal_impact_analysis.png'
        main_plot_b64 = self._image_to_base64(main_plot_path)
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Causal Impact Analysis Report</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
        }}
        .container {{ max-width: 900px; margin: 0 auto; padding: 20px; }}
        .header {{
            background: linear-gradient(135deg, #2E86AB, #A23B72);
            color: white;
            padding: 40px;
            text-align: center;
            border-radius: 12px;
            margin-bottom: 30px;
        }}
        .header h1 {{ font-size: 2.5rem; margin-bottom: 10px; }}
        .header p {{ opacity: 0.9; }}
        .card {{
            background: white;
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .card h2 {{ color: #2E86AB; margin-bottom: 15px; border-bottom: 2px solid #f0f0f0; padding-bottom: 10px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
        .metric {{
            text-align: center;
            padding: 20px;
            background: #f9f9f9;
            border-radius: 8px;
        }}
        .metric-value {{ font-size: 2rem; font-weight: bold; color: #2E86AB; }}
        .metric-label {{ color: #666; font-size: 0.9rem; }}
        .badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            color: white;
        }}
        .chart-container {{ text-align: center; margin: 20px 0; }}
        .chart-container img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f5f5f5; font-weight: 600; }}
        .footer {{ text-align: center; padding: 20px; color: #666; font-size: 0.9rem; }}
        .highlight {{ background: linear-gradient(135deg, #06A77D20, #06A77D10); border-left: 4px solid #06A77D; padding: 15px; border-radius: 0 8px 8px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Causal Impact Analysis Report</h1>
            <p>Marketing Campaign Effectiveness Study</p>
            <p style="margin-top: 10px; font-size: 0.9rem;">Generated: {self.report_date}</p>
        </div>

        <div class="card">
            <h2>🏆 Executive Summary</h2>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-value">${cum_effect:,.0f}</div>
                    <div class="metric-label">Revenue Impact</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{roi_pct:.0f}%</div>
                    <div class="metric-label">Return on Investment</div>
                </div>
                <div class="metric">
                    <div class="metric-value">${net_profit:,.0f}</div>
                    <div class="metric-label">Net Profit</div>
                </div>
                <div class="metric">
                    <div class="metric-value"><span class="badge" style="background: {sig_color}; font-size: 1rem;">{sig_badge}</span></div>
                    <div class="metric-label">p = {p_value:.4f}</div>
                </div>
            </div>
        </div>

        <div class="card highlight">
            <strong>Key Finding:</strong> The marketing campaign generated an estimated 
            <strong>${cum_effect:,.2f}</strong> in incremental revenue. After accounting for the 
            <strong>${campaign_cost:,.2f}</strong> campaign cost, the net profit is 
            <strong>${net_profit:,.2f}</strong>, representing a <strong>{roi_pct:.1f}%</strong> ROI.
        </div>

        <div class="card">
            <h2>📈 Causal Impact Visualization</h2>
            <div class="chart-container">
                {"<img src='data:image/png;base64," + main_plot_b64 + "' alt='Causal Impact Analysis'>" if main_plot_b64 else "<p>Chart not available</p>"}
            </div>
            <p style="color: #666; font-size: 0.9rem; text-align: center;">
                Blue line shows observed data; dashed line shows counterfactual prediction (what would have happened without intervention).
            </p>
        </div>

        <div class="card">
            <h2>📋 Detailed Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Cumulative Effect</td><td>${cum_effect:,.2f}</td></tr>
                <tr><td>Average Daily Effect</td><td>${metrics.get('average_effect', 0):,.2f}</td></tr>
                <tr><td>Relative Effect</td><td>{metrics.get('cumulative_effect_pct', 0):.2f}%</td></tr>
                <tr><td>Campaign Cost</td><td>${campaign_cost:,.2f}</td></tr>
                <tr><td>Net Profit</td><td>${net_profit:,.2f}</td></tr>
                <tr><td>ROI</td><td>{roi_pct:.2f}%</td></tr>
                <tr><td>P-Value</td><td>{p_value:.6f}</td></tr>
            </table>
        </div>

        <div class="card">
            <h2>🔬 Methodology</h2>
            <p>This analysis uses <strong>Bayesian Structural Time Series (BSTS)</strong> to estimate 
            the causal effect of the marketing intervention. The method constructs a counterfactual 
            prediction of what would have happened in the absence of the intervention, allowing us 
            to isolate the true incremental impact.</p>
            <br>
            <p><strong>Key assumptions:</strong></p>
            <ul style="margin-left: 20px; margin-top: 10px;">
                <li>Control group was not affected by the intervention</li>
                <li>Pre-intervention trends would have continued</li>
                <li>No other major events occurred during the study period</li>
            </ul>
        </div>

        <div class="footer">
            <p>Report generated by Causal Impact Analysis Pipeline</p>
            <p>Intervention Date: {self.config['dates']['intervention_date']}</p>
        </div>
    </div>
</body>
</html>
"""
        
        # Save report
        if output_path is None:
            output_path = Path(self.config['data']['output_dir']) / 'causal_impact_report.html'
        
        with open(output_path, 'w') as f:
            f.write(html)
        
        print(f"✓ Saved HTML report: {output_path}")
        
        return html


def main():
    """Generate reports"""
    print("=" * 80)
    print("AUTOMATED REPORT GENERATOR")
    print("=" * 80)
    
    generator = ReportGenerator('config.yaml')
    generator.load_results()
    generator.generate_html()
    
    print("\n✅ Report generation completed!")
    
    return generator


if __name__ == '__main__':
    main()
