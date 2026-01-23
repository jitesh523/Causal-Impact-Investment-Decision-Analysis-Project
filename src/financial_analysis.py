
"""
Financial Analysis Module
Step 6: Translating Model Outputs into Dollar Value Impact

This module:
- Calculates financial impact from causal analysis
- Computes ROI metrics
- Quantifies gains/losses prevented
- Generates business-friendly summaries
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml

class FinancialAnalyzer:
    """Translate causal impact to financial metrics"""
    
    def __init__(self, impact_metrics, campaign_cost=0):
        """
        Initialize with impact metrics and campaign cost
        
        Args:
            impact_metrics: Dict from CausalAnalyzer.get_impact_metrics()
            campaign_cost: Total cost of the intervention/campaign
        """
        self.metrics = impact_metrics
        self.campaign_cost = campaign_cost
        self.financial_results = {}
        self.segment_name = self.metrics.get('segment', 'aggregated')
        
    def calculate_roi(self):
        """Calculate Return on Investment metrics"""
        print("\n" + "=" * 80)
        print(f"FINANCIAL IMPACT ANALYSIS [{self.segment_name}]")
        print("=" * 80)
        
        # Revenue impact
        cumulative_revenue = self.metrics['cumulative_effect']
        avg_daily_revenue = self.metrics['average_effect']
        
        # Net profit (assuming revenue - cost)
        # Note: reliable mostly for aggregated view unless cost is segmented
        net_profit = cumulative_revenue - self.campaign_cost
        
        # ROI calculation
        if self.campaign_cost > 0:
            roi = (net_profit / self.campaign_cost) * 100
            roi_ratio = net_profit / self.campaign_cost
        else:
            roi = 0
            roi_ratio = 0
        
        self.financial_results = {
            'segment': self.segment_name,
            'cumulative_revenue_impact': cumulative_revenue,
            'average_daily_revenue_impact': avg_daily_revenue,
            'campaign_cost': self.campaign_cost,
            'net_profit': net_profit,
            'roi_percentage': roi,
            'roi_ratio': roi_ratio,
            'statistical_significance': self.metrics['p_value'],
            'is_significant': self.metrics['p_value'] < self.metrics.get('alpha', 0.05)
        }
        
        return self
    
    def get_summary(self):
        """Generate executive summary"""
        if not self.financial_results:
            self.calculate_roi()
        
        r = self.financial_results
        
        print("\n" + "=" * 80)
        print(f"EXECUTIVE FINANCIAL SUMMARY [{self.segment_name}]")
        print("=" * 80)
        
        print(f"\n📊 REVENUE IMPACT")
        print(f"  • Total revenue impact: ${r['cumulative_revenue_impact']:,.2f}")
        print(f"  • Average daily impact: ${r['average_daily_revenue_impact']:,.2f}")
        
        print(f"\n💰 COST-BENEFIT ANALYSIS")
        print(f"  • Campaign cost: ${r['campaign_cost']:,.2f}")
        print(f"  • Net profit: ${r['net_profit']:,.2f}")
        
        if r['campaign_cost'] > 0:
            print(f"\n📈 RETURN ON INVESTMENT")
            print(f"  • ROI: {r['roi_percentage']:.2f}%")
            print(f"  • ROI ratio: {r['roi_ratio']:.2f}x")
            
            if r['roi_percentage'] > 0:
                print(f"  • ✅ The campaign paid for itself {r['roi_ratio']:.2f} times over")
            else:
                print(f"  • ⚠️ The campaign did not generate positive ROI")
        
        print(f"\n📉 STATISTICAL CONFIDENCE")
        print(f"  • P-value: {r['statistical_significance']:.4f}")
        if r['is_significant']:
           print(f"  • ✅ Results are statistically significant (p < 0.05)")
        else:
            print(f"  • ⚠️ Results are NOT statistically significant (p >= 0.05)")
        
        return r
    
    def generate_business_narrative(self):
        """Create narrative summary for stakeholders"""
        if not self.financial_results:
            self.calculate_roi()
        
        r = self.financial_results
        
        narrative = []
        narrative.append("=" * 80)
        narrative.append(f"BUSINESS NARRATIVE [{self.segment_name}]")
        narrative.append("=" * 80)
        narrative.append("")
        
        # Opening statement
        if r['cumulative_revenue_impact'] > 0:
            narrative.append(f"✅ Our analysis estimates that the marketing campaign generated an ")
            narrative.append(f"   additional ${r['cumulative_revenue_impact']:,.2f} in revenue over the ")
            narrative.append(f"   post-intervention period.")
        else:
            narrative.append(f"⚠️ Our analysis estimates that the marketing campaign resulted in a ")
            narrative.append(f"   revenue decrease of ${abs(r['cumulative_revenue_impact']):,.2f} over ")
            narrative.append(f"   the post-intervention period.")
        
        narrative.append("")
        
        # ROI statement
        if r['campaign_cost'] > 0:
            narrative.append(f"💵 After accounting for the ${r['campaign_cost']:,.2f} campaign cost, ")
            narrative.append(f"   the net gain is approximately ${r['net_profit']:,.2f}, which ")
            narrative.append(f"   translates to an ROI of {r['roi_percentage']:.1f}%.")
            narrative.append("")
            
            if r['roi_ratio'] > 3:
                narrative.append(f"🎯 This is an EXCELLENT return - the campaign paid for itself ")
                narrative.append(f"   {r['roi_ratio']:.1f}x over!")
            elif r['roi_ratio'] > 1:
                narrative.append(f"👍 This is a POSITIVE return - the campaign was profitable.")
            elif r['roi_ratio'] > 0:
                narrative.append(f"⚠️ The campaign was marginally profitable but below expectations.")
            else:
                narrative.append(f"❌ The campaign did not achieve positive ROI.")
        else:
            narrative.append(f"ℹ️ No campaign cost data available for ROI calculation.")
        
        narrative.append("")
        
        # Statistical confidence
        if r['is_significant']:
            narrative.append(f"📊 These results are statistically significant (p = {r['statistical_significance']:.4f}), ")
            narrative.append(f"   meaning we can be confident that the observed impact is due to ")
            narrative.append(f"   the marketing intervention and not random chance.")
        else:
            narrative.append(f"📊 CAUTION: These results are not statistically significant ")
            narrative.append(f"   (p = {r['statistical_significance']:.4f}). This means the observed impact ")
            narrative.append(f"   could be due to random variation rather than the campaign.")
        
        narrative.append("")
        narrative.append("=" * 80)
        
        return "\n".join(narrative)
    
    def export_results(self, output_dir='reports'):
        """Export financial results"""
        if not self.financial_results:
            self.calculate_roi() # Calculate if not done
            
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # For aggregated results, we might overwrite. For segments, we should probably append or save separately.
        # Here we just return the result dict, main will handle bulk saving.
        
        # Save narrative ONLY for aggregated
        if self.segment_name == 'aggregated':
            narrative = self.generate_business_narrative()
            txt_path = output_path / 'business_narrative.txt'
            with open(txt_path, 'w') as f:
                f.write(narrative)
            print(f"✓ Exported business narrative: {txt_path}")
        
        return self.financial_results


def main():
    """Run financial analysis"""
    print("=" * 80)
    print("FINANCIAL ANALYSIS")
    print("Step 6: Translating to Dollar Value Impact")
    print("=" * 80)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    campaign_cost = config['campaign']['cost']
    print(f"Campaign cost (Total): ${campaign_cost:,.2f}")
    
    # Load impact metrics
    metrics_path = Path(config['data']['processed_dir']) / 'impact_metrics.csv'
    if not metrics_path.exists():
        print(f"Error: {metrics_path} not found. Run causal_analysis.py first.")
        return
        
    metrics_df = pd.read_csv(metrics_path)
    print(f"\nLoaded impact metrics from {metrics_path}")
    
    all_financial_results = []
    
    # Process each row (aggregated + segments)
    for _, row in metrics_df.iterrows():
        metrics = row.to_dict()
        segment = metrics.get('segment', 'aggregated')
        
        # Determine cost for this segment
        # If aggregated, use full cost.
        # If segment, we don't have split cost. 
        # Strategy: Use 0 cost for segments to show just Revenue Impact, 
        # OR use full cost but that implies full cost was spent on that segment (wrong).
        # OR just skip ROI for segments.
        
        current_cost = campaign_cost if segment == 'aggregated' else 0
        
        analyzer = FinancialAnalyzer(metrics, campaign_cost=current_cost)
        analyzer.calculate_roi()
        
        # Print summary
        if segment == 'aggregated':
            analyzer.get_summary()
            print("\n" + analyzer.generate_business_narrative())
        
        res = analyzer.export_results(output_dir=config['data']['output_dir'])
        all_financial_results.append(res)
        
    # Save all financial results
    fin_df = pd.DataFrame(all_financial_results)
    out_path = Path(config['data']['output_dir']) / 'financial_results.csv'
    fin_df.to_csv(out_path, index=False)
    print(f"\n✓ Exported all financial results to {out_path}")
    
    print("\n✅ Financial analysis completed!")
    
    return all_financial_results


if __name__ == '__main__':
    main()
