"""
Financial Analysis Module
Step 6: Translating Model Outputs into Dollar Value Impact

This module:
- Calculates financial impact from causal analysis
- ComputesROI metrics
- Quantifies gains/losses prevented
- Generates business-friendly summaries
"""

import pandas as pd
import numpy as np
from pathlib import Path


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
        
    def calculate_roi(self):
        """Calculate Return on Investment metrics"""
        print("\n" + "=" * 80)
        print("FINANCIAL IMPACT ANALYSIS")
        print("=" * 80)
        
        # Revenue impact
        cumulative_revenue = self.metrics['cumulative_effect']
        avg_daily_revenue = self.metrics['average_effect']
        
        # Net profit (assuming revenue - cost)
        net_profit = cumulative_revenue - self.campaign_cost
        
        # ROI calculation
        if self.campaign_cost > 0:
            roi = (net_profit / self.campaign_cost) * 100
            roi_ratio = net_profit / self.campaign_cost
        else:
            roi = 0
            roi_ratio = 0
        
        self.financial_results = {
            'cumulative_revenue_impact': cumulative_revenue,
            'average_daily_revenue_impact': avg_daily_revenue,
            'campaign_cost': self.campaign_cost,
            'net_profit': net_profit,
            'roi_percentage': roi,
            'roi_ratio': roi_ratio,
            'statistical_significance': self.metrics['p_value'],
            'is_significant': self.metrics['p_value'] < 0.05
        }
        
        return self
    
    def get_summary(self):
        """Generate executive summary"""
        if not self.financial_results:
            self.calculate_roi()
        
        r = self.financial_results
        
        print("\n" + "=" * 80)
        print("EXECUTIVE FINANCIAL SUMMARY")
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
        narrative.append("BUSINESS NARRATIVE")
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
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if not self.financial_results:
            self.calculate_roi()
        
        # Save as CSV
        df = pd.DataFrame([self.financial_results])
        csv_path = output_path / 'financial_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Exported financial results: {csv_path}")
        
        # Save narrative
        narrative = self.generate_business_narrative()
        txt_path = output_path / 'business_narrative.txt'
        with open(txt_path, 'w') as f:
            f.write(narrative)
        print(f"✓ Exported business narrative: {txt_path}")
        
        return self


def main():
    """Run financial analysis"""
    print("=" * 80)
    print("FINANCIAL ANALYSIS")
    print("Step 6: Translating to Dollar Value Impact")
    print("=" * 80)
    
    # Load impact metrics
    metrics_path = 'data/processed/impact_metrics.csv'
    metrics_df = pd.read_csv(metrics_path)
    metrics = metrics_df.to_dict('records')[0]
    
    print(f"\nLoaded impact metrics from {metrics_path}")
    
    # Assume campaign cost (you can modify this)
    campaign_cost = 5000.0  # Example: $5K campaign cost
    
    print(f"Campaign cost: ${campaign_cost:,.2f}")
    
    # Run financial analysis
    analyzer = FinancialAnalyzer(metrics, campaign_cost=campaign_cost)
    analyzer.calculate_roi()
    
    # Get summary
    summary = analyzer.get_summary()
    
    # Print narrative
    print("\n" + analyzer.generate_business_narrative())
    
    # Export results
    analyzer.export_results()
    
    print("\n✅ Financial analysis completed!")
    
    return analyzer


if __name__ == '__main__':
    main()
