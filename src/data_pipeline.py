
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import yaml

class DataPipeline:
    """Data cleaning and transformation pipeline"""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize pipeline with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.data_path = Path(self.config['data']['raw_path'])
        self.raw_data = None
        self.cleaned_data = None
        self.time_series_data = None
        
    def load_data(self):
        """Load data from Excel file"""
        print(f"Loading data from {self.data_path}...")
        self.raw_data = pd.read_excel(self.data_path)
        print(f"✓ Loaded {len(self.raw_data)} records")
        return self
    
    def clean_data(self):
        """Clean and validate data"""
        print("\nCleaning data...")
        df = self.raw_data.copy()
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['user_id'])
        
        # Validate treatment
        assert df['treatment_exposed'].isin([0, 1]).all(), "Invalid treatment flag"
        
        # Handle ROI calculation
        df['roi_calculated'] = np.where(
            df['spend_usd'] > 0,
            (df['revenue_usd'] - df['spend_usd']) / df['spend_usd'],
            df['roi']
        )
        
        self.cleaned_data = df
        return self
    
    def create_time_series(self, start_date=None, intervention_date=None, segment_col=None, segment_val=None):
        """
        Create time series data, optionally filtered by segment
        """
        # Use config dates if not provided
        if not start_date:
            start_date = self.config['dates']['start_date']
        if not intervention_date:
            intervention_date = self.config['dates']['intervention_date']
            
        print(f"\nCreating time series (intervention: {intervention_date})")
        if segment_col and segment_val:
            print(f"Filter: {segment_col} = {segment_val}")
            
        df = self.cleaned_data.copy()
        
        # Apply filter if specified
        if segment_col and segment_val:
            df = df[df[segment_col] == segment_val]
        
        if len(df) == 0:
            raise ValueError("No data found for specified segment")
            
        start = pd.to_datetime(start_date)
        intervention = pd.to_datetime(intervention_date)
        
        # Split by treatment
        treated = df[df['treatment_exposed'] == 1].copy()
        control = df[df['treatment_exposed'] == 0].copy()
        
        # Assign dates (Simulation Logic)
        # Treated: Post-intervention
        n_treated = len(treated)
        post_days = pd.date_range(intervention, periods=60, freq='D')
        if n_treated > 0:
            treated['date'] = np.random.choice(post_days, size=n_treated, replace=True)
        
        # Control: Full range
        n_control = len(control)
        all_days = pd.date_range(start, periods=150, freq='D')
        if n_control > 0:
            control['date'] = np.random.choice(all_days, size=n_control, replace=True)
            
        # Combine
        time_df = pd.concat([treated, control], ignore_index=True)
        time_df = time_df.sort_values('date').reset_index(drop=True)
        
        # Aggregate
        daily_agg = time_df.groupby(['date', 'treatment_exposed']).agg({
            'impressions': 'sum',
            'clicks': 'sum',
            'spend_usd': 'sum',
            'conversion': 'sum',
            'revenue_usd': 'sum',
            'user_id': 'count'
        }).rename(columns={'user_id': 'n_users'}).reset_index()
        
        # Calculate rates
        daily_agg['ctr'] = daily_agg['clicks'] / daily_agg['impressions'].replace(0, np.nan)
        daily_agg['conversion_rate'] = daily_agg['conversion'] / daily_agg['n_users']
        daily_agg['avg_revenue_per_user'] = daily_agg['revenue_usd'] / daily_agg['n_users']
        
        daily_agg = daily_agg.fillna(0)
        daily_agg['period'] = np.where(daily_agg['date'] < intervention, 'pre', 'post')
        
        self.time_series_data = daily_agg
        return self

    def get_analysis_series(self, metric='revenue_usd'):
        """Prepare data for causal analysis"""
        df = self.time_series_data.copy()
        
        treated = df[df['treatment_exposed'] == 1].groupby('date')[metric].sum()
        control = df[df['treatment_exposed'] == 0].groupby('date')[metric].sum()
        
        merged = pd.concat([treated, control], axis=1).fillna(0)
        merged.columns = ['treated', 'control']
        merged = merged.sort_index()
        
        intervention = pd.to_datetime(self.config['dates']['intervention_date'])
        
        # Find index for intervention
        dates = merged.index
        intervention_idx = np.searchsorted(dates, intervention)
        
        return {
            'data': merged,
            'y': merged['treated'].values,
            'X': merged[['control']].values,
            'dates': dates,
            'pre_period': [0, intervention_idx - 1],
            'post_period': [intervention_idx, len(merged) - 1]
        }
        
    def export_data(self, suffix=''):
        """Export data with optional suffix"""
        out_dir = Path(self.config['data']['processed_dir'])
        out_dir.mkdir(parents=True, exist_ok=True)
        
        name = f"time_series_data{'_' + suffix if suffix else ''}.csv"
        self.time_series_data.to_csv(out_dir / name, index=False)
        print(f"✓ Exported: {name}")

    def get_summary_stats(self):
        """Get summary statistics by treatment group"""
        summary = self.cleaned_data.groupby('treatment_exposed').agg({
            'user_id': 'count',
            'impressions': 'mean',
            'clicks': 'mean',
            'spend_usd': 'mean',
            'conversion': 'sum',
            'revenue_usd': ['sum', 'mean'],
            'roi_calculated': 'mean'
        }).round(2)
        
        return summary

def main():
    print("Running Data Pipeline with Config...")
    pipeline = DataPipeline()
    pipeline.load_data().clean_data().create_time_series().export_data()
    print("Done.")

if __name__ == '__main__':
    main()
