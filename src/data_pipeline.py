"""
Data Pipeline for Causal Impact Analysis
Step 3: Designing the Data Pipeline (SQL for Data Cleaning & Aggregation)

This module handles:
- Data loading from Excel
- Data cleaning (missing values, anomalies)
- Time series aggregation
- Treatment vs Control group creation
- Export for downstream analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


class DataPipeline:
    """Data cleaning and transformation pipeline"""
    
    def __init__(self, data_path: str):
        """Initialize the pipeline with data path"""
        self.data_path = Path(data_path)
        self.raw_data = None
        self.cleaned_data = None
        self.time_series_data = None
        
    def load_data(self):
        """Load data from Excel file"""
        print("Loading data from Excel...")
        self.raw_data = pd.read_excel(self.data_path)
        print(f"✓ Loaded {len(self.raw_data)} records with {len(self.raw_data.columns)} columns")
        return self
    
    def clean_data(self):
        """Clean and validate data"""
        print("\nCleaning data...")
        df = self.raw_data.copy()
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            print(f"Missing values found:\n{missing[missing > 0]}")
        
        # Remove duplicates
        original_len = len(df)
        df = df.drop_duplicates(subset=['user_id'])
        if len(df) < original_len:
            print(f"✓ Removed {original_len - len(df)} duplicate user_ids")
        
        # Validate treatment_exposed is binary
        assert df['treatment_exposed'].isin([0, 1]).all(), "treatment_exposed must be 0 or 1"
        
        # Handle negative ROI (replace -1 with actual calculation where possible)
        df['roi_calculated'] = np.where(
            df['spend_usd'] > 0,
            (df['revenue_usd'] - df['spend_usd']) / df['spend_usd'],
            df['roi']
        )
        
        self.cleaned_data = df
        print(f"✓ Cleaned data: {len(df)} records")
        return self
    
    def create_time_series(self, start_date='2024-01-01', intervention_date='2024-03-15'):
        """
        Create time series data from cross-sectional data
        
        Since the dataset doesn't have dates, we'll simulate a realistic scenario:
        - Assign users to sequential days in pre/post periods
        - Treatment starts on intervention_date
        - Control users throughout entire period
        """
        print(f"\nCreating time series (intervention: {intervention_date})...")
        df = self.cleaned_data.copy()
        
        start = pd.to_datetime(start_date)
        intervention = pd.to_datetime(intervention_date)
        
        # Split by treatment
        treated = df[df['treatment_exposed'] == 1].copy()
        control = df[df['treatment_exposed'] == 0].copy()
        
        # For treated: assign to post-intervention period
        n_treated = len(treated)
        post_days = pd.date_range(intervention, periods=60, freq='D')
        treated['date'] = np.random.choice(post_days, size=n_treated, replace=True)
        
        # For control: assign across entire period (pre + post)
        n_control = len(control)
        all_days = pd.date_range(start, periods=150, freq='D')
        control['date'] = np.random.choice(all_days, size=n_control, replace=True)
        
        # Combine
        time_df = pd.concat([treated, control], ignore_index=True)
        time_df = time_df.sort_values('date').reset_index(drop=True)
        
        # Aggregate by date and treatment status
        daily_agg = time_df.groupby(['date', 'treatment_exposed']).agg({
            'impressions': 'sum',
            'clicks': 'sum',
            'spend_usd': 'sum',
            'conversion': 'sum',
            'revenue_usd': 'sum',
            'user_id': 'count'  # number of users
        }).rename(columns={'user_id': 'n_users'}).reset_index()
        
        # Calculate derived metrics
        daily_agg['ctr'] = daily_agg['clicks'] / daily_agg['impressions'].replace(0, np.nan)
        daily_agg['conversion_rate'] = daily_agg['conversion'] / daily_agg['n_users']
        daily_agg['avg_revenue_per_user'] = daily_agg['revenue_usd'] / daily_agg['n_users']
        daily_agg['roi'] = (daily_agg['revenue_usd'] - daily_agg['spend_usd']) / daily_agg['spend_usd'].replace(0, np.nan)
        
        # Fill NaN with 0 for metrics
        daily_agg = daily_agg.fillna(0)
        
        # Mark pre/post period
        daily_agg['period'] = np.where(daily_agg['date'] < intervention, 'pre', 'post')
        
        self.time_series_data = daily_agg
        print(f"✓ Created time series: {len(daily_agg)} date-treatment combinations")
        print(f"  - Pre-period: {(daily_agg['period'] == 'pre').sum()} records")
        print(f"  - Post-period: {(daily_agg['period'] == 'post').sum()} records")
        print(f"  - Treated: {(daily_agg['treatment_exposed'] == 1).sum()} records")
        print(f"  - Control: {(daily_agg['treatment_exposed'] == 0).sum()} records")
        
        return self
    
    def get_analysis_series(self, metric='revenue_usd'):
        """
        Prepare data for causal impact analysis
        
        Returns:
        - y: Treated series (with both pre and post periods)
        - X: Control series (to use as predictor)
        - pre_period: [start_index, end_of_pre_index]
        - post_period: [start_of_post_index, end_index]
        """
        df = self.time_series_data.copy()
        
        # Get treated and control series
        treated_df = df[df['treatment_exposed'] == 1].groupby('date')[metric].sum().reset_index()
        control_df = df[df['treatment_exposed'] == 0].groupby('date')[metric].sum().reset_index()
        
        # Merge on date to ensure alignment
        merged = pd.merge(
            treated_df.rename(columns={metric: 'treated'}),
            control_df.rename(columns={metric: 'control'}),
            on='date',
            how='outer'
        ).sort_values('date').reset_index(drop=True)
        
        # Fill missing values with 0
        merged = merged.fillna(0)
        
        # Get intervention date
        intervention_idx = merged[merged['date'] >= pd.to_datetime('2024-03-15')].index[0]
        
        print(f"\n✓ Analysis series prepared for metric: {metric}")
        print(f"  - Total time points: {len(merged)}")
        print(f"  - Pre-period: 0 to {intervention_idx - 1} ({intervention_idx} points)")
        print(f"  - Post-period: {intervention_idx} to {len(merged) - 1} ({len(merged) - intervention_idx} points)")
        
        return {
            'data': merged,
            'y': merged['treated'].values,
            'X': merged[['control']].values,
            'dates': merged['date'].values,
            'pre_period': [0, intervention_idx - 1],
            'post_period': [intervention_idx, len(merged) - 1]
        }
    
    def export_data(self, output_dir='data/processed'):
        """Export cleaned and time series data"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export cleaned data
        cleaned_path = output_path / 'cleaned_data.csv'
        self.cleaned_data.to_csv(cleaned_path, index=False)
        print(f"\n✓ Exported cleaned data: {cleaned_path}")
        
        # Export time series
        ts_path = output_path / 'time_series_data.csv'
        self.time_series_data.to_csv(ts_path, index=False)
        print(f"✓ Exported time series: {ts_path}")
        
        # Summary statistics
        summary_path = output_path / 'data_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("=== Data Summary ===\n\n")
            f.write(f"Total records: {len(self.cleaned_data)}\n")
            f.write(f"Treatment exposed: {(self.cleaned_data['treatment_exposed'] == 1).sum()}\n")
            f.write(f"Control: {(self.cleaned_data['treatment_exposed'] == 0).sum()}\n\n")
            f.write("Time Series Summary:\n")
            f.write(self.time_series_data.describe().to_string())
        
        print(f"✓ Exported summary: {summary_path}")
        return self
    
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
    """Run the data pipeline"""
    print("=" * 80)
    print("CAUSAL IMPACT ANALYSIS - DATA PIPELINE")
    print("Step 3: Data Cleaning & Aggregation")
    print("=" * 80)
    
    # Initialize pipeline
    pipeline = DataPipeline('Dataset.xlsx')
    
    # Run pipeline steps
    pipeline.load_data() \
            .clean_data() \
            .create_time_series() \
            .export_data()
    
    # Display summary
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS BY TREATMENT GROUP")
    print("=" * 80)
    print(pipeline.get_summary_stats())
    
    print("\n✅ Data pipeline completed successfully!")
    
    return pipeline


if __name__ == '__main__':
    main()
