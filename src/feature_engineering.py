"""
Feature Engineering Module
==========================

Automated feature engineering for causal inference and ML models.
Creates time-series features, interaction terms, and transformations.

Author: Causal Impact Analysis Project
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, mutual_info_regression
import warnings


@dataclass
class FeatureInfo:
    """Information about a generated feature."""
    name: str
    source_columns: List[str]
    transformation: str
    category: str
    importance: Optional[float] = None


class FeatureEngineer:
    """
    Automated Feature Engineering for Causal Analysis.
    
    Creates features from raw data including:
    - Time-series features (lags, rolling stats)
    - Date/time features
    - Interaction terms
    - Polynomial features
    - Statistical transformations
    
    Example:
        >>> fe = FeatureEngineer()
        >>> fe.fit(df, date_col='date', numeric_cols=['revenue', 'users'])
        >>> df_features = fe.transform(df)
    """
    
    def __init__(
        self,
        create_lags: bool = True,
        lag_periods: List[int] = None,
        create_rolling: bool = True,
        rolling_windows: List[int] = None,
        create_date_features: bool = True,
        create_interactions: bool = False,
        create_polynomials: bool = False,
        polynomial_degree: int = 2
    ):
        """
        Initialize Feature Engineer.
        
        Args:
            create_lags: Create lag features
            lag_periods: Lag periods to create (default [1, 7, 14, 30])
            create_rolling: Create rolling statistics
            rolling_windows: Rolling window sizes (default [7, 14, 30])
            create_date_features: Extract date components
            create_interactions: Create pairwise interactions
            create_polynomials: Create polynomial features
            polynomial_degree: Degree for polynomial features
        """
        self.create_lags = create_lags
        self.lag_periods = lag_periods or [1, 7, 14, 30]
        self.create_rolling = create_rolling
        self.rolling_windows = rolling_windows or [7, 14, 30]
        self.create_date_features = create_date_features
        self.create_interactions = create_interactions
        self.create_polynomials = create_polynomials
        self.polynomial_degree = polynomial_degree
        
        self._fitted = False
        self._feature_info: List[FeatureInfo] = []
        self._numeric_cols: List[str] = []
        self._date_col: Optional[str] = None
        self._scaler: Optional[StandardScaler] = None
    
    def fit(
        self,
        df: pd.DataFrame,
        date_col: Optional[str] = None,
        numeric_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None
    ) -> 'FeatureEngineer':
        """
        Fit feature engineer to data.
        
        Args:
            df: Input DataFrame
            date_col: Date/datetime column
            numeric_cols: Numeric columns for feature creation
            target_col: Target variable (for feature selection)
        
        Returns:
            self
        """
        self._date_col = date_col
        
        # Auto-detect numeric columns
        if numeric_cols is None:
            self._numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col and target_col in self._numeric_cols:
                self._numeric_cols.remove(target_col)
        else:
            self._numeric_cols = numeric_cols
        
        # Fit scaler
        if self._numeric_cols:
            self._scaler = StandardScaler()
            self._scaler.fit(df[self._numeric_cols].fillna(0))
        
        self._fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data with engineered features.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with new features
        """
        if not self._fitted:
            raise ValueError("Must fit before transform")
        
        result = df.copy()
        self._feature_info = []
        
        # Date features
        if self.create_date_features and self._date_col:
            result = self._add_date_features(result)
        
        # Lag features
        if self.create_lags:
            result = self._add_lag_features(result)
        
        # Rolling statistics
        if self.create_rolling:
            result = self._add_rolling_features(result)
        
        # Interactions
        if self.create_interactions:
            result = self._add_interaction_features(result)
        
        # Polynomials
        if self.create_polynomials:
            result = self._add_polynomial_features(result)
        
        return result
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        date_col: Optional[str] = None,
        numeric_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df, date_col, numeric_cols, target_col)
        return self.transform(df)
    
    def _add_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from date column."""
        date_series = pd.to_datetime(df[self._date_col])
        
        # Year, month, day
        df['year'] = date_series.dt.year
        df['month'] = date_series.dt.month
        df['day'] = date_series.dt.day
        df['day_of_week'] = date_series.dt.dayofweek
        df['day_of_year'] = date_series.dt.dayofyear
        df['week_of_year'] = date_series.dt.isocalendar().week
        df['quarter'] = date_series.dt.quarter
        
        # Binary features
        df['is_weekend'] = (date_series.dt.dayofweek >= 5).astype(int)
        df['is_month_start'] = date_series.dt.is_month_start.astype(int)
        df['is_month_end'] = date_series.dt.is_month_end.astype(int)
        
        # Cyclical encoding for month and day of week
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        for col in ['year', 'month', 'day_of_week', 'is_weekend']:
            self._feature_info.append(FeatureInfo(
                name=col,
                source_columns=[self._date_col],
                transformation='date_extract',
                category='temporal'
            ))
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features."""
        for col in self._numeric_cols:
            for lag in self.lag_periods:
                lag_col = f'{col}_lag_{lag}'
                df[lag_col] = df[col].shift(lag)
                
                self._feature_info.append(FeatureInfo(
                    name=lag_col,
                    source_columns=[col],
                    transformation=f'lag_{lag}',
                    category='temporal'
                ))
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistics."""
        for col in self._numeric_cols:
            for window in self.rolling_windows:
                # Rolling mean
                mean_col = f'{col}_roll_mean_{window}'
                df[mean_col] = df[col].rolling(window=window, min_periods=1).mean()
                
                # Rolling std
                std_col = f'{col}_roll_std_{window}'
                df[std_col] = df[col].rolling(window=window, min_periods=1).std()
                
                # Rolling min/max
                min_col = f'{col}_roll_min_{window}'
                df[min_col] = df[col].rolling(window=window, min_periods=1).min()
                
                max_col = f'{col}_roll_max_{window}'
                df[max_col] = df[col].rolling(window=window, min_periods=1).max()
                
                for feat_col, transform in [(mean_col, 'mean'), (std_col, 'std')]:
                    self._feature_info.append(FeatureInfo(
                        name=feat_col,
                        source_columns=[col],
                        transformation=f'rolling_{transform}_{window}',
                        category='temporal'
                    ))
        
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pairwise interaction features."""
        if len(self._numeric_cols) < 2:
            return df
        
        for i, col1 in enumerate(self._numeric_cols):
            for col2 in self._numeric_cols[i + 1:]:
                # Multiplication
                mult_col = f'{col1}_x_{col2}'
                df[mult_col] = df[col1] * df[col2]
                
                # Ratio (with protection)
                ratio_col = f'{col1}_div_{col2}'
                df[ratio_col] = df[col1] / (df[col2] + 1e-6)
                
                self._feature_info.append(FeatureInfo(
                    name=mult_col,
                    source_columns=[col1, col2],
                    transformation='multiplication',
                    category='interaction'
                ))
        
        return df
    
    def _add_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add polynomial features."""
        if not self._numeric_cols:
            return df
        
        poly = PolynomialFeatures(degree=self.polynomial_degree, include_bias=False)
        
        numeric_data = df[self._numeric_cols].fillna(0)
        poly_features = poly.fit_transform(numeric_data)
        
        feature_names = poly.get_feature_names_out(self._numeric_cols)
        
        for i, name in enumerate(feature_names):
            if name not in self._numeric_cols:  # Skip original features
                df[f'poly_{name}'] = poly_features[:, i]
                
                self._feature_info.append(FeatureInfo(
                    name=f'poly_{name}',
                    source_columns=self._numeric_cols,
                    transformation='polynomial',
                    category='polynomial'
                ))
        
        return df
    
    def get_feature_info(self) -> pd.DataFrame:
        """Get information about generated features."""
        if not self._feature_info:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                'name': f.name,
                'sources': ', '.join(f.source_columns),
                'transformation': f.transformation,
                'category': f.category
            }
            for f in self._feature_info
        ])
    
    def select_features(
        self,
        df: pd.DataFrame,
        target: pd.Series,
        k: int = 20,
        method: str = 'mutual_info'
    ) -> List[str]:
        """
        Select top k features based on importance.
        
        Args:
            df: Feature DataFrame
            target: Target variable
            k: Number of features to select
            method: Selection method ('mutual_info', 'correlation')
        
        Returns:
            List of selected feature names
        """
        # Get numeric columns only
        numeric_df = df.select_dtypes(include=[np.number]).dropna(axis=1)
        
        if method == 'mutual_info':
            selector = SelectKBest(mutual_info_regression, k=min(k, len(numeric_df.columns)))
            selector.fit(numeric_df.fillna(0), target)
            
            selected_mask = selector.get_support()
            selected_features = numeric_df.columns[selected_mask].tolist()
            
        elif method == 'correlation':
            correlations = numeric_df.apply(lambda x: x.corr(target)).abs()
            selected_features = correlations.nlargest(k).index.tolist()
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return selected_features


def main():
    """Demo feature engineering."""
    print("=" * 60)
    print("FEATURE ENGINEERING DEMO")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'revenue': 1000 + np.random.randn(100).cumsum() * 50,
        'users': 500 + np.random.randint(-20, 30, 100).cumsum(),
        'spend': 200 + np.random.randn(100) * 20
    })
    
    print(f"\nOriginal shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Create feature engineer
    fe = FeatureEngineer(
        create_lags=True,
        lag_periods=[1, 7],
        create_rolling=True,
        rolling_windows=[7],
        create_date_features=True,
        create_interactions=True,
        create_polynomials=False
    )
    
    # Transform
    df_features = fe.fit_transform(
        df,
        date_col='date',
        numeric_cols=['revenue', 'users', 'spend']
    )
    
    print(f"\nTransformed shape: {df_features.shape}")
    print(f"New columns: {len(df_features.columns) - len(df.columns)}")
    
    # Feature info
    print("\nFeature Categories:")
    info = fe.get_feature_info()
    for category in info['category'].unique():
        count = len(info[info['category'] == category])
        print(f"  {category}: {count} features")
    
    print("\nSample features:")
    print(info.head(10).to_string())
    
    print("\n✓ Feature engineering demo completed!")


if __name__ == '__main__':
    main()
