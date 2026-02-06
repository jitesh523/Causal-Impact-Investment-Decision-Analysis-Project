"""
Unit Tests for Data Pipeline Module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_pipeline import DataPipeline


class TestDataPipeline:
    """Test suite for DataPipeline class"""
    
    @pytest.fixture
    def pipeline(self, mock_config_file):
        """Create a pipeline instance for testing"""
        return DataPipeline(str(mock_config_file))
    
    @pytest.fixture
    def loaded_pipeline(self, pipeline):
        """Pipeline with data loaded"""
        pipeline.load_data()
        return pipeline
    
    @pytest.fixture
    def cleaned_pipeline(self, loaded_pipeline):
        """Pipeline with cleaned data"""
        loaded_pipeline.clean_data()
        return cleaned_pipeline
    
    def test_pipeline_initialization(self, pipeline):
        """Test that pipeline initializes correctly"""
        assert pipeline is not None
        assert pipeline.config is not None
        assert 'data' in pipeline.config
        assert 'dates' in pipeline.config
    
    def test_load_data(self, loaded_pipeline):
        """Test data loading"""
        assert loaded_pipeline.raw_data is not None
        assert isinstance(loaded_pipeline.raw_data, pd.DataFrame)
        assert len(loaded_pipeline.raw_data) > 0
    
    def test_load_data_columns(self, loaded_pipeline):
        """Test that expected columns exist"""
        expected_cols = ['user_id', 'treatment_exposed', 'revenue_usd', 'impressions', 'clicks']
        for col in expected_cols:
            assert col in loaded_pipeline.raw_data.columns, f"Missing column: {col}"
    
    def test_clean_data(self, loaded_pipeline):
        """Test data cleaning"""
        loaded_pipeline.clean_data()
        
        assert loaded_pipeline.cleaned_data is not None
        assert 'roi_calculated' in loaded_pipeline.cleaned_data.columns
        
        # Check no duplicate user_ids
        assert loaded_pipeline.cleaned_data['user_id'].is_unique
    
    def test_treatment_values(self, loaded_pipeline):
        """Test treatment column has valid values"""
        loaded_pipeline.clean_data()
        
        treatment_vals = loaded_pipeline.cleaned_data['treatment_exposed'].unique()
        assert set(treatment_vals).issubset({0, 1})
    
    def test_create_time_series(self, loaded_pipeline):
        """Test time series creation"""
        loaded_pipeline.clean_data()
        loaded_pipeline.create_time_series()
        
        assert loaded_pipeline.time_series_data is not None
        assert 'date' in loaded_pipeline.time_series_data.columns
        assert 'period' in loaded_pipeline.time_series_data.columns
    
    def test_get_analysis_series(self, loaded_pipeline):
        """Test analysis series extraction"""
        loaded_pipeline.clean_data()
        loaded_pipeline.create_time_series()
        
        result = loaded_pipeline.get_analysis_series(metric='revenue_usd')
        
        assert 'data' in result
        assert 'y' in result
        assert 'X' in result
        assert 'dates' in result
        assert 'pre_period' in result
        assert 'post_period' in result
    
    def test_segment_filtering(self, loaded_pipeline):
        """Test segment-based filtering"""
        loaded_pipeline.clean_data()
        
        # Get a valid segment value
        if 'channel' in loaded_pipeline.cleaned_data.columns:
            segment_val = loaded_pipeline.cleaned_data['channel'].iloc[0]
            loaded_pipeline.create_time_series(segment_col='channel', segment_val=segment_val)
            
            assert loaded_pipeline.time_series_data is not None
            assert len(loaded_pipeline.time_series_data) > 0


class TestDataIntegrity:
    """Test data integrity and edge cases"""
    
    def test_no_negative_revenue(self, mock_config_file):
        """Test that revenue values are non-negative"""
        pipeline = DataPipeline(str(mock_config_file))
        pipeline.load_data().clean_data()
        
        assert (pipeline.cleaned_data['revenue_usd'] >= 0).all()
    
    def test_date_range_validity(self, mock_config_file):
        """Test that date ranges are valid"""
        pipeline = DataPipeline(str(mock_config_file))
        pipeline.load_data().clean_data().create_time_series()
        
        dates = pipeline.time_series_data['date']
        assert dates.min() < dates.max()
