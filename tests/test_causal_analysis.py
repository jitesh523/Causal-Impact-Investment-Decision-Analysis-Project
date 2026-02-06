"""
Unit Tests for Causal Analysis Module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_pipeline import DataPipeline
from causal_analysis import CausalAnalyzer


class TestCausalAnalyzer:
    """Test suite for CausalAnalyzer class"""
    
    @pytest.fixture
    def analysis_data(self, mock_config_file):
        """Create analysis data for testing"""
        pipeline = DataPipeline(str(mock_config_file))
        pipeline.load_data().clean_data().create_time_series()
        return pipeline.get_analysis_series(metric='revenue_usd')
    
    @pytest.fixture
    def analyzer(self, analysis_data):
        """Create analyzer instance"""
        return CausalAnalyzer(analysis_data)
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initializes correctly"""
        assert analyzer is not None
        assert analyzer.y is not None
        assert analyzer.X is not None
        assert analyzer.dates is not None
    
    def test_run_causal_impact(self, analyzer):
        """Test causal impact analysis runs"""
        analyzer.run_causal_impact()
        
        assert analyzer.model is not None
        assert analyzer.predictions is not None
        assert analyzer.impact_results is not None
    
    def test_predictions_structure(self, analyzer):
        """Test predictions have correct structure"""
        analyzer.run_causal_impact()
        
        expected_keys = ['actual', 'predicted', 'predicted_lower', 
                        'predicted_upper', 'point_effect']
        for key in expected_keys:
            assert key in analyzer.predictions, f"Missing prediction key: {key}"
    
    def test_impact_results_structure(self, analyzer):
        """Test impact results have correct structure"""
        analyzer.run_causal_impact()
        
        expected_keys = ['average_effect', 'cumulative_effect', 
                        'p_value', 'cumulative_effect_pct']
        for key in expected_keys:
            assert key in analyzer.impact_results, f"Missing result key: {key}"
    
    def test_p_value_range(self, analyzer):
        """Test p-value is in valid range"""
        analyzer.run_causal_impact()
        
        p_value = analyzer.impact_results['p_value']
        assert 0 <= p_value <= 1, f"Invalid p-value: {p_value}"
    
    def test_get_summary(self, analyzer):
        """Test summary generation"""
        analyzer.run_causal_impact()
        result = analyzer.get_summary()
        
        assert isinstance(result, dict)
        assert 'cumulative_effect' in result
    
    def test_get_impact_metrics(self, analyzer):
        """Test metrics extraction"""
        analyzer.run_causal_impact()
        metrics = analyzer.get_impact_metrics()
        
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
    
    def test_plot_results(self, analyzer, tmp_path):
        """Test plot generation"""
        analyzer.run_causal_impact()
        
        plot_path = tmp_path / "test_plot.png"
        fig = analyzer.plot_results(save_path=str(plot_path))
        
        assert fig is not None
        assert plot_path.exists()


class TestCausalAnalyzerEdgeCases:
    """Test edge cases and error handling"""
    
    def test_analysis_without_run_raises_error(self, mock_config_file):
        """Test that getting summary before run raises error"""
        pipeline = DataPipeline(str(mock_config_file))
        pipeline.load_data().clean_data().create_time_series()
        data = pipeline.get_analysis_series(metric='revenue_usd')
        
        analyzer = CausalAnalyzer(data)
        
        with pytest.raises(ValueError):
            analyzer.get_summary()
    
    def test_metrics_without_run_raises_error(self, mock_config_file):
        """Test that getting metrics before run raises error"""
        pipeline = DataPipeline(str(mock_config_file))
        pipeline.load_data().clean_data().create_time_series()
        data = pipeline.get_analysis_series(metric='revenue_usd')
        
        analyzer = CausalAnalyzer(data)
        
        with pytest.raises(ValueError):
            analyzer.get_impact_metrics()
