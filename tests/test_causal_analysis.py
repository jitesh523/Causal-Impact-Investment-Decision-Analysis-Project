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


@pytest.mark.parametrize("field,interval_sign", [
    ("point_effect", 0),
    ("point_effect_lower", -1),
    ("point_effect_upper", 1),
])
def test_integer_observations_preserve_fractional_point_effects(monkeypatch, field, interval_sign):
    """Integer-valued metrics still have fractional counterfactual effects."""
    from unittest.mock import Mock
    from scipy import stats

    observed = np.arange(12, dtype=np.int64) + 10
    effect = np.tile([0.25, -0.75, 0.5, -0.125], 3)
    predicted = observed.astype(float) - effect
    uncertainty = np.full(12, 0.2)
    fitted_model = Mock()
    fitted_model.predict.return_value = predicted, uncertainty
    fitted_model.score.return_value = 1.0
    monkeypatch.setattr("causal_analysis.BayesianRidge", lambda **kwargs: fitted_model)
    analyzer = CausalAnalyzer({
        "data": pd.DataFrame({"observed": observed}),
        "y": observed,
        "X": np.ones((12, 1)),
        "dates": pd.date_range("2024-01-01", periods=12),
        "pre_period": [0, 7],
        "post_period": [8, 11],
    }).run_causal_impact()

    expected = effect[8:] + interval_sign * stats.norm.ppf(0.975) * uncertainty[8:]
    np.testing.assert_allclose(analyzer.predictions[field][8:], expected, atol=1e-12)
    np.testing.assert_array_equal(analyzer.predictions[field][:8], np.zeros(8))


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
