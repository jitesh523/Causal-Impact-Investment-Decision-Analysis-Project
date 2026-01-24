"""
Unit Tests for Financial Analysis Module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from financial_analysis import FinancialAnalyzer


class TestFinancialAnalyzer:
    """Test suite for FinancialAnalyzer class"""
    
    @pytest.fixture
    def sample_metrics(self):
        """Create sample impact metrics for testing"""
        return {
            'segment': 'aggregated',
            'cumulative_effect': 50000.0,
            'average_effect': 500.0,
            'cumulative_effect_pct': 25.0,
            'p_value': 0.001,
            'alpha': 0.05
        }
    
    @pytest.fixture
    def analyzer(self, sample_metrics):
        """Create analyzer instance"""
        return FinancialAnalyzer(sample_metrics, campaign_cost=5000.0)
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initializes correctly"""
        assert analyzer is not None
        assert analyzer.campaign_cost == 5000.0
        assert analyzer.metrics is not None
    
    def test_calculate_roi(self, analyzer):
        """Test ROI calculation"""
        analyzer.calculate_roi()
        
        assert analyzer.financial_results is not None
        assert 'roi_percentage' in analyzer.financial_results
        assert 'net_profit' in analyzer.financial_results
    
    def test_roi_values(self, analyzer):
        """Test ROI calculation values"""
        analyzer.calculate_roi()
        
        # Net profit = 50000 - 5000 = 45000
        assert analyzer.financial_results['net_profit'] == 45000.0
        
        # ROI = (45000 / 5000) * 100 = 900%
        assert analyzer.financial_results['roi_percentage'] == 900.0
    
    def test_significance_flag(self, analyzer):
        """Test significance flag is set correctly"""
        analyzer.calculate_roi()
        
        # p_value = 0.001 < 0.05, so should be significant
        assert analyzer.financial_results['is_significant'] == True
    
    def test_get_summary(self, analyzer):
        """Test summary generation"""
        result = analyzer.get_summary()
        
        assert isinstance(result, dict)
        assert 'roi_percentage' in result
    
    def test_generate_narrative(self, analyzer):
        """Test business narrative generation"""
        narrative = analyzer.generate_business_narrative()
        
        assert isinstance(narrative, str)
        assert len(narrative) > 0
        assert '$' in narrative  # Should contain dollar amounts


class TestFinancialAnalyzerEdgeCases:
    """Test edge cases"""
    
    def test_zero_campaign_cost(self):
        """Test handling of zero campaign cost"""
        metrics = {
            'cumulative_effect': 10000.0,
            'average_effect': 100.0,
            'p_value': 0.01,
            'alpha': 0.05
        }
        
        analyzer = FinancialAnalyzer(metrics, campaign_cost=0)
        analyzer.calculate_roi()
        
        assert analyzer.financial_results['roi_percentage'] == 0
        assert analyzer.financial_results['roi_ratio'] == 0
    
    def test_negative_effect(self):
        """Test handling of negative cumulative effect"""
        metrics = {
            'cumulative_effect': -5000.0,
            'average_effect': -50.0,
            'p_value': 0.01,
            'alpha': 0.05
        }
        
        analyzer = FinancialAnalyzer(metrics, campaign_cost=1000.0)
        analyzer.calculate_roi()
        
        # Net profit should be -6000 (loss)
        assert analyzer.financial_results['net_profit'] == -6000.0
        assert analyzer.financial_results['roi_percentage'] < 0
    
    def test_not_significant_result(self):
        """Test handling of non-significant results"""
        metrics = {
            'cumulative_effect': 1000.0,
            'average_effect': 10.0,
            'p_value': 0.15,  # Not significant
            'alpha': 0.05
        }
        
        analyzer = FinancialAnalyzer(metrics, campaign_cost=500.0)
        analyzer.calculate_roi()
        
        assert analyzer.financial_results['is_significant'] == False
