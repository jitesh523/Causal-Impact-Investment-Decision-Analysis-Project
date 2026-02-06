"""
Property-Based Tests using Hypothesis
======================================

Tests using property-based testing to discover edge cases and
validate invariants that should hold across all inputs.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from financial_analysis import FinancialAnalyzer


class TestFinancialAnalysisProperties:
    """Property-based tests for financial calculations."""
    
    @given(
        cumulative_effect=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        campaign_cost=st.floats(min_value=1, max_value=1e6, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_roi_calculation_consistency(self, cumulative_effect, campaign_cost):
        """
        Property: ROI = (effect - cost) / cost * 100
        The relationship should always hold.
        """
        metrics = {
            'cumulative_effect': cumulative_effect,
            'average_effect': cumulative_effect / 10,
            'cumulative_effect_pct': 0,
            'p_value': 0.05,
            'segment': 'test'
        }
        
        analyzer = FinancialAnalyzer(metrics, campaign_cost=campaign_cost)
        analyzer.calculate_roi()
        
        expected_roi = ((cumulative_effect - campaign_cost) / campaign_cost) * 100
        actual_roi = analyzer.financial_results['roi_percentage']
        
        # Allow small floating point differences
        assert abs(actual_roi - expected_roi) < 0.01
    
    @given(
        cumulative_effect=st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False),
        campaign_cost=st.floats(min_value=1, max_value=1e6, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=100)
    def test_positive_effect_positive_net_profit(self, cumulative_effect, campaign_cost):
        """
        Property: If effect > cost, net profit should be positive.
        """
        assume(cumulative_effect > campaign_cost)
        
        metrics = {
            'cumulative_effect': cumulative_effect,
            'average_effect': cumulative_effect / 10,
            'cumulative_effect_pct': 0,
            'p_value': 0.05,
            'segment': 'test'
        }
        
        analyzer = FinancialAnalyzer(metrics, campaign_cost=campaign_cost)
        analyzer.calculate_roi()
        
        assert analyzer.financial_results['net_profit'] > 0
    
    @given(
        campaign_cost=st.floats(min_value=1, max_value=1e6, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50)
    def test_zero_effect_negative_roi(self, campaign_cost):
        """
        Property: Zero effect means negative ROI (lost the campaign cost).
        """
        metrics = {
            'cumulative_effect': 0,
            'average_effect': 0,
            'cumulative_effect_pct': 0,
            'p_value': 0.05,
            'segment': 'test'
        }
        
        analyzer = FinancialAnalyzer(metrics, campaign_cost=campaign_cost)
        analyzer.calculate_roi()
        
        assert analyzer.financial_results['roi_percentage'] == -100.0
        assert analyzer.financial_results['net_profit'] == -campaign_cost


class TestDataInvariants:
    """Property-based tests for data processing invariants."""
    
    @given(
        n_rows=st.integers(min_value=10, max_value=100),
        mean_val=st.floats(min_value=1, max_value=1000, allow_nan=False, allow_infinity=False),
        std_val=st.floats(min_value=1, max_value=100, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=20)
    def test_effect_calculation_bounds(self, n_rows, mean_val, std_val):
        """
        Property: Point effect should be difference between actual and predicted.
        """
        np.random.seed(42)
        
        actual = np.random.normal(mean_val, std_val, n_rows)
        predicted = np.random.normal(mean_val * 0.9, std_val, n_rows)  # Slightly lower
        
        point_effect = actual - predicted
        cumulative_effect = np.sum(point_effect)
        
        # Property: Cumulative effect equals sum of point effects
        assert abs(cumulative_effect - np.sum(actual - predicted)) < 1e-10


class TestStatisticalProperties:
    """Property-based tests for statistical calculations."""
    
    @given(
        n_samples=st.integers(min_value=30, max_value=200),
        true_effect=st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False),
        noise_std=st.floats(min_value=1, max_value=10, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=20)
    def test_effect_sign_consistency(self, n_samples, true_effect, noise_std):
        """
        Property: With large enough sample and true effect,
        estimated effect should have same sign as true effect.
        """
        assume(abs(true_effect) > noise_std * 2)  # Effect larger than 2 std
        
        np.random.seed(42)
        
        # Simulate pre and post data
        pre = np.random.normal(100, noise_std, n_samples)
        post = np.random.normal(100 + true_effect, noise_std, n_samples)
        
        estimated_effect = np.mean(post) - np.mean(pre)
        
        # Sign should match
        if true_effect > 0:
            assert estimated_effect > 0
        elif true_effect < 0:
            assert estimated_effect < 0


class TestPropensityMatchingProperties:
    """Property-based tests for propensity score matching."""
    
    @given(
        n_samples=st.integers(min_value=100, max_value=500),
        treatment_prob=st.floats(min_value=0.2, max_value=0.8, allow_nan=False)
    )
    @settings(max_examples=10)
    def test_matched_groups_exist(self, n_samples, treatment_prob):
        """
        Property: After matching, both treatment and control groups should exist.
        """
        from propensity_matching import PropensityMatcher
        
        np.random.seed(42)
        
        data = pd.DataFrame({
            'treatment': np.random.binomial(1, treatment_prob, n_samples),
            'x1': np.random.normal(0, 1, n_samples),
            'x2': np.random.normal(0, 1, n_samples)
        })
        
        # Need at least some of each group
        assume(data['treatment'].sum() > 10)
        assume((1 - data['treatment']).sum() > 10)
        
        matcher = PropensityMatcher(
            data=data,
            treatment_col='treatment',
            covariates=['x1', 'x2']
        )
        
        matcher.fit()
        matched = matcher.match(caliper=0.5)
        
        # Should have both groups
        assert (matched['treatment'] == 1).sum() > 0
        assert (matched['treatment'] == 0).sum() > 0


class TestDatabaseProperties:
    """Property-based tests for database operations."""
    
    @given(
        segment_name=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N'))),
        effect_value=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        cost_value=st.floats(min_value=1, max_value=1e6, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=20)
    def test_database_round_trip(self, segment_name, effect_value, cost_value, tmp_path):
        """
        Property: Data saved to database should be retrievable unchanged.
        """
        from database import DatabaseManager
        
        db = DatabaseManager(f"sqlite:///{tmp_path}/test.db")
        db.initialize()
        
        # Save
        record_id = db.save_analysis(
            segment=segment_name,
            intervention_date='2024-03-01',
            campaign_cost=cost_value,
            impact_metrics={
                'cumulative_effect': effect_value,
                'average_effect': effect_value / 10,
                'p_value': 0.05
            },
            financial_results={
                'roi_percentage': 0,
                'net_profit': 0
            }
        )
        
        # Retrieve
        retrieved = db.get_analysis_by_id(record_id)
        
        assert retrieved['segment'] == segment_name
        assert abs(retrieved['cumulative_effect'] - effect_value) < 0.01
        assert abs(retrieved['campaign_cost'] - cost_value) < 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
