"""
Integration Tests for Causal Impact Analysis
=============================================

End-to-end tests covering the complete analysis pipeline from data loading
to report generation.
"""

import pytest
import sys
import os
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_pipeline import DataPipeline
from causal_analysis import CausalAnalyzer
from financial_analysis import FinancialAnalyzer
from database import DatabaseManager
from experiment_tracker import ExperimentTracker


class TestEndToEndPipeline:
    """Integration tests for the complete analysis pipeline."""
    
    @pytest.fixture
    def mock_dataset(self, tmp_path):
        """Create a mock dataset for testing."""
        np.random.seed(42)
        n_rows = 200
        
        dates = pd.date_range('2024-01-01', periods=n_rows, freq='D')
        
        # Generate data
        data = {
            'date': dates,
            'revenue_usd': np.random.uniform(1000, 5000, n_rows) + np.linspace(0, 1000, n_rows),
            'channel': np.random.choice(['email', 'social', 'search'], n_rows),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_rows),
            'transactions': np.random.randint(10, 100, n_rows),
            'customers': np.random.randint(5, 50, n_rows)
        }
        
        df = pd.DataFrame(data)
        
        # Add intervention effect after day 100
        df.loc[df['date'] >= '2024-04-10', 'revenue_usd'] += 500
        
        dataset_path = tmp_path / 'test_data.xlsx'
        df.to_excel(dataset_path, index=False, engine='openpyxl')
        
        return dataset_path
    
    @pytest.fixture
    def mock_config(self, tmp_path, mock_dataset):
        """Create a mock config file."""
        config = {
            'data': {'file_path': str(mock_dataset)},
            'dates': {
                'start_date': '2024-01-01',
                'end_date': '2024-07-18',
                'intervention_date': '2024-04-10'
            },
            'campaign': {'cost': 5000},
            'segments': ['channel', 'region'],
            'model': {'alpha': 0.05}
        }
        
        config_path = tmp_path / 'config.yaml'
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return config_path, config
    
    def test_full_pipeline_aggregated(self, mock_config):
        """Test complete pipeline with aggregated data."""
        config_path, config = mock_config
        
        # Step 1: Load and clean data
        pipeline = DataPipeline(str(config_path))
        pipeline.load_data()
        pipeline.clean_data()
        
        assert pipeline.cleaned_data is not None
        assert len(pipeline.cleaned_data) > 0
        
        # Step 2: Create time series
        pipeline.create_time_series(intervention_date='2024-04-10')
        
        assert pipeline.time_series_data is not None
        
        # Step 3: Run causal analysis
        analysis_data = pipeline.get_analysis_series(metric='revenue_usd')
        analyzer = CausalAnalyzer(analysis_data, config=config)
        analyzer.run_causal_impact()
        
        # Step 4: Get results
        metrics = analyzer.get_impact_metrics()
        
        assert 'cumulative_effect' in metrics
        assert 'p_value' in metrics
        assert 'average_effect' in metrics
        
        # Step 5: Financial analysis
        fin_analyzer = FinancialAnalyzer(metrics, campaign_cost=5000)
        fin_analyzer.calculate_roi()
        
        assert fin_analyzer.financial_results is not None
        assert 'roi_percentage' in fin_analyzer.financial_results
    
    def test_full_pipeline_segmented(self, mock_config):
        """Test pipeline with segment filtering."""
        config_path, config = mock_config
        
        pipeline = DataPipeline(str(config_path))
        pipeline.load_data().clean_data()
        
        # Test for email segment
        pipeline.create_time_series(
            intervention_date='2024-04-10',
            segment_col='channel',
            segment_val='email'
        )
        
        analysis_data = pipeline.get_analysis_series(metric='revenue_usd')
        analyzer = CausalAnalyzer(
            analysis_data, 
            config=config,
            segment=('channel', 'email')
        )
        analyzer.run_causal_impact()
        
        metrics = analyzer.get_impact_metrics()
        
        assert metrics['segment'] == 'channel:email'


class TestDatabaseIntegration:
    """Integration tests for database operations."""
    
    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create a temporary database."""
        db_path = tmp_path / 'test.db'
        db = DatabaseManager(f"sqlite:///{db_path}")
        db.initialize()
        return db
    
    def test_save_and_retrieve_analysis(self, temp_db):
        """Test saving and retrieving analysis results."""
        db = temp_db
        
        # Save analysis
        analysis_id = db.save_analysis(
            segment='test_segment',
            intervention_date='2024-03-01',
            campaign_cost=5000,
            impact_metrics={
                'cumulative_effect': 10000,
                'average_effect': 100,
                'p_value': 0.01
            },
            financial_results={
                'roi_percentage': 100,
                'net_profit': 5000
            }
        )
        
        assert analysis_id is not None
        
        # Retrieve
        retrieved = db.get_analysis_by_id(analysis_id)
        
        assert retrieved is not None
        assert retrieved['segment'] == 'test_segment'
        assert retrieved['cumulative_effect'] == 10000
    
    def test_analysis_history_filtering(self, temp_db):
        """Test filtering analysis history."""
        db = temp_db
        
        # Save multiple analyses
        for i, segment in enumerate(['segment_a', 'segment_b', 'segment_a']):
            db.save_analysis(
                segment=segment,
                intervention_date='2024-03-01',
                campaign_cost=5000,
                impact_metrics={'cumulative_effect': i * 1000, 'average_effect': 0, 'p_value': 0.05},
                financial_results={'roi_percentage': 0, 'net_profit': 0}
            )
        
        # Filter by segment
        segment_a_results = db.get_analysis_history(segment='segment_a')
        
        assert len(segment_a_results) == 2
        
    def test_summary_stats(self, temp_db):
        """Test summary statistics calculation."""
        db = temp_db
        
        # Save some analyses
        for i in range(5):
            db.save_analysis(
                segment=f'segment_{i}',
                intervention_date='2024-03-01',
                campaign_cost=5000,
                impact_metrics={'cumulative_effect': i * 100, 'average_effect': 0, 'p_value': 0.01},
                financial_results={'roi_percentage': i * 10, 'net_profit': 0}
            )
        
        stats = db.get_summary_stats()
        
        assert stats['total_analyses'] == 5
        assert stats['unique_segments'] == 5


class TestExperimentTracking:
    """Integration tests for experiment tracking."""
    
    @pytest.fixture
    def temp_tracker(self, tmp_path):
        """Create a temporary tracker."""
        return ExperimentTracker(str(tmp_path / 'experiments'))
    
    def test_experiment_lifecycle(self, temp_tracker):
        """Test full experiment lifecycle."""
        tracker = temp_tracker
        
        # Create experiment
        tracker.create_experiment('test_exp', 'Test experiment')
        
        # Start run
        with tracker.start_run('test_exp', 'run_001') as run:
            run.log_param('param1', 'value1')
            run.log_param('param2', 42)
            run.log_metric('metric1', 0.95)
            run.log_metric('metric2', 100.0)
        
        # Verify run was saved
        runs = tracker.list_runs('test_exp')
        assert len(runs) == 1
        assert runs[0]['run_name'] == 'run_001'
        assert runs[0]['status'] == 'completed'
    
    def test_compare_runs(self, temp_tracker):
        """Test run comparison."""
        tracker = temp_tracker
        
        # Create multiple runs
        for i in range(3):
            with tracker.start_run('comparison_exp', f'run_{i}') as run:
                run.log_param('iteration', i)
                run.log_metric('accuracy', 0.9 + i * 0.02)
                run.log_metric('loss', 0.1 - i * 0.01)
        
        # Compare
        comparison = tracker.compare_runs('comparison_exp')
        
        assert len(comparison) == 3
        assert 'accuracy' in comparison.columns
    
    def test_get_best_run(self, temp_tracker):
        """Test finding best run."""
        tracker = temp_tracker
        
        # Create runs with different metrics
        scores = [0.7, 0.9, 0.8]
        for i, score in enumerate(scores):
            with tracker.start_run('best_run_exp', f'run_{i}') as run:
                run.log_metric('score', score)
        
        best = tracker.get_best_run('best_run_exp', 'score', minimize=False)
        
        assert best is not None
        assert best.metrics['score'] == 0.9


class TestAdvancedAnalytics:
    """Integration tests for new advanced analytics modules."""
    
    def test_propensity_matching_integration(self):
        """Test propensity score matching module."""
        from propensity_matching import PropensityMatcher
        
        np.random.seed(42)
        n = 200
        
        data = pd.DataFrame({
            'treatment': np.random.binomial(1, 0.3, n),
            'age': np.random.normal(40, 10, n),
            'income': np.random.lognormal(10, 0.5, n),
            'outcome': np.random.normal(100, 20, n)
        })
        
        matcher = PropensityMatcher(
            data=data,
            treatment_col='treatment',
            covariates=['age', 'income'],
            outcome_col='outcome'
        )
        
        matcher.fit()
        matched = matcher.match(caliper=0.2)
        
        assert matched is not None
        assert len(matched) > 0
    
    def test_diff_in_diff_integration(self):
        """Test difference-in-differences module."""
        from diff_in_diff import DifferenceInDifferences
        
        np.random.seed(42)
        n = 100
        
        data = pd.DataFrame({
            'unit': np.repeat(range(n), 2),
            'treatment': np.tile([0, 1] * (n // 2), 2),
            'post': np.repeat([0, 1], n),
            'outcome': np.random.normal(100, 10, 2 * n)
        })
        
        # Add treatment effect
        data.loc[(data['treatment'] == 1) & (data['post'] == 1), 'outcome'] += 20
        
        did = DifferenceInDifferences(
            data=data,
            outcome='outcome',
            treatment='treatment',
            time='post'
        )
        
        results = did.estimate()
        
        assert results is not None
        assert hasattr(results, 'ate')


class TestAPIIntegration:
    """Integration tests for FastAPI endpoints."""
    
    @pytest.fixture
    def test_client(self):
        """Create test client for API."""
        try:
            from fastapi.testclient import TestClient
            from api import app
            return TestClient(app)
        except ImportError:
            pytest.skip("FastAPI test client not available")
    
    def test_health_endpoint(self, test_client):
        """Test health check endpoint."""
        response = test_client.get('/health')
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
    
    def test_root_endpoint(self, test_client):
        """Test root endpoint."""
        response = test_client.get('/')
        
        assert response.status_code == 200
        data = response.json()
        assert 'name' in data
        assert 'version' in data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
