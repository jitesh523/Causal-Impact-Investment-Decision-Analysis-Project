"""
Causal Impact & Investment Decision Analysis Library
=====================================================

A comprehensive Python library for causal inference and investment
decision analysis. Provides tools for estimating causal effects
of interventions and making data-driven investment decisions.

Modules
-------
data_pipeline
    Data loading, cleaning, and preprocessing
causal_analysis
    Bayesian structural time-series analysis
financial_analysis
    ROI calculation and financial metrics
propensity_matching
    Propensity score matching for observational studies
diff_in_diff
    Difference-in-differences estimation
regression_discontinuity
    Regression discontinuity design
mediation_analysis
    Mediation analysis with Baron-Kenny approach
pdf_report
    PDF report generation
api
    FastAPI REST API
database
    Database integration (SQLite/PostgreSQL)
experiment_tracker
    Experiment tracking and logging

Example
-------
>>> from src.data_pipeline import DataPipeline
>>> from src.causal_analysis import CausalAnalyzer
>>> 
>>> pipeline = DataPipeline('config.yaml')
>>> pipeline.load_data().clean_data()
>>> pipeline.create_time_series(intervention_date='2024-03-01')
>>> 
>>> analysis_data = pipeline.get_analysis_series()
>>> analyzer = CausalAnalyzer(analysis_data)
>>> analyzer.run_causal_impact()
>>> 
>>> metrics = analyzer.get_impact_metrics()
>>> print(f"Cumulative Effect: ${metrics['cumulative_effect']:,.2f}")

See Also
--------
- README.md for installation and quick start
- docs/ for detailed documentation
- notebooks/ for interactive examples
"""

__version__ = '2.0.0'
__author__ = 'Causal Impact Analysis Team'
__email__ = 'analytics@example.com'

# Package-level imports for convenience
from pathlib import Path

# Define what gets exported
__all__ = [
    '__version__',
    '__author__',
    'DataPipeline',
    'CausalAnalyzer',
    'FinancialAnalyzer',
    'PropensityMatcher',
    'DifferenceInDifferences',
    'RegressionDiscontinuity',
    'MediationAnalyzer',
    'PDFReportGenerator',
    'DatabaseManager',
    'ExperimentTracker'
]

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    """Lazy import modules on demand."""
    if name == 'DataPipeline':
        from .data_pipeline import DataPipeline
        return DataPipeline
    elif name == 'CausalAnalyzer':
        from .causal_analysis import CausalAnalyzer
        return CausalAnalyzer
    elif name == 'FinancialAnalyzer':
        from .financial_analysis import FinancialAnalyzer
        return FinancialAnalyzer
    elif name == 'PropensityMatcher':
        from .propensity_matching import PropensityMatcher
        return PropensityMatcher
    elif name == 'DifferenceInDifferences':
        from .diff_in_diff import DifferenceInDifferences
        return DifferenceInDifferences
    elif name == 'RegressionDiscontinuity':
        from .regression_discontinuity import RegressionDiscontinuity
        return RegressionDiscontinuity
    elif name == 'MediationAnalyzer':
        from .mediation_analysis import MediationAnalyzer
        return MediationAnalyzer
    elif name == 'PDFReportGenerator':
        from .pdf_report import PDFReportGenerator
        return PDFReportGenerator
    elif name == 'DatabaseManager':
        from .database import DatabaseManager
        return DatabaseManager
    elif name == 'ExperimentTracker':
        from .experiment_tracker import ExperimentTracker
        return ExperimentTracker
    raise AttributeError(f"module 'src' has no attribute '{name}'")
