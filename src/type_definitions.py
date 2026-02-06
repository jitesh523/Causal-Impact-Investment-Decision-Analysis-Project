"""
Type Stubs and Common Types
Provides type aliases and custom types for the project
"""

from typing import TypedDict, Dict, List, Tuple, Optional, Any, Union
import pandas as pd
import numpy as np
from numpy.typing import NDArray


# Type aliases
DateRange = Tuple[int, int]
SegmentInfo = Optional[Tuple[str, Any]]


class AnalysisData(TypedDict):
    """Structure for analysis data passed to CausalAnalyzer"""
    data: pd.DataFrame
    y: NDArray[np.float64]
    X: NDArray[np.float64]
    dates: pd.DatetimeIndex
    pre_period: DateRange
    post_period: DateRange


class ImpactMetrics(TypedDict):
    """Structure for impact metrics returned by CausalAnalyzer"""
    segment: str
    average_actual: float
    average_predicted: float
    average_effect: float
    average_effect_pct: float
    cumulative_actual: float
    cumulative_predicted: float
    cumulative_effect: float
    cumulative_effect_pct: float
    p_value: float
    t_statistic: float
    alpha: float


class FinancialResults(TypedDict):
    """Structure for financial analysis results"""
    segment: str
    cumulative_revenue_impact: float
    average_daily_revenue_impact: float
    campaign_cost: float
    net_profit: float
    roi_percentage: float
    roi_ratio: float
    statistical_significance: float
    is_significant: bool


class ConfigDict(TypedDict, total=False):
    """Structure for configuration dictionary"""
    data: Dict[str, str]
    dates: Dict[str, str]
    campaign: Dict[str, float]
    model: Dict[str, float]
    segments: List[str]


# Predictions dictionary type
class Predictions(TypedDict):
    """Structure for model predictions"""
    actual: NDArray[np.float64]
    predicted: NDArray[np.float64]
    predicted_lower: NDArray[np.float64]
    predicted_upper: NDArray[np.float64]
    std: NDArray[np.float64]
    point_effect: NDArray[np.float64]
    point_effect_lower: NDArray[np.float64]
    point_effect_upper: NDArray[np.float64]
