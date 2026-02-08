"""
Model Drift Detection Module
============================

Detects drift in model performance and data distributions over time.
Alerts when models may need retraining or when data quality issues arise.

Types of drift detected:
- Data drift: Input feature distribution changes
- Concept drift: Relationship between features and target changes
- Model performance drift: Accuracy/metrics degradation

Author: Causal Impact Analysis Project
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
import warnings


@dataclass
class DriftResult:
    """Result of a drift detection test."""
    drift_detected: bool
    drift_type: str
    drift_score: float
    p_value: Optional[float]
    threshold: float
    details: Dict[str, Any]
    timestamp: str


@dataclass
class DriftReport:
    """Comprehensive drift report."""
    overall_drift: bool
    data_drift: Dict[str, DriftResult]
    performance_drift: Optional[DriftResult]
    recommendations: List[str]
    timestamp: str


class DriftDetector:
    """
    Detects various types of drift in ML models and data.
    
    Supports:
    - Statistical tests for data drift (KS, Chi-squared, PSI)
    - Performance monitoring for model drift
    - Concept drift detection
    - Automated alerting
    
    Example:
        >>> detector = DriftDetector()
        >>> detector.set_reference(reference_data)
        >>> report = detector.detect(new_data, predictions, actuals)
    """
    
    def __init__(
        self,
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.05,
        performance_threshold: float = 0.1,
        window_size: int = 100
    ):
        """
        Initialize drift detector.
        
        Args:
            psi_threshold: PSI threshold for data drift (>0.2 = significant)
            ks_threshold: KS test p-value threshold
            performance_threshold: Relative performance degradation threshold
            window_size: Rolling window size for performance monitoring
        """
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.performance_threshold = performance_threshold
        self.window_size = window_size
        
        self._reference_data: Optional[pd.DataFrame] = None
        self._reference_stats: Dict[str, Dict] = {}
        self._baseline_performance: Optional[float] = None
        self._performance_history: List[Tuple[datetime, float]] = []
        self._feature_names: List[str] = []
    
    def set_reference(
        self,
        data: pd.DataFrame,
        target: Optional[pd.Series] = None,
        predictions: Optional[pd.Series] = None
    ):
        """
        Set reference (training) data for drift comparison.
        
        Args:
            data: Reference feature data
            target: Reference target values
            predictions: Reference model predictions
        """
        self._reference_data = data.copy()
        self._feature_names = list(data.columns)
        
        # Compute reference statistics for each feature
        for col in data.columns:
            self._reference_stats[col] = {
                'mean': data[col].mean(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max(),
                'quantiles': data[col].quantile([0.25, 0.5, 0.75]).values,
                'histogram': np.histogram(data[col].dropna(), bins=10)
            }
        
        # Set baseline performance if provided
        if target is not None and predictions is not None:
            self._baseline_performance = r2_score(target, predictions)
            self._performance_history.append(
                (datetime.now(), self._baseline_performance)
            )
    
    def compute_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Compute Population Stability Index (PSI).
        
        PSI interpretation:
        - < 0.1: No significant change
        - 0.1 - 0.2: Moderate change, monitor
        - > 0.2: Significant change, action required
        """
        # Create bins from reference data
        _, bin_edges = np.histogram(reference, bins=n_bins)
        
        # Count observations in each bin
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        curr_counts, _ = np.histogram(current, bins=bin_edges)
        
        # Convert to proportions (add small epsilon to avoid division by zero)
        epsilon = 1e-6
        ref_props = (ref_counts + epsilon) / (len(reference) + n_bins * epsilon)
        curr_props = (curr_counts + epsilon) / (len(current) + n_bins * epsilon)
        
        # Compute PSI
        psi = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))
        
        return psi
    
    def detect_data_drift(
        self,
        current_data: pd.DataFrame,
        method: str = 'psi'
    ) -> Dict[str, DriftResult]:
        """
        Detect data drift for each feature.
        
        Args:
            current_data: Current data to compare against reference
            method: Detection method ('psi', 'ks', 'chi2')
        
        Returns:
            Dict mapping feature names to drift results
        """
        if self._reference_data is None:
            raise ValueError("Reference data not set. Call set_reference() first.")
        
        results = {}
        timestamp = datetime.now().isoformat()
        
        for col in self._feature_names:
            if col not in current_data.columns:
                continue
            
            ref_values = self._reference_data[col].dropna().values
            curr_values = current_data[col].dropna().values
            
            if len(curr_values) == 0:
                continue
            
            if method == 'psi':
                score = self.compute_psi(ref_values, curr_values)
                drift_detected = score > self.psi_threshold
                p_value = None
                
            elif method == 'ks':
                stat, p_value = stats.ks_2samp(ref_values, curr_values)
                score = stat
                drift_detected = p_value < self.ks_threshold
                
            elif method == 'chi2':
                # Use Chi-squared for categorical-like data
                n_bins = min(10, len(np.unique(ref_values)))
                ref_hist, bins = np.histogram(ref_values, bins=n_bins)
                curr_hist, _ = np.histogram(curr_values, bins=bins)
                
                # Normalize to expected frequencies
                ref_hist = ref_hist * len(curr_values) / len(ref_values)
                
                stat, p_value = stats.chisquare(curr_hist, f_exp=ref_hist + 1e-6)
                score = stat
                drift_detected = p_value < self.ks_threshold
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            results[col] = DriftResult(
                drift_detected=drift_detected,
                drift_type='data_drift',
                drift_score=float(score),
                p_value=float(p_value) if p_value is not None else None,
                threshold=self.psi_threshold if method == 'psi' else self.ks_threshold,
                details={
                    'method': method,
                    'ref_mean': float(self._reference_stats[col]['mean']),
                    'curr_mean': float(np.mean(curr_values)),
                    'ref_std': float(self._reference_stats[col]['std']),
                    'curr_std': float(np.std(curr_values))
                },
                timestamp=timestamp
            )
        
        return results
    
    def detect_performance_drift(
        self,
        actuals: np.ndarray,
        predictions: np.ndarray,
        metric: str = 'r2'
    ) -> DriftResult:
        """
        Detect drift in model performance.
        
        Args:
            actuals: Actual target values
            predictions: Model predictions
            metric: Performance metric ('r2', 'mse', 'rmse')
        
        Returns:
            DriftResult for performance drift
        """
        timestamp = datetime.now().isoformat()
        
        # Compute current performance
        if metric == 'r2':
            current_perf = r2_score(actuals, predictions)
        elif metric == 'mse':
            current_perf = -mean_squared_error(actuals, predictions)  # Negative so higher is better
        elif metric == 'rmse':
            current_perf = -np.sqrt(mean_squared_error(actuals, predictions))
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Record in history
        self._performance_history.append((datetime.now(), current_perf))
        
        # Compare with baseline
        if self._baseline_performance is None:
            drift_detected = False
            relative_change = 0.0
        else:
            relative_change = (self._baseline_performance - current_perf) / abs(self._baseline_performance + 1e-6)
            drift_detected = relative_change > self.performance_threshold
        
        return DriftResult(
            drift_detected=drift_detected,
            drift_type='performance_drift',
            drift_score=float(relative_change),
            p_value=None,
            threshold=self.performance_threshold,
            details={
                'metric': metric,
                'baseline_performance': float(self._baseline_performance) if self._baseline_performance else None,
                'current_performance': float(current_perf),
                'relative_change': float(relative_change),
                'n_samples': len(actuals)
            },
            timestamp=timestamp
        )
    
    def detect(
        self,
        current_data: pd.DataFrame,
        predictions: Optional[np.ndarray] = None,
        actuals: Optional[np.ndarray] = None,
        method: str = 'psi'
    ) -> DriftReport:
        """
        Run comprehensive drift detection.
        
        Args:
            current_data: Current feature data
            predictions: Current model predictions
            actuals: Current actual values
            method: Data drift detection method
        
        Returns:
            Comprehensive DriftReport
        """
        timestamp = datetime.now().isoformat()
        
        # Detect data drift
        data_drift_results = self.detect_data_drift(current_data, method=method)
        
        # Detect performance drift if data available
        perf_drift_result = None
        if predictions is not None and actuals is not None:
            perf_drift_result = self.detect_performance_drift(actuals, predictions)
        
        # Determine overall drift status
        features_with_drift = [f for f, r in data_drift_results.items() if r.drift_detected]
        has_data_drift = len(features_with_drift) > 0
        has_perf_drift = perf_drift_result.drift_detected if perf_drift_result else False
        overall_drift = has_data_drift or has_perf_drift
        
        # Generate recommendations
        recommendations = []
        if has_data_drift:
            recommendations.append(
                f"Data drift detected in {len(features_with_drift)} features: {features_with_drift}. "
                "Consider investigating data quality or updating the model."
            )
        if has_perf_drift:
            recommendations.append(
                "Model performance has degraded significantly. "
                "Consider retraining the model with recent data."
            )
        if not overall_drift:
            recommendations.append("No significant drift detected. Continue monitoring.")
        
        return DriftReport(
            overall_drift=overall_drift,
            data_drift=data_drift_results,
            performance_drift=perf_drift_result,
            recommendations=recommendations,
            timestamp=timestamp
        )
    
    def get_performance_history(self) -> pd.DataFrame:
        """Get performance history as DataFrame."""
        if not self._performance_history:
            return pd.DataFrame(columns=['timestamp', 'performance'])
        
        data = [{'timestamp': ts, 'performance': perf} 
                for ts, perf in self._performance_history]
        return pd.DataFrame(data)


class CausalModelMonitor:
    """
    Monitor causal impact models specifically.
    
    Tracks:
    - Treatment effect stability
    - Covariate balance over time
    - Statistical significance trends
    """
    
    def __init__(self, window_size: int = 10):
        """Initialize causal model monitor."""
        self.window_size = window_size
        self._effect_history: List[Dict[str, Any]] = []
        self._balance_history: List[Dict[str, Any]] = []
    
    def record_analysis(
        self,
        effect: float,
        effect_se: float,
        p_value: float,
        segment: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """Record an analysis result."""
        self._effect_history.append({
            'timestamp': datetime.now().isoformat(),
            'effect': effect,
            'effect_se': effect_se,
            'p_value': p_value,
            'segment': segment,
            'metadata': metadata or {}
        })
    
    def check_effect_stability(self) -> Dict[str, Any]:
        """Check if treatment effects are stable over time."""
        if len(self._effect_history) < 3:
            return {'stable': True, 'message': 'Insufficient history'}
        
        recent = self._effect_history[-self.window_size:]
        effects = [r['effect'] for r in recent]
        
        mean_effect = np.mean(effects)
        std_effect = np.std(effects)
        cv = std_effect / abs(mean_effect) if mean_effect != 0 else float('inf')
        
        # Effect is unstable if CV > 0.5
        stable = cv < 0.5
        
        return {
            'stable': stable,
            'mean_effect': mean_effect,
            'std_effect': std_effect,
            'coefficient_of_variation': cv,
            'n_observations': len(recent),
            'message': 'Effect stable' if stable else 'Effect shows high variability'
        }
    
    def check_significance_trend(self) -> Dict[str, Any]:
        """Check trend in statistical significance."""
        if len(self._effect_history) < 3:
            return {'trend': 'insufficient_data'}
        
        recent = self._effect_history[-self.window_size:]
        p_values = [r['p_value'] for r in recent]
        
        # Check if p-values are trending up (losing significance)
        if len(p_values) >= 5:
            correlation = np.corrcoef(range(len(p_values)), p_values)[0, 1]
            if correlation > 0.5:
                trend = 'decreasing_significance'
            elif correlation < -0.5:
                trend = 'increasing_significance'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'mean_p_value': np.mean(p_values),
            'latest_p_value': p_values[-1],
            'significant_rate': np.mean([p < 0.05 for p in p_values])
        }
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary of recorded analyses."""
        return pd.DataFrame(self._effect_history)


def main():
    """Demo drift detection."""
    print("=" * 60)
    print("DRIFT DETECTION DEMO")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate reference data
    n_ref = 1000
    reference = pd.DataFrame({
        'feature_1': np.random.normal(100, 15, n_ref),
        'feature_2': np.random.uniform(0, 50, n_ref),
        'feature_3': np.random.exponential(10, n_ref)
    })
    
    # Generate target and predictions for reference
    ref_target = reference['feature_1'] * 0.5 + reference['feature_2'] * 0.3 + np.random.randn(n_ref) * 5
    ref_predictions = ref_target + np.random.randn(n_ref) * 2  # Good predictions
    
    # Initialize detector
    detector = DriftDetector(psi_threshold=0.2)
    detector.set_reference(reference, target=ref_target, predictions=ref_predictions)
    
    print("\n1. Testing with SIMILAR data (no drift expected):")
    print("-" * 40)
    
    # Generate similar current data
    n_curr = 500
    current_similar = pd.DataFrame({
        'feature_1': np.random.normal(100, 15, n_curr),
        'feature_2': np.random.uniform(0, 50, n_curr),
        'feature_3': np.random.exponential(10, n_curr)
    })
    
    report = detector.detect(current_similar)
    print(f"Overall drift detected: {report.overall_drift}")
    for feature, result in report.data_drift.items():
        status = "⚠️ DRIFT" if result.drift_detected else "✓ OK"
        print(f"  {feature}: {status} (PSI={result.drift_score:.4f})")
    
    print("\n2. Testing with DRIFTED data:")
    print("-" * 40)
    
    # Generate drifted data
    current_drifted = pd.DataFrame({
        'feature_1': np.random.normal(120, 20, n_curr),  # Mean shifted
        'feature_2': np.random.uniform(10, 60, n_curr),  # Range shifted
        'feature_3': np.random.exponential(10, n_curr)   # No change
    })
    
    report = detector.detect(current_drifted)
    print(f"Overall drift detected: {report.overall_drift}")
    for feature, result in report.data_drift.items():
        status = "⚠️ DRIFT" if result.drift_detected else "✓ OK"
        print(f"  {feature}: {status} (PSI={result.drift_score:.4f})")
    
    print("\nRecommendations:")
    for rec in report.recommendations:
        print(f"  • {rec}")
    
    print("\n3. Testing Causal Model Monitor:")
    print("-" * 40)
    
    monitor = CausalModelMonitor()
    
    # Simulate analysis history
    for i in range(10):
        effect = 5000 + np.random.randn() * 500
        monitor.record_analysis(
            effect=effect,
            effect_se=500,
            p_value=0.01 + np.random.rand() * 0.03,
            segment=f'segment_{i % 3}'
        )
    
    stability = monitor.check_effect_stability()
    print(f"Effect stability: {stability['message']}")
    print(f"  Mean effect: {stability['mean_effect']:.2f}")
    print(f"  CV: {stability['coefficient_of_variation']:.3f}")
    
    significance = monitor.check_significance_trend()
    print(f"Significance trend: {significance['trend']}")
    print(f"  Significant rate: {significance['significant_rate']:.1%}")
    
    print("\n✓ Drift detection demo completed!")


if __name__ == '__main__':
    main()
