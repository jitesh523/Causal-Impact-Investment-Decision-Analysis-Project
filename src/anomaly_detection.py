"""
Anomaly Detection Module
========================

Detects anomalies in time series data and analysis metrics.
Useful for identifying unusual patterns that may indicate
data quality issues or significant events.

Author: Causal Impact Analysis Project
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


@dataclass
class Anomaly:
    """Represents a detected anomaly."""
    timestamp: datetime
    value: float
    anomaly_score: float
    method: str
    severity: str  # 'low', 'medium', 'high'
    context: Dict[str, Any]


@dataclass
class AnomalyReport:
    """Report of anomaly detection results."""
    n_anomalies: int
    anomaly_rate: float
    anomalies: List[Anomaly]
    summary: Dict[str, Any]
    

class AnomalyDetector:
    """
    Multi-method anomaly detection for time series and metrics.
    
    Supports multiple detection methods:
    - Z-Score: Statistical outlier detection
    - IQR: Interquartile range method
    - Isolation Forest: Tree-based anomaly detection
    - LOF: Local Outlier Factor
    - Rolling Statistics: Detect deviations from rolling mean/std
    
    Example:
        >>> detector = AnomalyDetector(method='isolation_forest')
        >>> detector.fit(historical_data)
        >>> anomalies = detector.detect(new_data)
    """
    
    METHODS = ['zscore', 'iqr', 'isolation_forest', 'lof', 'rolling']
    
    def __init__(
        self,
        method: str = 'zscore',
        threshold: float = 3.0,
        contamination: float = 0.05,
        window_size: int = 7,
        min_samples: int = 20
    ):
        """
        Initialize anomaly detector.
        
        Args:
            method: Detection method
            threshold: Threshold for statistical methods (z-score, rolling)
            contamination: Expected proportion of anomalies (for ML methods)
            window_size: Window size for rolling statistics
            min_samples: Minimum samples required for detection
        """
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}")
        
        self.method = method
        self.threshold = threshold
        self.contamination = contamination
        self.window_size = window_size
        self.min_samples = min_samples
        
        self._model = None
        self._scaler = StandardScaler()
        self._fitted_mean = None
        self._fitted_std = None
        self._fitted_q1 = None
        self._fitted_q3 = None
        self._is_fitted = False
    
    def fit(
        self,
        data: Union[pd.Series, np.ndarray],
        timestamps: Optional[pd.DatetimeIndex] = None
    ) -> 'AnomalyDetector':
        """
        Fit detector on historical data.
        
        Args:
            data: Time series values
            timestamps: Optional timestamps
        
        Returns:
            Self
        """
        if isinstance(data, pd.Series):
            values = data.values.reshape(-1, 1)
        else:
            values = np.asarray(data).reshape(-1, 1)
        
        if len(values) < self.min_samples:
            raise ValueError(f"Need at least {self.min_samples} samples, got {len(values)}")
        
        # Fit scaler
        self._scaler.fit(values)
        scaled = self._scaler.transform(values).flatten()
        
        # Method-specific fitting
        if self.method == 'zscore':
            self._fitted_mean = np.mean(scaled)
            self._fitted_std = np.std(scaled)
        
        elif self.method == 'iqr':
            self._fitted_q1 = np.percentile(scaled, 25)
            self._fitted_q3 = np.percentile(scaled, 75)
        
        elif self.method == 'isolation_forest':
            self._model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
            self._model.fit(values)
        
        elif self.method == 'lof':
            self._model = LocalOutlierFactor(
                n_neighbors=min(20, len(values) - 1),
                contamination=self.contamination,
                novelty=True
            )
            self._model.fit(values)
        
        elif self.method == 'rolling':
            self._fitted_mean = np.mean(scaled)
            self._fitted_std = np.std(scaled)
        
        self._is_fitted = True
        return self
    
    def detect(
        self,
        data: Union[pd.Series, np.ndarray],
        timestamps: Optional[Union[pd.DatetimeIndex, List]] = None
    ) -> AnomalyReport:
        """
        Detect anomalies in data.
        
        Args:
            data: Time series values
            timestamps: Optional timestamps
        
        Returns:
            AnomalyReport with detected anomalies
        """
        if not self._is_fitted:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        if isinstance(data, pd.Series):
            if timestamps is None and isinstance(data.index, pd.DatetimeIndex):
                timestamps = data.index
            values = data.values.reshape(-1, 1)
        else:
            values = np.asarray(data).reshape(-1, 1)
        
        n = len(values)
        
        if timestamps is None:
            timestamps = pd.date_range(start='2024-01-01', periods=n, freq='D')
        
        # Scale data
        scaled = self._scaler.transform(values).flatten()
        
        # Detect based on method
        anomaly_mask, scores = self._detect_method(values, scaled)
        
        # Create anomaly objects
        anomalies = []
        for i in np.where(anomaly_mask)[0]:
            severity = self._get_severity(scores[i])
            
            anomaly = Anomaly(
                timestamp=pd.Timestamp(timestamps[i]),
                value=float(values[i, 0]),
                anomaly_score=float(scores[i]),
                method=self.method,
                severity=severity,
                context={
                    'index': i,
                    'scaled_value': float(scaled[i]),
                    'threshold': self.threshold
                }
            )
            anomalies.append(anomaly)
        
        # Create report
        report = AnomalyReport(
            n_anomalies=len(anomalies),
            anomaly_rate=len(anomalies) / n,
            anomalies=anomalies,
            summary={
                'method': self.method,
                'threshold': self.threshold,
                'total_samples': n,
                'high_severity': sum(1 for a in anomalies if a.severity == 'high'),
                'medium_severity': sum(1 for a in anomalies if a.severity == 'medium'),
                'low_severity': sum(1 for a in anomalies if a.severity == 'low')
            }
        )
        
        return report
    
    def _detect_method(
        self,
        values: np.ndarray,
        scaled: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply detection method."""
        n = len(scaled)
        
        if self.method == 'zscore':
            z_scores = np.abs((scaled - self._fitted_mean) / self._fitted_std)
            anomaly_mask = z_scores > self.threshold
            scores = z_scores
        
        elif self.method == 'iqr':
            iqr = self._fitted_q3 - self._fitted_q1
            lower = self._fitted_q1 - 1.5 * iqr
            upper = self._fitted_q3 + 1.5 * iqr
            anomaly_mask = (scaled < lower) | (scaled > upper)
            scores = np.maximum(
                np.abs(scaled - lower) / iqr,
                np.abs(scaled - upper) / iqr
            )
        
        elif self.method == 'isolation_forest':
            predictions = self._model.predict(values)
            anomaly_mask = predictions == -1
            scores = -self._model.score_samples(values)  # Higher = more anomalous
        
        elif self.method == 'lof':
            predictions = self._model.predict(values)
            anomaly_mask = predictions == -1
            scores = -self._model.score_samples(values)
        
        elif self.method == 'rolling':
            # Rolling window detection
            rolling_mean = pd.Series(scaled).rolling(self.window_size, min_periods=1).mean().values
            rolling_std = pd.Series(scaled).rolling(self.window_size, min_periods=1).std().values
            rolling_std = np.clip(rolling_std, 0.1, None)  # Avoid division by zero
            
            z_scores = np.abs((scaled - rolling_mean) / rolling_std)
            anomaly_mask = z_scores > self.threshold
            scores = z_scores
        
        return anomaly_mask, scores
    
    def _get_severity(self, score: float) -> str:
        """Determine anomaly severity based on score."""
        if self.method in ['isolation_forest', 'lof']:
            if score > 0.7:
                return 'high'
            elif score > 0.5:
                return 'medium'
            return 'low'
        else:
            if score > self.threshold * 1.5:
                return 'high'
            elif score > self.threshold:
                return 'medium'
            return 'low'
    
    def fit_detect(
        self,
        data: Union[pd.Series, np.ndarray],
        timestamps: Optional[pd.DatetimeIndex] = None
    ) -> AnomalyReport:
        """Fit and detect in one step."""
        self.fit(data, timestamps)
        return self.detect(data, timestamps)


class MetricAnomalyMonitor:
    """
    Monitor analysis metrics for anomalies.
    
    Tracks metrics over time and alerts when unusual values are detected.
    """
    
    def __init__(
        self,
        metrics: List[str],
        sensitivity: str = 'medium'
    ):
        """
        Initialize metric monitor.
        
        Args:
            metrics: List of metric names to monitor
            sensitivity: Detection sensitivity ('low', 'medium', 'high')
        """
        self.metrics = metrics
        self.sensitivity = sensitivity
        
        thresholds = {'low': 4.0, 'medium': 3.0, 'high': 2.0}
        self.threshold = thresholds.get(sensitivity, 3.0)
        
        self._history: Dict[str, List[Tuple[datetime, float]]] = {m: [] for m in metrics}
        self._detectors: Dict[str, AnomalyDetector] = {}
    
    def record(self, metric: str, value: float, timestamp: Optional[datetime] = None):
        """Record a metric value."""
        if metric not in self.metrics:
            self.metrics.append(metric)
            self._history[metric] = []
        
        timestamp = timestamp or datetime.now()
        self._history[metric].append((timestamp, value))
    
    def record_batch(self, metrics: Dict[str, float], timestamp: Optional[datetime] = None):
        """Record multiple metrics at once."""
        for metric, value in metrics.items():
            self.record(metric, value, timestamp)
    
    def check_anomalies(self, min_history: int = 10) -> Dict[str, Optional[Anomaly]]:
        """
        Check all metrics for anomalies.
        
        Returns:
            Dict mapping metric names to detected anomalies (or None)
        """
        results = {}
        
        for metric in self.metrics:
            history = self._history.get(metric, [])
            
            if len(history) < min_history:
                results[metric] = None
                continue
            
            timestamps, values = zip(*history)
            values = np.array(values)
            
            # Fit detector if needed
            if metric not in self._detectors:
                self._detectors[metric] = AnomalyDetector(
                    method='zscore',
                    threshold=self.threshold
                )
                self._detectors[metric].fit(values[:-1])  # Fit on all but last
            
            # Check latest value
            report = self._detectors[metric].detect(
                values[-1:],
                timestamps=list(timestamps)[-1:]
            )
            
            if report.n_anomalies > 0:
                results[metric] = report.anomalies[0]
            else:
                results[metric] = None
            
            # Update detector periodically
            if len(history) % 50 == 0:
                self._detectors[metric].fit(values)
        
        return results
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary of monitored metrics."""
        summary = []
        
        for metric in self.metrics:
            history = self._history.get(metric, [])
            if not history:
                continue
            
            values = [v for _, v in history]
            summary.append({
                'metric': metric,
                'n_observations': len(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'latest': values[-1] if values else None
            })
        
        return pd.DataFrame(summary)


def main():
    """Demo anomaly detection."""
    print("=" * 60)
    print("ANOMALY DETECTION DEMO")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate time series with anomalies
    n = 100
    dates = pd.date_range('2024-01-01', periods=n, freq='D')
    
    # Normal data with trend
    normal = 100 + np.arange(n) * 0.5 + np.random.randn(n) * 5
    
    # Inject anomalies
    anomaly_indices = [20, 45, 78]
    for idx in anomaly_indices:
        normal[idx] += np.random.choice([-1, 1]) * 30  # Large deviation
    
    data = pd.Series(normal, index=dates)
    
    print(f"\nData: {n} observations, {len(anomaly_indices)} injected anomalies")
    
    # Test different methods
    methods = ['zscore', 'iqr', 'isolation_forest', 'rolling']
    
    for method in methods:
        print(f"\n{'=' * 40}")
        print(f"Method: {method.upper()}")
        print(f"{'=' * 40}")
        
        detector = AnomalyDetector(method=method)
        report = detector.fit_detect(data, dates)
        
        print(f"Detected: {report.n_anomalies} anomalies ({report.anomaly_rate:.1%})")
        
        if report.anomalies:
            print("\nAnomalies found:")
            for a in report.anomalies[:5]:
                print(f"  {a.timestamp.strftime('%Y-%m-%d')}: "
                      f"value={a.value:.1f}, score={a.anomaly_score:.2f}, "
                      f"severity={a.severity}")
    
    # Test metric monitor
    print(f"\n{'=' * 60}")
    print("METRIC MONITORING DEMO")
    print(f"{'=' * 60}")
    
    monitor = MetricAnomalyMonitor(
        metrics=['roi', 'effect', 'p_value'],
        sensitivity='medium'
    )
    
    # Simulate metric history
    for i in range(30):
        monitor.record_batch({
            'roi': 100 + np.random.randn() * 10,
            'effect': 5000 + np.random.randn() * 500,
            'p_value': 0.02 + np.random.rand() * 0.03
        })
    
    # Add anomalous value
    monitor.record('roi', 180)  # Unusually high ROI
    
    # Check for anomalies
    anomalies = monitor.check_anomalies()
    
    print("\nMetric Summary:")
    print(monitor.get_summary().to_string(index=False))
    
    print("\nAnomaly Check:")
    for metric, anomaly in anomalies.items():
        if anomaly:
            print(f"  ⚠️  {metric}: Anomaly detected! "
                  f"(value={anomaly.value:.2f}, severity={anomaly.severity})")
        else:
            print(f"  ✓ {metric}: Normal")
    
    print("\n✓ Anomaly detection demo completed!")


if __name__ == '__main__':
    main()
