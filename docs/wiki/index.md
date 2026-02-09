# 📚 Causal Impact Analysis - Code Wiki

> Comprehensive documentation for the Causal Impact & Investment Decision Analysis project.

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        User Interface                             │
├─────────────────┬────────────────────┬───────────────────────────┤
│  Dashboard      │    REST API        │    Jupyter Notebooks      │
│ (Streamlit)     │   (FastAPI)        │                           │
├─────────────────┴────────────────────┴───────────────────────────┤
│                     Analysis Engine                               │
├──────────┬──────────┬──────────┬──────────┬──────────────────────┤
│  Causal  │  Causal  │  Double  │  DiD /   │   Mediation /       │
│  Impact  │  Forest  │   ML     │  RDD     │   Propensity        │
├──────────┴──────────┴──────────┴──────────┴──────────────────────┤
│                   Infrastructure Layer                            │
├──────────┬──────────┬──────────┬──────────┬──────────────────────┤
│ Database │ Scheduler│  Cloud   │  Drift   │   Experiment        │
│          │          │ Connect  │ Detection│   Tracking          │
└──────────┴──────────┴──────────┴──────────┴──────────────────────┘
```

---

## 📖 Module Reference

### Core Analysis
| Module | Purpose | Key Classes |
|--------|---------|-------------|
| [causal_analysis.py](modules/causal_analysis.md) | BSTS causal inference | `CausalImpactAnalysis` |
| [causal_forest.py](modules/causal_forest.md) | Heterogeneous effects | `CausalForest` |
| [double_ml.py](modules/double_ml.md) | Debiased ML | `DoubleML` |

### Statistical Methods
| Module | Purpose | Key Classes |
|--------|---------|-------------|
| [propensity_matching.py](modules/propensity_matching.md) | Propensity Score Matching | `PropensityMatcher` |
| [diff_in_diff.py](modules/diff_in_diff.md) | Difference-in-Differences | `DifferenceInDifferences` |
| [regression_discontinuity.py](modules/regression_discontinuity.md) | RDD Analysis | `RegressionDiscontinuity` |
| [mediation_analysis.py](modules/mediation_analysis.md) | Mediation Effects | `MediationAnalysis` |

### Production
| Module | Purpose | Key Classes |
|--------|---------|-------------|
| [api.py](modules/api.md) | REST API endpoints | FastAPI app |
| [database.py](modules/database.md) | Data persistence | `AnalysisDatabase` |
| [scheduler.py](modules/scheduler.md) | Job scheduling | `AnalysisScheduler` |
| [experiment_tracker.py](modules/experiment_tracker.md) | Experiment logging | `ExperimentTracker` |

### Monitoring
| Module | Purpose | Key Classes |
|--------|---------|-------------|
| [anomaly_detection.py](modules/anomaly_detection.md) | Anomaly detection | `AnomalyDetector` |
| [drift_detection.py](modules/drift_detection.md) | Drift monitoring | `DriftDetector` |

### Utilities
| Module | Purpose |
|--------|---------|
| [data_pipeline.py](modules/data_pipeline.md) | ETL processing |
| [financial_analysis.py](modules/financial_analysis.md) | Financial metrics |
| [cloud_connectors.py](modules/cloud_connectors.md) | Cloud storage |
| [data_upload.py](modules/data_upload.md) | File uploads |

---

## 🚀 Quick Start

### 1. Basic Causal Impact Analysis
```python
from src.causal_analysis import CausalImpactAnalysis

# Initialize
analyzer = CausalImpactAnalysis()

# Run analysis
results = analyzer.run(
    data=df,
    intervention_date='2024-03-01',
    pre_period=['2024-01-01', '2024-02-28'],
    post_period=['2024-03-01', '2024-03-31']
)

print(f"Effect: ${results.cumulative_effect:,.2f}")
print(f"P-value: {results.p_value:.4f}")
```

### 2. Heterogeneous Treatment Effects
```python
from src.causal_forest import CausalForest

cf = CausalForest(n_estimators=100)
cf.fit(X, treatment, outcome)

# Get individual treatment effects
cates = cf.predict(X_new)

# Find groups with different effects
groups = cf.get_heterogeneity_groups(X_new, n_groups=4)
```

### 3. Production API Usage
```bash
# Start API server
uvicorn src.api:app --reload

# Run analysis via API
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"intervention_date": "2024-03-01"}'
```

---

## 📊 Data Flow

```
Input Data (CSV/Parquet)
       │
       ▼
┌─────────────────┐
│  Data Pipeline  │  ← Validation, cleaning, aggregation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Analysis Engine │  ← BSTS, Causal Forest, Double ML
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Results      │  ← Effects, p-values, CIs
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌──────┐  ┌──────┐
│ API  │  │Report│
└──────┘  └──────┘
```

---

## 🔗 See Also

- [Getting Started Guide](getting-started.md)
- [API Reference](api-reference.md)
- [Configuration Guide](configuration.md)
- [Deployment Guide](deployment.md)
