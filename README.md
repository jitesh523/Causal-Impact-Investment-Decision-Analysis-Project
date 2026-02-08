# 🚀 Causal Impact & Investment Decision Analysis

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![CI](https://github.com/jitesh523/Causal-Impact-Investment-Decision-Analysis-Project/actions/workflows/ci.yml/badge.svg)
![Coverage](https://img.shields.io/badge/Coverage-85%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

A comprehensive, end-to-end framework for measuring the **true financial impact** of marketing campaigns using **Bayesian causal inference** and **advanced machine learning**. This project moves beyond simple correlation to determine the *incremental* value driven by specific interventions.

---

## 📊 Executive Summary

This project implements a rigorous **Causal Impact Analysis** to evaluate marketing campaigns. Using **Bayesian Structural Time Series (BSTS)** modeling, we compare actual observed revenue against a counterfactual "synthetic" baseline.

### 🏆 Key Results

| Metric | Result | Interpretation |
|:---:|:---:|:---|
| **Cumulative Revenue Impact** | **\$42,137.64** | Total additional revenue generated |
| **Net Profit** | **\$37,137.64** | Profit after deducting campaign costs |
| **ROI** | **742.75%** | For every \$1 spent, returned \$7.43 |
| **Statistical Confidence** | **99.9%** | Highly significant (p < 0.0001) |

---

## ✨ Features

### 🔬 Advanced Causal Inference
| Module | Description |
|--------|-------------|
| `causal_analysis.py` | Core BSTS engine for causal impact |
| `propensity_matching.py` | Propensity Score Matching for observational studies |
| `diff_in_diff.py` | Difference-in-Differences estimation |
| `regression_discontinuity.py` | Sharp & Fuzzy RDD analysis |
| `causal_forest.py` | **NEW** Heterogeneous Treatment Effects (HTE) |
| `double_ml.py` | **NEW** Debiased ML for causal inference |
| `mediation_analysis.py` | Baron-Kenny mediation with bootstrap CIs |

### 📈 Analytics & Visualization
| Module | Description |
|--------|-------------|
| `dashboard.py` | Interactive Streamlit dashboard |
| `pdf_report.py` | Executive PDF report generation |
| `anomaly_detection.py` | Multi-method anomaly detection |
| `drift_detection.py` | **NEW** Model & data drift monitoring |
| `multi_metric_analysis.py` | Multi-metric impact analysis |
| `decay_modeling.py` | Campaign effect decay modeling |

### 🏭 Production Features
| Module | Description |
|--------|-------------|
| `api.py` | FastAPI REST API with auto-docs |
| `database.py` | SQLite/PostgreSQL integration |
| `experiment_tracker.py` | MLflow-style experiment tracking |
| `scheduler.py` | Automated analysis scheduling |
| `cloud_connectors.py` | **NEW** AWS S3/GCS/BigQuery connectors |
| `data_upload.py` | **NEW** Data upload with validation |

---

## 🚀 Quick Start

### Option A: Local Python
```bash
# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run src/dashboard.py

# Run API server
uvicorn src.api:app --reload
```

### Option B: Docker Compose (Full Stack)
```bash
# Start all services (API, DB, Dashboard, Scheduler)
docker-compose up -d

# Access:
# - Dashboard: http://localhost:8501
# - API Docs: http://localhost:8000/docs
# - Grafana: http://localhost:3000 (with monitoring profile)
```

### Option C: Docker (API Only)
```bash
docker build -t causal-impact-api .
docker run -p 8000:8000 causal-impact-api
```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/analyze` | POST | Run causal impact analysis |
| `/segments` | GET | List available segments |
| `/segments/{type}` | GET | Batch segment analysis |
| `/config` | GET | Get current configuration |

**Example:**
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"intervention_date": "2024-03-01", "segment_type": "channel", "segment_value": "email"}'
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=src --cov-report=html

# Run property-based tests
pytest tests/test_properties.py -v
```

---

## 🛠 Project Structure

```
Causal-Impact-Investment-Decision-Analysis-Project/
├── 📂 src/
│   ├── causal_analysis.py      # Core BSTS engine
│   ├── causal_forest.py        # Heterogeneous Treatment Effects
│   ├── double_ml.py            # Debiased Machine Learning
│   ├── propensity_matching.py  # Propensity Score Matching
│   ├── anomaly_detection.py    # Anomaly Detection
│   ├── scheduler.py            # Job Scheduling
│   ├── api.py                  # FastAPI REST API
│   ├── database.py             # Database Integration
│   ├── dashboard.py            # Streamlit Dashboard
│   └── ...
├── 📂 tests/
│   ├── test_integration.py     # Integration tests
│   ├── test_properties.py      # Property-based tests
│   └── ...
├── 📂 notebooks/
│   └── tutorial.ipynb          # Interactive tutorial
├── docker-compose.yml          # Full stack orchestration
├── Dockerfile                  # Container image
├── config.yaml                 # Configuration
└── requirements.txt            # Dependencies
```

---

## 📦 New in v2.0

### Advanced ML
- **Causal Forest**: Estimate heterogeneous treatment effects across subgroups
- **Double ML**: Debiased machine learning with cross-fitting

### Automation
- **Anomaly Detection**: Multi-method detection (Z-Score, IQR, Isolation Forest, LOF)
- **Scheduled Jobs**: Cron-like automation for recurring analyses

### Infrastructure
- **Docker Compose**: Full stack with API, PostgreSQL, Redis, Dashboard
- **Prometheus/Grafana**: Optional monitoring stack

---

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

---

*Built with ❤️ by the Analytics Team*
