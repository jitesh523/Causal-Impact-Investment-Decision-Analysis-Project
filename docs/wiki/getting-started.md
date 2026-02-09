# Getting Started Guide

> Complete guide to set up and use the Causal Impact Analysis framework.

---

## 📋 Prerequisites

- Python 3.9+
- pip or conda
- Git

---

## 🔧 Installation

### 1. Clone Repository
```bash
git clone https://github.com/jitesh523/Causal-Impact-Investment-Decision-Analysis-Project.git
cd Causal-Impact-Investment-Decision-Analysis-Project
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Option 1: Dashboard (Recommended for Exploration)
```bash
streamlit run src/dashboard.py
```
Open http://localhost:8501 in your browser.

### Option 2: Python Script
```python
from src.causal_analysis import CausalImpactAnalysis
from src.data_pipeline import load_data

# Load data
df = load_data('data/raw_data.csv')

# Run analysis
analyzer = CausalImpactAnalysis()
results = analyzer.run(
    data=df,
    intervention_date='2024-03-01',
    pre_period=['2024-01-01', '2024-02-28'],
    post_period=['2024-03-01', '2024-03-31']
)

print(f"Campaign Impact: ${results.cumulative_effect:,.2f}")
print(f"ROI: {results.relative_effect:.1%}")
```

### Option 3: REST API
```bash
# Start server
uvicorn src.api:app --reload

# In another terminal
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"intervention_date": "2024-03-01"}'
```

### Option 4: Docker
```bash
docker-compose up -d
# Dashboard: http://localhost:8501
# API: http://localhost:8000
```

---

## 📊 Data Format

Your data should have these columns:

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `date` | datetime | ✓ | Observation date |
| `user_id` | string | ✓ | Unique user identifier |
| `treatment` | int | ✓ | 0=control, 1=treated |
| `revenue_usd` | float | ✓ | Outcome metric |
| `channel` | string | | Segment (optional) |
| `country` | string | | Segment (optional) |

**Example:**
```csv
date,user_id,treatment,revenue_usd,channel
2024-01-01,u001,1,125.50,email
2024-01-01,u002,0,0.00,social
```

---

## 📈 Common Workflows

### 1. Segment Analysis
```python
from src.dashboard import analyze_segments

# Compare across channels
results = analyze_segments(
    data=df,
    segment_column='channel',
    intervention_date='2024-03-01'
)
```

### 2. Heterogeneous Effects
```python
from src.causal_forest import CausalForest

# Find who responds best
cf = CausalForest()
cf.fit(features, treatment, outcome)
top_responders = cf.get_heterogeneity_groups(features)
```

### 3. Scheduled Reports
```python
from src.scheduler import AnalysisScheduler

scheduler = AnalysisScheduler()
scheduler.add_job(
    name='Daily Analysis',
    schedule_expr='daily at 9am',
    task_type='causal_impact'
)
scheduler.start()
```

---

## 📖 Next Steps

1. [Module Reference](index.md#-module-reference) - Detailed API docs
2. [API Reference](api-reference.md) - REST endpoints
3. [Deployment Guide](deployment.md) - Production setup

---

## ❓ Troubleshooting

### "No module named 'src'"
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Port already in use
```bash
uvicorn src.api:app --port 8001  # Use different port
```

### Docker issues
```bash
docker-compose down -v  # Clean restart
docker-compose up --build
```
