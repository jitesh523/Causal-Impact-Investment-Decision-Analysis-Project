# Causal Impact & Investment Decision Analysis

A complete implementation of causal impact analysis using Bayesian methods to measure marketing campaign effectiveness and translate findings into business value.

## 📊 Project Overview

This project implements the 8-step methodology for causal impact analysis to evaluate the effectiveness of marketing campaigns. Using Bayesian Structural Time Series (BSTS) modeling, we estimate the counterfactual (what would have happened without the intervention) and quantify the causal effect in both statistical and financial terms.

### Key Results
- **Revenue Impact**: $42,137.64 cumulative increase
- **ROI**: 7.43x return (742.75%)
- **Statistical Significance**: p < 0.0001
- **Campaign Cost**: $5,000

## 🎯 Methodology

Following the roadmap from "Building a Causal Impact & Investment Decision Analysis Project":

1. **Data Acquisition** - Marketing campaign dataset (5,000 records)
2. **Intervention Definition** - Treatment vs. control groups
3. **Data Pipeline** - Cleaning, aggregation, time series creation
4. **BSTS Modeling** - Bayesian regression with confidence intervals
5. **Results Interpretation** - Counterfactual predictions and impact quantification
6. **Financial Translation** - ROI, net profit, and business metrics
7. **Visualization** - Interactive dashboards and reports
8. **Best Practices** - Placebo tests, sensitivity analysis

## 📁 Project Structure

```
.
├── Dataset.xlsx                    # Raw marketing campaign data
├── requirements.txt                # Python dependencies
├── src/
│   ├── data_pipeline.py           # Data cleaning & time series creation
│   ├── causal_analysis.py         # Bayesian causal impact analysis
│   └── financial_analysis.py      # ROI & financial translations
├── data/
│   ├── processed/                 # Cleaned data outputs
│   └── raw/                       # Original datasets
├── reports/
│   ├── figures/                   # Visualizations
│   ├── financial_results.csv     # Financial metrics
│   └── business_narrative.txt    # Executive summary
└── notebooks/                     # Jupyter notebooks

```

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis

```bash
# Step 1: Data Pipeline
python src/data_pipeline.py

# Step 2: Causal Impact Analysis
python src/causal_analysis.py

# Step 3: Financial Analysis
python src/financial_analysis.py
```

## 📈 Dataset

**Source**: Marketing Campaign Performance Dataset  
**Size**: 5,000 customer records  
**Period**: Q1 2024 (Jan 1 - May 29)

### Key Fields
| Field | Description |
|-------|-------------|
| `treatment_exposed` | 1 = treated, 0 = control |
| `campaign_id` | Campaign identifier (CMP001-CMP010) |
| `channel` | Marketing channel (Search, Social, Display, Video, Email) |
| `revenue_usd` | Customer revenue |
| `conversion` | Binary conversion flag |
| `roi` | Return on investment |

### Treatment Split
- **Treated**: 2,316 customers (46.4%)
- **Control**: 2,672 customers (53.6%)

## 📊 Results Summary

### Causal Impact
- **Average Daily Impact**: $554.44
- **Cumulative Impact**: $42,137.64
- **Relative Effect**: Significant positive lift
- **Confidence**: 95% CI with p < 0.0001

### Financial Metrics
- **Campaign Cost**: $5,000.00
- **Gross Revenue**: $42,137.64
- **Net Profit**: $37,137.64
- **ROI**: 742.75% (7.43x)

### Interpretation
✅ The marketing campaign generated an additional **$42,137.64** in revenue over the post-intervention period. After accounting for the **$5,000** campaign cost, the net gain is approximately **$37,137.64**, translating to an **ROI of 742.8%**. These results are **statistically significant** (p < 0.0001).

## 📁 Outputs

### Generated Files
- `data/processed/cleaned_data.csv` - Cleaned dataset
- `data/processed/time_series_data.csv` - Time series format
- `data/processed/impact_metrics.csv` - Causal impact metrics
- `reports/figures/causal_impact_analysis.png` - Visualization
- `reports/financial_results.csv` - Financial metrics
- `reports/business_narrative.txt` - Executive summary

## 🔬 Technical Details

### Model Specifications
- **Method**: Bayesian Ridge Regression
- **Training Period**: 74 days (pre-intervention)
- **Evaluation Period**: 76 days (post-intervention)
- **Control Variables**: Control group revenue series
- **Confidence Level**: 95%

### Model Performance
- **R² Score** (pre-period): 1.0000
- **P-value**: < 0.0001
- **Statistical Power**: High

## 🎓 Assumptions & Limitations

1. **Parallel Trends**: Assumes control and treatment groups would have had similar trends without intervention
2. **No Spillover**: Treatment doesn't affect control group
3. **Stable Relationships**: Pre-period relationships hold in post-period
4. **No Confounding**: No major external shocks during analysis period

## 📚 References

- Brodersen et al. (2015) - "Inferring causal impact using Bayesian structural time-series models"
- Google CausalImpact Documentation
- Rob J. Hyndman - "Forecasting: Principles and Practice"

## 📝 License

This project is for educational and analytical purposes.

## 👥 Contact

For questions about this analysis, please open an issue or contact the project maintainer.

---

**Last Updated**: January 2026  
**Analysis Period**: Q1-Q2 2024  
**Model Version**: 1.0
