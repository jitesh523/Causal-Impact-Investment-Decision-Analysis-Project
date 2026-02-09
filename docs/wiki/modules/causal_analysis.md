# Causal Analysis Module

> Core Bayesian Structural Time Series (BSTS) engine for causal impact estimation.

## Overview

The `causal_analysis.py` module implements the primary causal inference algorithm using BSTS modeling. It compares observed outcomes against a counterfactual "what would have happened without the intervention" baseline.

## Class: `CausalImpactAnalysis`

### Constructor

```python
from src.causal_analysis import CausalImpactAnalysis

analyzer = CausalImpactAnalysis(
    alpha=0.05,           # Significance level for confidence intervals
    n_seasons=7,          # Seasonality period (7 for weekly)
    standardize=True      # Whether to standardize data
)
```

### Methods

#### `run(data, intervention_date, pre_period, post_period) -> CausalImpactResults`

Run causal impact analysis.

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `data` | `pd.DataFrame` | Time series data with date index |
| `intervention_date` | `str` | Date when intervention started |
| `pre_period` | `list[str]` | [start, end] of pre-intervention period |
| `post_period` | `list[str]` | [start, end] of post-intervention period |

**Returns:** `CausalImpactResults` dataclass

**Example:**
```python
results = analyzer.run(
    data=df,
    intervention_date='2024-03-01',
    pre_period=['2024-01-01', '2024-02-28'],
    post_period=['2024-03-01', '2024-03-31']
)

print(f"Cumulative Effect: ${results.cumulative_effect:,.2f}")
print(f"Average Effect: ${results.average_effect:,.2f}")
print(f"P-value: {results.p_value:.4f}")
```

---

## Results Dataclass

```python
@dataclass
class CausalImpactResults:
    cumulative_effect: float       # Total impact over post-period
    cumulative_effect_lower: float # Lower CI bound
    cumulative_effect_upper: float # Upper CI bound
    average_effect: float          # Average daily impact
    relative_effect: float         # Percentage change
    p_value: float                 # Statistical significance
    posterior_prob: float          # Probability effect is positive
```

---

## Methodology

1. **Model Training**: Fit BSTS model on pre-intervention data
2. **Counterfactual**: Predict what would have happened post-intervention
3. **Impact**: Compare actual vs predicted to estimate causal effect
4. **Inference**: Compute confidence intervals and p-values

---

## See Also

- [Causal Forest](causal_forest.md) - For heterogeneous effects
- [Double ML](double_ml.md) - Machine learning approach
- [API Reference](../api-reference.md) - REST API endpoints
