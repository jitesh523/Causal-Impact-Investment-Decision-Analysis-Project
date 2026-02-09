# Causal Forest Module

> Tree-based estimation of Heterogeneous Treatment Effects (HTE).

## Overview

The `causal_forest.py` module implements Generalized Random Forests methodology for discovering how treatment effects vary across different subgroups in your data.

## Class: `CausalForest`

### Constructor

```python
from src.causal_forest import CausalForest

cf = CausalForest(
    n_estimators=100,       # Number of trees
    max_depth=None,         # Max tree depth (None = unlimited)
    min_samples_leaf=5,     # Min samples per leaf
    honesty=True,           # Use honest estimation
    random_state=42
)
```

### Methods

#### `fit(X, treatment, outcome) -> self`

Fit the causal forest.

```python
cf.fit(X, treatment, outcome)
```

#### `predict(X, return_se=False) -> np.ndarray`

Predict Conditional Average Treatment Effects (CATEs).

```python
cates = cf.predict(X_new)               # Just effects
cates, se = cf.predict(X_new, return_se=True)  # With SEs
```

#### `get_heterogeneity_groups(X, n_groups=4) -> pd.DataFrame`

Identify groups with different treatment effects.

```python
groups = cf.get_heterogeneity_groups(X, n_groups=4)
print(groups)
#   group  n_samples  mean_cate   std_cate
# 0     1        250      2.34       1.12
# 1     2        250      4.56       1.45
# ...
```

#### `get_feature_importance() -> pd.DataFrame`

Get feature importance for effect heterogeneity.

```python
importance = cf.get_feature_importance()
# Shows which features drive variation in treatment effects
```

---

## Results Dataclass

```python
@dataclass
class CausalForestResults:
    ate: float                             # Average Treatment Effect
    ate_se: float                          # Standard error of ATE
    cate: np.ndarray                       # Individual CATEs
    cate_se: np.ndarray                    # Standard errors
    feature_importance: Dict[str, float]   # Importance scores
    variable_importance: pd.DataFrame      # Full importance table
```

---

## Example: Finding Best Responders

```python
from src.causal_forest import CausalForest

# Fit model
cf = CausalForest(n_estimators=100)
cf.fit(X, treatment=campaign_exposed, outcome=revenue)

# Get individual effects
cates = cf.predict(X)

# Find who benefits most
top_responders = X[cates > np.percentile(cates, 90)]

# What drives heterogeneity?
importance = cf.get_feature_importance()
print("Top factors affecting treatment response:")
print(importance.head(5))
```

---

## When to Use

| Use Case | Recommended Method |
|----------|-------------------|
| Overall campaign impact | `CausalImpactAnalysis` |
| Who responds best? | **`CausalForest`** ✓ |
| Debiased estimates with ML | `DoubleML` |

---

## See Also

- [Double ML](double_ml.md) - Alternative ML-based approach
- [Propensity Matching](propensity_matching.md) - Classic matching
