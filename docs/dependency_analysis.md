# Dependency Analysis Report

## Module Status Overview

| module | missing dependency | proposed fix |
|--------|---------------------|---------------|
| entropy_estimators.py | None | Complete with Shannon, Rényi-α=2, Tsallis-q=2, kNN-based entropy |
| statistical_estimators.py | None | Abstract interfaces for chaos analysis estimators |
| metrics/compat.py | None | Compatibility layer for estimator fallbacks |

## Missing Functions/Classes Identified

### Entropy Estimators
- Shannon entropy ✓
- Rényi-α=2 entropy ✓
- Tsallis-q=2 entropy ✓
- kNN-based entropy ✓

### Statistical Estimators
- LyapunovExponentEstimator ✓
- CorrelationDimensionEstimator ✓
- ConfidenceEstimation ✓
- StatisticalEstimatorConfig ✓

## Compatibility Layer

The `metrics/compat.py` module provides:
- `wrap_estimator()`: Wraps estimator with fallback if unavailable
- `compat_metric()`: Decorator for metrics that use compatibility layer

### Usage Example
```python
from metrics.compat import compat_metric

@compat_metric
def my_metric(x):
    return entropy(x)  # Uses Helios if available, else default
```

## Next Steps

1. Integrate `entropy_estimators.py` into existing analysis pipelines
2. Add statistical estimators to chaos_analysis.py module
3. Test compatibility layer with existing metrics
4. Document all new estimator interfaces