# TensorAnalyzer Usage Guide

## Overview

The `TensorAnalyzer` class provides quantum-inspired tensor analysis capabilities for QRNG data streams. It follows a simple fit-then-analyze pattern.

## Basic Usage

```python
from cuquantum_accelerator.tensor_analysis import TensorAnalyzer

# Initialize analyzer
analyzer = TensorAnalyzer()

# Fit to data stream
analyzer.fit(data)

# Get analysis results
result = analyzer.analyze()
```

## Flow Diagram

```
Input --> QRNG --> Preprocessing --> Analysis --> Output
```

## Installation Requirements

```bash
pip install pytest hypothesis
```

## Testing

Run the test suite:

```bash
pytest tests/test_tensor_analysis.py --cov=cuquantum_accelerator.tensor_analysis
```

## Example: Complete Analysis Pipeline

```python
import numpy as np
from cuquantum_accelerator.tensor_analysis import TensorAnalyzer

# Generate sample QRNG data
np.random.seed(42)
data = [np.random.randn() for _ in range(1000)]

# Initialize and fit analyzer
analyzer = TensorAnalyzer()
analyzer.fit(data)

# Get analysis results
result = analyzer.analyze()
print(f"Entropy: {result.get('entropy', 'N/A')}")
print(f"Stationarity score: {result.get('stationarity_score', 'N/A')}")
```

## Integration with Robustness Analysis

The TensorAnalyzer can be integrated with the robustness checks defined in `docs/robustness_analysis.md`:

```python
from cuquantum_accelerator.tensor_analysis import TensorAnalyzer
from scipy.stats import entropy as scipy_entropy

analyzer = TensorAnalyzer()
analyzer.fit(data)
result = analyzer.analyze()

# Additional time-series entropy check
time_series_entropies = compute_time_series_entropy(data, window_size=10)
```

## See Also

- [`docs/robustness_analysis.md`](../docs/robustness_analysis.md) - Robustness analysis for QRNG metrics
- [`tests/test_helios_anomaly_scope.py`](../../tests/test_helios_anomaly_scope.py) - Edge case tests
- [`inference_framework/tests/test_inference_framework.py`](../../inference_framework/tests/test_inference_framework.py) - Inference framework tests

---

*Last updated: 2024*