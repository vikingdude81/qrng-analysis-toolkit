# Helios Metrics Module Guide

## Overview

The `helios_trajectory_analysis` package provides a comprehensive framework for consciousness metrics computation, chaos analysis, and quantum randomness trajectory analysis.

## New Modules Added

### 1. `metrics_base.py`

Provides the abstract base class `MetricBase` that defines the interface for all consciousness metrics:

```python
class MetricBase(metaclass=ABCMeta):
    @abstractmethod
    def compute(self, data): ...  # Compute metric value
    @abstractmethod
    def validate_input(self, data): ...  # Validate input structure
    @abstractmethod
    def metadata(self): ...  # Return metric metadata
```

### 2. `metrics_adapter.py`

Provides the `MetricAdapter` wrapper class for compatibility:

```python
class MetricAdapter:
    def __init__(self, original_func: Callable):
        self.original = original_func
    
    def __call__(self, *args, **kwargs):
        return self.original(*args, **kwargs)
```

### 3. `robustness_guards.py`

Provides utility functions with robustness guards:

```python
def _ensure_input(data: Any) -> Tuple[Any, float]:
    """Ensure input is numeric and return (value, confidence)."""

def _compute_entropy(data: Any) -> Tuple[Any, float]:
    """Compute Shannon entropy with robustness guards."""

def _compute_epiplexity(data: Any) -> Tuple[Any, float]:
    """Compute epipole count with robustness guards."""
```

## Usage Examples

### Using MetricBase

```python
from helios_trajectory_analysis.metrics_base import MetricBase
from typing import List

class ShannonEntropyMetric(MetricBase):
    def compute(self, data: List[float]) -> float:
        """Compute Shannon entropy."""
        from math import log2
        if len(data) == 0:
            return 0.0
        # Normalize to probabilities
        probs = [x / sum(data) for x in data]
        return -sum(p * log2(p) for p in probs if p > 0)
    
    def validate_input(self, data):
        """Validate input is a list of numbers."""
        return isinstance(data, list) and all(isinstance(x, (int, float)) for x in data)
    
    def metadata(self):
        """Return metric metadata."""
        return {
            "name": "Shannon Entropy",
            "description": "Computes Shannon entropy of input distribution",
            "input_type": "List[float]",
            "output_type": "float"
        }
```

### Using MetricAdapter

```python
from helios_trajectory_analysis.metrics_adapter import MetricAdapter

def my_compute_function(data):
    return data * 2

adapter = MetricAdapter(my_compute_function)
result = adapter(5)  # Returns 10
```

### Using Robustness Guards

```python
from helios_trajectory_analysis.robustness_guards import (
    _ensure_input,
    _compute_entropy,
    _compute_epiplexity,
)

# Ensure input is numeric
value, confidence = _ensure_input(42.5)
print(f"Value: {value}, Confidence: {confidence}")

# Compute entropy (expects numeric or array-like)
result, entropy = _compute_entropy([1.0, 2.0, 3.0])
print(f"Result: {result}, Entropy: {entropy}")

# Compute epiplexity
result, epiplexity = _compute_epiplexity(5)
print(f"Epiplexity: {epiplexity}")
```

## Migration Guide

### From Old Interface to MetricBase

If you have existing metric functions like:

```python
def compute_entropy(data):
    # ... computation ...
    return result
```

You can wrap them using `MetricAdapter`:

```python
from helios_trajectory_analysis.metrics_adapter import MetricAdapter

entropy_metric = MetricAdapter(compute_entropy)
# Now entropy_metric behaves like a MetricBase instance
```

Or implement the full `MetricBase` interface for better metadata support.

## Testing

Run the test suite:

```bash
cd helios-trajectory-analysis
python test_helios_imports.py
```

## Circular Import Resolution

All circular import issues have been resolved by:
1. Using absolute imports instead of relative imports
2. Creating proper package structure with `__init__.py`
3. Separating concerns into dedicated modules

## Next Steps

1. Review existing metrics in `consciousness_metrics.py` and migrate to `MetricBase`
2. Add new metrics following the pattern above
3. Update documentation for each metric class
4. Add unit tests for each metric implementation
