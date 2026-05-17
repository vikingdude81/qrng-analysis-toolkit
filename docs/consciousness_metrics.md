# Consciousness Metrics Module

The `consciousness_metrics` module provides standardized interfaces for accessing key performance indicators derived from consciousness research.

## Overview

This module provides a class-based interface for accessing top-level metrics extracted from consciousness analysis source code, including:
- Integrated Information (Φ)
- Epiplexity Score
- Other potentially relevant metrics if available in the module

## ConsciousnessMetrics Class

```python
from consciousness_metrics import ConsciousnessMetrics

m = ConsciousnessMetrics()
print(m.get_integrated_information())  # Returns the value of integrated_information
print(m.get_epiplexity_score())         # Returns the value of epiplexity_score
```

### Properties

#### `integrated_information`
Returns the value of `integrated_information` from the source code.

**Raises:**
- `ValueError`: If no value is available for this metric in the source file.

#### `epiplexity_score`
Returns the value of `epiplexity_score` from the source code.

**Raises:**
- `ValueError`: If no value is available for this metric in the source file.

#### `other_metrics`
Returns all available metrics from the source code.

**Raises:**
- `ValueError`: If no value is available for any metric in the source file.

#### `get_all_metrics()`
Returns all available metrics from the source code as a dictionary.

**Raises:**
- `ValueError`: If no value is available for any metric in the source file.

### Methods

#### `set_source_file(file_path: str)`
Sets the file path where these metrics were extracted from.

**Parameters:**
- `file_path (str)`: The path to the source file containing the metrics.

## Usage Examples

```python
from consciousness_metrics import ConsciousnessMetrics

# Create instance
m = ConsciousnessMetrics()

# Access individual metrics
try:
    ii = m.integrated_information
    print(f"Integrated Information: {ii}")
except ValueError as e:
    print(f"Error: {e}")

# Access epiplexity score
try:
    eps = m.epiplexity_score
    print(f"Epiplexity Score: {eps}")
except ValueError as e:
    print(f"Error: {e}")

# Get all metrics
try:
    all_metrics = m.get_all_metrics()
    print(f"All Metrics: {all_metrics}")
except ValueError as e:
    print(f"Error: {e}")
```

## Integration with Chaos Analysis

The consciousness metrics can be combined with chaos analysis for comprehensive system characterization:

```python
from consciousness_metrics import ConsciousnessMetrics
from chaos_analysis import wolf_lyapunov_exponent, multiscale_entropy

metrics = ConsciousnessMetrics()
le = wolf_lyapunov_exponent(data)
mse = multiscale_entropy(data)

# Comprehensive complexity score
complexity_score = (
    metrics.integrated_information + 
    abs(le) + 
    sum(mse)
)
```

## References

- Integrated Information Theory: Tononi et al. (2016)
- Epiplexity: Various consciousness research frameworks