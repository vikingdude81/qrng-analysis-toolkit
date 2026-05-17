# Entropy Analysis Module

The `entropy_analysis` module provides comprehensive entropy computation and validation tools for analyzing the complexity of time series data in the context of consciousness research.

## Overview

This module implements multiple entropy measures that are essential for:
- Quantifying information flow in neural systems
- Detecting chaos and unpredictability in cognitive processes
- Analyzing multiscale complexity patterns
- Validating entropy computations against theoretical bounds

## Available Entropy Measures

### 1. Sample Entropy (SampEn)
Measures the likelihood that similar patterns of observations remain similar when extended by one more observation.

```python
from entropy_analysis import sample_entropy

entropy = sample_entropy(data, window_size=5, scaling_factor=1.5)
```

### 2. Approximate Entropy (ApEn)
Similar to SampEn but uses a different normalization approach and is computationally simpler.

```python
from entropy_analysis import approximate_entropy

entropy = approximate_entropy(data, window_size=5, scaling_factor=1.5)
```

### 3. Multiscale Entropy (MSE)
Computes entropy at multiple scales by rescaling the time series and analyzing complexity at each scale.

```python
from entropy_analysis import multiscale_entropy

entropies = multiscale_entropy(data, scale_factors=[1, 2, 3, 4, 5])
```

### 4. Permutation Entropy (PE)
Counts unique permutations of the time series to detect chaos and complexity.

```python
from entropy_analysis import permutation_entropy

entropy = permutation_entropy(data, n_perm=1000)
```

### 5. Fuzzy Entropy
Uses fuzzy membership functions for pattern matching, providing more robust estimates.

```python
from entropy_analysis import fuzzy_entropy

entropy = fuzzy_entropy(data, window_size=5, scaling_factor=1.5)
```

### 6. Renyi Entropy
Computes entropy of order alpha, useful for analyzing different aspects of complexity.

```python
from entropy_analysis import renyi_entropy

entropy = renyi_entropy(data, alpha=2.0)
```

### 7. Shannon Entropy
Basic information-theoretic measure of uncertainty in the discretized data.

```python
from entropy_analysis import shannon_entropy

entropy = shannon_entropy(data)
```

## Validation Functions

### Entropy Bounds Validation
Validates computed entropy values against theoretical bounds to ensure computational correctness.

```python
from entropy_analysis import validate_entropy_bounds

is_valid = validate_entropy_bounds(entropy_value, data_length, max_entropy_factor=1.0)
```

### Compute All Entropies
Computes all available entropy measures in a single call for comprehensive analysis.

```python
from entropy_analysis import compute_all_entropies

results = compute_all_entropies(data, window_size=5, scaling_factor=1.5)
print(results)
```

## Integration with Consciousness Metrics

The entropy measures can be integrated with consciousness metrics to provide a comprehensive view of system complexity:

```python
from consciousness_metrics import ConsciousnessMetrics
from entropy_analysis import compute_all_entropies

metrics = ConsciousnessMetrics()
entropies = compute_all_entropies(data)

# Combine for comprehensive analysis
complexity_score = metrics.integrated_information + sum(entropies.values())
```

## Theoretical Bounds

For a time series of length N, the maximum possible entropy is log₂(N). The `validate_entropy_bounds` function ensures that computed values are within theoretical limits.

## References

- Sample Entropy: Richman et al. (2005)
- Approximate Entropy: Pincus (1991)
- Multiscale Entropy: Costa et al. (2002)
- Permutation Entropy: Bandt and Pompe (2002)
- Fuzzy Entropy: Liang et al. (2005)
- Renyi Entropy: Renyi (1961)
