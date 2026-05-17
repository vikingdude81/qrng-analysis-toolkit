# Chaos Analysis Module

The `chaos_analysis` module provides comprehensive chaos analysis tools for analyzing the dynamics and complexity of time series data in the context of consciousness research.

## Overview

This module implements advanced chaos analysis techniques including:
- Lyapunov Exponent Estimation (Wolf Algorithm)
- Multiscale Entropy (MSE)
- Permutation Entropy (PE)
- Fuzzy Entropy
- Renyi Entropy
- Shannon Entropy

## Available Functions

### 1. Wolf Lyapunov Exponent
Computes the Lyapunov exponent using the Wolf algorithm, which is more accurate than Rosenstein's method for nonlinear systems.

```python
from chaos_analysis import wolf_lyapunov_exponent

le = wolf_lyapunov_exponent(data, max_lag=100, min_lag=5)
```

**Parameters:**
- `data`: Time series data (numpy array)
- `max_lag`: Maximum embedding lag to consider (default: 100)
- `min_lag`: Minimum embedding lag to consider (default: 5)

**Returns:** Lyapunov exponent or None if computation fails

### 2. Rosenstein Lyapunov Exponent
Computes the Lyapunov exponent using Rosenstein's algorithm as an alternative method.

```python
from chaos_analysis import rosenstein_lyapunov_exponent

le = rosenstein_lyapunov_exponent(data, max_lag=100, min_lag=5)
```

### 3. Multiscale Entropy (MSE)
Computes entropy at multiple scales by rescaling the time series and analyzing complexity at each scale.

```python
from chaos_analysis import multiscale_entropy

entropies = multiscale_entropy(data, scale_factors=[1, 2, 3, 4, 5])
```

**Parameters:**
- `data`: Time series data
- `scale_factors`: List of scale factors to compute entropy at (default: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
- `window_size`: Window size for sample entropy computation (default: 10)
- `scaling_factor`: Factor by which to rescale data at each level (default: 1.5)

**Returns:** List of entropy values at each scale

### 4. Permutation Entropy
Counts unique permutations of the time series to detect chaos and complexity.

```python
from chaos_analysis import permutation_entropy

entropy = permutation_entropy(data, n_perm=1000)
```

**Parameters:**
- `data`: Time series data
- `n_perm`: Number of permutations to consider (default: 1000)

**Returns:** Permutation entropy value

### 5. Fuzzy Entropy
Uses fuzzy membership functions for pattern matching, providing more robust estimates.

```python
from chaos_analysis import fuzzy_entropy

entropy = fuzzy_entropy(data, window_size=5, scaling_factor=1.5)
```

**Parameters:**
- `data`: Time series data
- `window_size`: Length of comparison patterns (default: 10)
- `scaling_factor`: Scaling factor for tolerance calculation (default: 1.5)

**Returns:** Fuzzy entropy value or nan if computation fails

### 6. Renyi Entropy
Computes entropy of order alpha, useful for analyzing different aspects of complexity.

```python
from chaos_analysis import renyi_entropy

entropy = renyi_entropy(data, alpha=2.0)
```

**Parameters:**
- `data`: Time series data (discretized)
- `alpha`: Order of Renyi entropy (default: 2.0)

**Returns:** Renyi entropy value or nan if computation fails

### 7. Shannon Entropy
Basic information-theoretic measure of uncertainty in the discretized data.

```python
from chaos_analysis import shannon_entropy

entropy = shannon_entropy(data)
```

**Parameters:**
- `data`: Time series data (discretized)

**Returns:** Shannon entropy value or nan if computation fails

### 8. Entropy Bounds Validation
Validates computed entropy values against theoretical bounds to ensure computational correctness.

```python
from chaos_analysis import validate_entropy_bounds

is_valid = validate_entropy_bounds(entropy_value, data_length, max_entropy_factor=1.0)
```

**Parameters:**
- `entropy_value`: Computed entropy value
- `data_length`: Length of the time series
- `max_entropy_factor`: Factor to adjust theoretical bound (default: 1.0)

**Returns:** True if entropy is within bounds, False otherwise

### 9. Compute All Entropies
Computes all available entropy measures in a single call for comprehensive analysis.

```python
from chaos_analysis import compute_all_entropies

results = compute_all_entropies(data, window_size=5, scaling_factor=1.5)
print(results)
```

**Parameters:**
- `data`: Time series data
- `window_size`: Window size for sample/approximate/fuzzy entropy (default: 10)
- `scaling_factor`: Scaling factor for tolerance calculation (default: 1.5)
- `scale_factors`: Scale factors for multiscale entropy (default: None, uses default scale factors)

**Returns:** Dictionary of all computed entropy values

## Integration with Consciousness Metrics

The chaos analysis functions can be integrated with consciousness metrics to provide a comprehensive view of system dynamics:

```python
from consciousness_metrics import ConsciousnessMetrics
from chaos_analysis import wolf_lyapunov_exponent, multiscale_entropy

metrics = ConsciousnessMetrics()
le = wolf_lyapunov_exponent(data)
mse = multiscale_entropy(data)

# Combine for comprehensive analysis
dynamics_score = metrics.integrated_information + abs(le) + sum(mse)
```

## Theoretical Bounds

For a time series of length N, the maximum possible entropy is log₂(N). The `validate_entropy_bounds` function ensures that computed values are within theoretical limits.

## References

- Wolf Algorithm: Wolf et al. (1985)
- Rosenstein's Algorithm: Rosenstein et al. (1993)
- Multiscale Entropy: Costa et al. (2002)
- Permutation Entropy: Bandt and Pompe (2002)
- Fuzzy Entropy: Liang et al. (2005)
- Renyi Entropy: Renyi (1961)
