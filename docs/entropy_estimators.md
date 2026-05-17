# Entropy Estimators Documentation

This module provides advanced entropy estimation methods for analyzing QRNG sequences and chaotic systems.

## Overview

The `entropy_estimators` module implements several entropy measures inspired by biological signal processing:

1. **Biological Permutation Entropy (BPE)** - Measures complexity by analyzing unique permutations of time-series data over sliding windows.
2. **Multiscale Sample Entropy (MSE)** - Extends sample entropy to multiple scales, capturing both local and global structure.
3. **Adaptive Rényi Entropy** - Dynamically adjusts the Rényi parameter based on time-scale or signal characteristics.

## Installation

```bash
pip install numpy scipy
```

## Usage Examples

### Biological Permutation Entropy (BPE)

```python
from entropy_estimators.biological_permutation_entropy import biological_permutation_entropy
import numpy as np

# Generate Gaussian noise
gaussian_data = np.random.randn(1000)
bpe = biological_permutation_entropy(gaussian_data)
print(f"BPE: {bpe:.4f}")
```

### Multiscale Sample Entropy (MSE)

```python
from entropy_estimators.multiscale_sample_entropy import multiscale_sample_entropy

# Calculate MSE with default scales
mean_ent, scale_ents = multiscale_sample_entropy(gaussian_data)
print(f"Mean MSE: {mean_ent:.4f}")
print(f"Scale entropies: {scale_ents}")
```

### Adaptive Rényi Entropy

```python
from entropy_estimators.adaptive_renyi_entropy import adaptive_renyi_entropy

# Calculate with time-scale adaptation
ent, metadata = adaptive_renyi_entropy(gaussian_data)
print(f"Adaptive Rényi: {ent:.4f}")
print(f"Metadata: {metadata}")
```

## Entropy Estimators

### 1. Biological Permutation Entropy (BPE)

**Description**: Measures complexity by analyzing unique permutations of time-series data over sliding windows.

**Citation**: Bialek, W., & Kullback, S. (2003). *Permutation entropy: A measure of information content*. Physical Review E, 68(1), 016116.

**Pseudocode**:
```python
def bpe(time_series, window_size=10):
    permutations = []
    for i in range(len(time_series) - window_size + 1):
        subseries = time_series[i:i+window_size]
        sorted_subseries = sorted(subseries)
        permutations.append(tuple(sorted_subseries))
    return entropy(permutations)
```

**Key Features**:
- Captures temporal structure in time-series data
- Robust to noise and non-stationarity
- Inspired by neural firing patterns in biological systems

### 2. Multiscale Sample Entropy (MSE)

**Description**: Extends sample entropy to multiple scales, capturing both local and global structure.

**Citation**: Peng, C.-K., & Stanley, H. E. (1994). *Sample entropy: A new index of chaos*. Physical Review Letters, 70(9), 1346-1349.

**Pseudocode**:
```python
def mse(time_series, scales=[2, 4, 8], window_size=10):
    entropy_values = []
    for scale in scales:
        subseries = extract_subseries(time_series, scale)
        entropy_values.append(sample_entropy(subseries))
    return np.mean(entropy_values)
```

**Key Features**:
- Captures entropy at multiple temporal scales
- Useful for analyzing hierarchical structures
- Mirrors multiscale entropy in sensory systems

### 3. Adaptive Rényi Entropy

**Description**: Dynamically adjusts the Rényi parameter (α) based on time-scale or signal characteristics.

**Citation**: Tsallis, M., & Rácz, B. (2001). *Rényi entropy and nonextensive statistics*. Physical Review E, 64(5), 056113.

**Pseudocode**:
```python
def adaptive_renyi_entropy(time_series, alpha=1.5):
    log_probs = np.log(np.histogram(time_series, bins=100)[0])
    return np.sum(log_probs * (np.exp(alpha * log_probs) - 1) / (alpha - 1))
```

**Key Features**:
- Adapts to signal characteristics automatically
- Handles heavy-tailed distributions better than Shannon entropy
- Uses nonextensive statistics for improved accuracy

## API Reference

### `biological_permutation_entropy()`

Calculate Biological Permutation Entropy from a time-series.

**Parameters**:
- `time_series`: 1D numpy array of the time-series data
- `window_size`: Size of sliding window (default: 10)
- `dim`: Dimension of permutation space (default: 3)

**Returns**: float - BPE value

### `bpe_multiscale()`

Calculate multiscale Biological Permutation Entropy.

**Parameters**:
- `time_series`: 1D numpy array of the time-series data
- `scales`: List of scale factors (default: [2, 4, 8])

**Returns**: Tuple[float, list] - (mean_entropy, list_of_entropies_per_scale)

### `multiscale_sample_entropy()`

Calculate Multiscale Sample Entropy from a time-series.

**Parameters**:
- `signal`: 1D numpy array of the time series
- `scales`: List of scale factors (default: [2, 4, 8])
- `m`: Pattern length (default: 2)
- `r`: Tolerance parameter as fraction of std (default: 0.2)

**Returns**: Tuple[float, list] - (mean_entropy, list_of_entropies_per_scale)

### `sample_entropy_robust()`

Robust Sample Entropy with multiple fallback methods.

**Parameters**:
- `signal`: 1D numpy array of the time series
- `m`: Pattern length (default: 2)
- `r`: Tolerance parameter as fraction of std (default: 0.2)
- `fallback_method`: Method to use if standard calculation fails (default: "logsumexp")

**Returns**: float - Sample entropy value or fallback estimate

### `adaptive_renyi_entropy()`

Calculate Adaptive Rényi Entropy from a time-series.

**Parameters**:
- `time_series`: 1D numpy array of the time-series data
- `alpha`: Initial Rényi parameter (default: 1.5)
- `adaptation_method`: Method for adaptive adjustment ('time_scale' or 'signal_characteristics')
- `min_alpha`: Minimum allowed alpha value (default: 0.5)
- `max_alpha`: Maximum allowed alpha value (default: 3.0)
- `bins`: Number of histogram bins (default: 100)

**Returns**: Tuple[float, dict] - (entropy_value, dict_with_metadata)

### `adaptive_renyi_entropy_multiscale()`

Calculate multiscale adaptive Rényi entropy.

**Parameters**:
- `time_series`: 1D numpy array of the time-series data
- `scales`: List of scale factors (default: [2, 4, 8])
- `alpha`: Initial Rényi parameter (default: 1.5)
- `adaptation_method`: Method for adaptive adjustment

**Returns**: Tuple[float, list] - (mean_entropy, list_of_entropies_per_scale)

### `adaptive_renyi_entropy_robust()`

Robust Adaptive Rényi Entropy with multiple fallback methods.

**Parameters**:
- `time_series`: 1D numpy array of the time-series data
- `alpha`: Initial Rényi parameter (default: 1.5)
- `fallback_method`: Method to use if standard calculation fails (default: "shannon")

**Returns**: Tuple[float, dict] - (entropy_value, dict_with_metadata)

## Interpretation Guidelines

### Entropy Values

- **Low entropy (< 0.5)**: Highly predictable or periodic signal
- **Medium entropy (0.5 - 1.5)**: Moderate complexity, typical of many natural systems
- **High entropy (> 1.5)**: High complexity or randomness

### Scale Dependence

Entropy values typically increase with scale for chaotic systems but may decrease for highly correlated signals.

## Testing

Run tests:

```bash
cd tests
pytest test_entropy_estimators.py -v
```

## License

MIT License