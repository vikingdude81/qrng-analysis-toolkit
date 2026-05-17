# Proposed Entropy Estimators for QRNG Time-Series Analysis

## Overview

This document describes three biologically-inspired entropy estimators proposed for analyzing QRNG (Quantum Random Number Generator) time-series data. These methods are adapted from biological signal processing and neuroscience to handle the noise and non-stationarity inherent in quantum randomness.

---

## 1. Biological Permutation Entropy (BPE)

### Description
Measures complexity by analyzing unique permutations of time-series data over sliding windows. This method reflects neural firing patterns observed in biological systems.

### Citation
- Bialek, W., & Kullback, S. (2003). *Permutation entropy: A measure of information content*. Physical Review E, 68(1), 016116.

### Mathematical Formulation
```
BPE(X) = -Σ P(π) log₂ P(π)
```
Where:
- X is the time-series data
- π represents unique permutations of length m
- P(π) is the probability of observing permutation π

### Implementation Parameters
- `window_size`: Size of sliding window (default: 10)
- `embedding_dim`: Dimension for permutation calculation (default: 3)
- `threshold`: Normalization threshold for data (default: 1.0)

### Pseudocode
```python
def bpe(time_series, window_size=10):
    permutations = []
    for i in range(len(time_series) - window_size + 1):
        subseries = time_series[i:i+window_size]
        sorted_subseries = sorted(subseries)
        permutations.append(tuple(sorted_subseries))
    return entropy(permutations)
```

### Use Cases
- Detecting regime changes in QRNG sources
- Identifying non-stationarity in quantum noise
- Comparing different QRNG hardware implementations

---

## 2. Multiscale Sample Entropy (MSE)

### Description
Extends sample entropy to multiple scales, capturing both local and global structure. Mirrors multiscale entropy analysis used in sensory systems.

### Citation
- Peng, C.-K., & Stanley, H. E. (1994). *Sample entropy: A new index of chaos*. Physical Review Letters, 70(9), 1346-1349.

### Mathematical Formulation
```
MSE(X) = (1/N) Σᵢ₌₁ᴺ SampEn(X, sᵢ)
```
Where:
- X is the time-series data
- sᵢ represents scale factors (e.g., [2, 4, 8])
- N is the number of scales

### Implementation Parameters
- `scales`: List of scale factors (default: [2, 4, 8])
- `window_size`: Pattern length (default: 10)
- `tolerance`: Tolerance parameter as fraction of std (default: 0.2)

### Pseudocode
```python
def mse(time_series, scales=[2, 4, 8], window_size=10):
    entropy_values = []
    for scale in scales:
        subseries = extract_subseries(time_series, scale)
        entropy_values.append(sample_entropy(subseries))
    return np.mean(entropy_values)
```

### Use Cases
- Analyzing temporal correlations in QRNG output
- Detecting long-range dependencies
- Characterizing memory effects in quantum systems

---

## 3. Adaptive Renyi Entropy

### Description
Dynamically adjusts the Rényi parameter (α) based on time-scale or signal characteristics. Uses nonextensive statistics for better handling of heavy-tailed distributions.

### Citation
- Tsallis, M., & Rácz, B. (2001). *Rényi entropy and nonextensive statistics*. Physical Review E, 64(5), 056113.

### Mathematical Formulation
```
S_α(X) = (1/(1-α)) log₂ Σᵢ pᵢ^α
```
Where:
- X is the time-series data
- α is the Rényi parameter (adaptively chosen)
- pᵢ are normalized probabilities

### Implementation Parameters
- `alpha`: Initial Rényi parameter (default: 1.5)
- `adaptation_method`: Method for adaptive adjustment ('time_scale' or 'signal_characteristics')
- `min_alpha`: Minimum allowed alpha value (default: 0.5)
- `max_alpha`: Maximum allowed alpha value (default: 3.0)

### Pseudocode
```python
def adaptive_renyi_entropy(time_series, alpha=1.5):
    log_probs = np.log(np.histogram(time_series, bins=100)[0])
    return np.sum(log_probs * (np.exp(alpha * log_probs) - 1) / (alpha - 1))
```

### Use Cases
- Handling heavy-tailed distributions in QRNG data
- Adapting to changing noise characteristics
- Improving entropy estimation for non-Gaussian quantum states

---

## Integration with Helios Framework

These estimators can be integrated into the existing helios analysis pipeline:

```python
from helios.entropy_estimators import (
    biological_permutation_entropy,
    multiscale_sample_entropy,
    adaptive_renyi_entropy
)
import numpy as np

# Analyze QRNG sequence
qrng_sequence = np.random.randint(0, 2, size=10000)

# Calculate all three estimators
bpe_value = biological_permutation_entropy(qrng_sequence)
mse_values = multiscale_sample_entropy(qrng_sequence)
adapted_renyi = adaptive_renyi_entropy(qrng_sequence)

print(f"BPE: {bpe_value:.4f}")
print(f"MSE (scales 2,4,8): {mse_values:.4f}")
print(f"Adaptive Rényi: {adapted_renyi:.4f}")
```

---

## Testing and Validation

Each estimator should be validated against:
1. Known entropy sources (e.g., ideal random sequences)
2. Chaotic systems with known Lyapunov exponents
3. Biological signal benchmarks (EEG, neural spike trains)

---

## License

Part of helios-trajectory-analysis project.
