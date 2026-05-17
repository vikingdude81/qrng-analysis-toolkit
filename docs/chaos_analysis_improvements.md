# Chaos Analysis Improvements

## Review Summary for `chaos_analysis.py`:

### 1. Lyapunov Exponent Estimation:
- **Issue:** If using Rosenstein's algorithm (e.g., `rSSE`), it may underestimate Lyapunov exponents due to sensitivity to noise and small sample sizes.
- **Recommendation:** Replace with **Wolf algorithm** (`wolf`) for higher accuracy, especially for nonlinear systems.

### 2. Entropy Bounds:
- **Issue:** If using `sample_entropy` or `approximate_entropy`, ensure the window size and scaling factor are appropriate (e.g., `window_size=10`, `scaling_factor=1.5`).
- **Recommendation:** Validate entropy values against theoretical bounds (e.g., for a 1D system, entropy should be < log₂(N) where N is data length).

### 3. Proposed New Analyses:

#### Multiscale Entropy (MSE):
Compute entropy at multiple scales by rescaling the time series and analyzing complexity.
```python
from scipy.signal import resample
def multiscale_entropy(data, scale_factors):
    entropies = []
    for sf in scale_factors:
        scaled_data = resample(data, len(data) * sf)
        entropy = sample_entropy(scaled_data)
        entropies.append(entropy)
    return entropies
```

#### Permutation Entropy (PE):
Count unique permutations of the time series to detect chaos.
```python
from scipy.signal import permutation_entropy
def permutation_entropy(data):
    return permutation_entropy(data, n_perm=1000)
```

## Key Improvements:
- Replace Rosenstein with Wolf for Lyapunov exponents.
- Validate entropy measures with proper scaling and window sizes.
- Add multiscale and permutation entropy for deeper complexity analysis.
