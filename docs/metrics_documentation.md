# Helios Metrics Documentation

## Overview
This document defines the statistical and entropy measures used in Helios for QRNG analysis and consciousness research.

## Terminology Standards

### Sample vs Samples
- Use **`samples`** (plural) when referring to the number of data points
- Example: `samples = 1000`

### Approximate vs Exact
- **`exact`**: Full entropy calculation using all available data
- **`estimated`/`approximate`**: Methods that use sampling or binning for efficiency

### Permutation vs Shuffling
- **`permutation`**: Explicitly refers to the permutation-based entropy calculation method (e.g., permutation entropy)
- **`shuffling`**: Refers to random reordering of data for baseline comparison

## Implemented Metrics

### 1. Shannon Entropy
```python
def calculate_shannon_entropy(data):
    """Calculate exact Shannon entropy from data samples."""
    # Exact calculation using all samples
```

### 2. Permutation Entropy
```python
def calculate_permutation_entropy(data, dim=3, eps=None):
    """Calculate permutation entropy with configurable embedding dimension."""
    # Uses permutation-based approach for complexity measurement
```

### 3. Multiscale Entropy (MSE)
```python
def calculate_mse(data, scales=[2, 4, 8]):
    """Calculate multiscale entropy across multiple time scales.
    
    Args:
        data: Input time series
        scales: List of scale factors to analyze
    
    Returns:
        Dictionary mapping each scale to its entropy value
    """
```

### 4. Conditional Entropy
```python
def calculate_conditional_entropy(X, Y):
    """Calculate conditional entropy H(Y|X) - information gain between variables.
    
    Args:
        X: Conditioning variable
        Y: Target variable
    
    Returns:
        Conditional entropy value
    """
```

### 5. Approximate Entropy (ApEn)
```python
def calculate_approximate_entropy(data, m=2, r=0.2):
    """Calculate approximate entropy for regularity measurement.
    
    Args:
        data: Input time series
        m: Pattern length
        r: Tolerance factor (as fraction of SD)
    
    Returns:
        Approximate entropy value
    """
```

### 6. Sample Entropy (SampEn)
```python
def calculate_sample_entropy(data, m=2, r=0.2):
    """Calculate sample entropy for improved regularity measurement.
    
    Args:
        data: Input time series
        m: Pattern length
        r: Tolerance factor (as fraction of SD)
    
    Returns:
        Sample entropy value
    """
```

## Consciousness Metrics

### Epiplexity Estimator
```python
def estimate_epiplexity(data, threshold=0.1):
    """Estimate epiplexity - measure of information coupling.
    
    Args:
        data: Input time series
        threshold: Coupling strength threshold
    
    Returns:
        Epiplexity score (0-1 scale)
    """
```

### Influence Detection
```python
def detect_influence_patterns(data, window_size=50):
    """Detect influence patterns in time series data.
    
    Args:
        data: Input time series
        window_size: Analysis window size
    
    Returns:
        List of detected influence events with timestamps
    """
```

### Chaos Analysis
```python
def calculate_lyapunov_exponents(data, max_lag=100):
    """Calculate Lyapunov exponents for chaos detection.
    
    Args:
        data: Input time series
        max_lag: Maximum lag for embedding
    
    Returns:
        List of Lyapunov exponents
    """
```

## Alignment with nolds/entropy-py

All metrics should be cross-checked against:
- `nolds` library for entropy calculations
- `entropy-py` for additional measures

See `docs/metrics_reference.md` for detailed API documentation.
