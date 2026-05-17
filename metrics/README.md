# QRNG Analysis Toolkit - Metrics Module

## Overview

The metrics module in QRNG-Analysis-Toolkit provides a comprehensive suite of entropy, chaos, and consciousness metrics for analyzing quantum randomness and time series data. This module is designed with cross-project compatibility with Helios-Trajectory-Analysis.

## Features

- **Unified Interface**: Standardized API for computing multiple entropy metrics
- **Configurable Parameters**: Adjustable normalization, window size, embedding dimension, and time lag
- **Multiple Entropy Measures**: Shannon, Sample, Fuzzy, Permutation, Lyapunov Spectrum, and Epiplexity
- **Batch Processing**: Efficient computation across multiple time series
- **Cross-Project Compatibility**: Compatible with Helios-Trajectory-Analysis metrics module

## Installation

```bash
pip install scipy nolds numpy
```

## Usage

### Basic Usage

```python
from metrics.comprehensive_metrics import compute_all_metrics, MetricsConfig
import numpy as np

# Create a time series
time_series = np.random.randn(1000)

# Compute all metrics with default configuration
metrics = compute_all_metrics(time_series)
print(metrics)
```

### Custom Configuration

```python
from metrics.comprehensive_metrics import MetricsConfig, compute_all_metrics

# Create custom configuration
config = MetricsConfig(
    normalization='zscore',
    window_size=200,
    embedding_dim=7,
    time_lag=1
)

# Compute metrics with custom config
metrics = compute_all_metrics(time_series, config)
```

### Batch Processing

```python
from metrics.comprehensive_metrics import compute_metrics_batch
import numpy as np

# Create batch of time series
time_series_list = [
    np.random.randn(100),
    np.random.randn(200),
    np.random.randn(300)
]

# Compute metrics for all series
batch_results = compute_metrics_batch(time_series_list)
print(batch_results)
```

## Metrics Explained

### Shannon Entropy
Measures the uncertainty or randomness in a time series. Higher values indicate more randomness.

### Sample Entropy
Estimates the complexity of a time series based on the probability that similar patterns of observations remain similar when one additional observation is added.

### Fuzzy Entropy
A measure of signal regularity using fuzzy set theory, providing more robust estimates for short data sets.

### Permutation Entropy
Measures the complexity of a system by analyzing the permutation patterns in time series data.

### Lyapunov Spectrum
Characterizes the rate of divergence of nearby trajectories in phase space, indicating chaos.

### Epiplexity
Measures the degree of epistemic uncertainty or irreducible randomness in a system.

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| normalization | 'zscore' | Normalization method ('minmax' or 'zscore') |
| window_size | 100 | Fixed window size for analysis |
| embedding_dim | 5 | Embedding dimension m for permutation entropy |
| time_lag | 2 | Time lag τ for permutation entropy |
| sample_entropy_m | 3 | Embedding dimension for sample entropy |
| sample_entropy_r | 0.1 | Tolerance factor for sample entropy |

## Cross-Project Compatibility

This module is designed to be compatible with Helios-Trajectory-Analysis metrics module. Both use the same:
- Function signatures
- Configuration class structure
- Metric computation methods
- Return value formats

## See Also

- `helios-trajectory-analysis`: The primary QRNG analysis platform
- `consistency_analysis.md`: Analysis of metric inconsistencies across projects
