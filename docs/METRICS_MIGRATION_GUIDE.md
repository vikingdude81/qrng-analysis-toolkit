# Metrics Module Migration Guide

## Overview

This guide explains how to migrate from the legacy metrics implementations to the new unified metrics interface in Helios-Trajectory-Analysis. The new interface provides cross-project compatibility with QRNG-Analysis-Toolkit.

## What Changed

### Before (Legacy)
```python
# Old approach - inconsistent across modules
from consciousness_metrics import compute_shannon_entropy
from chaos_analysis import compute_sample_entropy
from epiplexity_estimator import compute_epiplexity

shannon = compute_shannon_entropy(data)
sample = compute_sample_entropy(data, m=3, r=0.1)
epi = compute_epiplexity(data)
```

### After (Unified)
```python
# New unified approach
from src.metrics.comprehensive_metrics import compute_all_metrics, MetricsConfig

config = MetricsConfig()
metrics = compute_all_metrics(data, config)
print(metrics['shannon_entropy'])
print(metrics['sample_entropy'])
print(metrics['epiplexity'])
```

## Migration Steps

### Step 1: Import the New Module

```python
# Add to your imports
from src.metrics.comprehensive_metrics import (
    compute_all_metrics,
    MetricsConfig,
    compute_metrics_batch
)
```

### Step 2: Replace Individual Metric Calls

```python
# Old code
shannon = compute_shannon_entropy(data, base=2)
sample = sample_entropy(data, m=3, r=0.1)
fuzzy = fuzzy_entropy(data)
perm = permutation_entropy(data, embedding_dim=5, tau=2)
lyap = lyapunov_spectrum(data, method='approx')
epi = epiplexity(data)

# New code
config = MetricsConfig()
metrics = compute_all_metrics(data, config)
```

### Step 3: Update Configuration

```python
# Old approach - hardcoded parameters
m = 5
tau = 2

# New approach - use configuration object
config = MetricsConfig(
    normalization='zscore',
    embedding_dim=5,
    time_lag=2
)
```

### Step 4: Handle Batch Processing

```python
# Old approach - loop through data
for series in time_series_list:
    metrics = compute_all_metrics(series, config)

# New approach - use batch function
batch_results = compute_metrics_batch(time_series_list, config)
```

## Configuration Options

### Normalization Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| 'zscore' | Standardize to mean=0, std=1 | Default for statistical analysis |
| 'minmax' | Scale to [0, 1] range | When you need bounded values |

### Parameter Defaults

| Parameter | Default | Range |
|-----------|---------|-------|
| embedding_dim | 5 | 2-10 |
| time_lag | 2 | 1-5 |
| window_size | 100 | 50-500 |

## Quick Start

```python
from src.metrics.comprehensive_metrics import compute_all_metrics, MetricsConfig
import numpy as np

# Create sample data
data = np.random.randn(1000)

# Compute all metrics with defaults
metrics = compute_all_metrics(data)
print(metrics)
```

## See Also

- `METRICS_README.md`: Full documentation
- `consistency_analysis.md`: Cross-project compatibility analysis
