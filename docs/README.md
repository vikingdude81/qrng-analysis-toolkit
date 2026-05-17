# Helios Trajectory Analysis

A comprehensive platform for QRNG (Quantum Random Number Generator) analysis, consciousness metrics computation, and chaos analysis.

## Project Structure

```
helios-trajectory-analysis/
├── src/
│   ├── consciousness_metrics.py      # Consciousness metrics extraction
│   ├── chaos_analysis.py             # Chaos analysis tools
│   ├── deep_pattern_analysis.py      # Deep pattern analysis
│   ├── epiplexity_estimator.py       # Epiplexity estimation
│   ├── influence_detection.py        # Influence detection
│   └── helios_anomaly_scope.py       # Anomaly scope analysis
├── tests/
│   ├── test_consciousness_metrics.py
│   ├── test_chaos_analysis.py
│   └── ...
├── docs/
│   ├── consciousness_metrics.md
│   ├── chaos_analysis.md
│   ├── entropy_analysis.md
│   └── README.md
└── ...
```

## Key Modules

### 1. Consciousness Metrics (`consciousness_metrics.py`)
Provides standardized interfaces for accessing key performance indicators derived from consciousness research.

**Key Features:**
- Integrated Information (Φ) extraction
- Epiplexity score computation
- Standardized metric access interface

**Usage:**
```python
from consciousness_metrics import ConsciousnessMetrics

m = ConsciousnessMetrics()
print(m.get_integrated_information())
print(m.get_epiplexity_score())
```

### 2. Chaos Analysis (`chaos_analysis.py`)
Comprehensive chaos analysis tools for analyzing the dynamics and complexity of time series data.

**Key Features:**
- Wolf Lyapunov Exponent (more accurate than Rosenstein's method)
- Multiscale Entropy (MSE) computation
- Permutation Entropy (PE) analysis
- Fuzzy Entropy estimation
- Renyi Entropy computation
- Shannon Entropy calculation
- Entropy bounds validation

**Usage:**
```python
from chaos_analysis import (
    wolf_lyapunov_exponent,
    multiscale_entropy,
    permutation_entropy,
)

le = wolf_lyapunov_exponent(data, max_lag=100, min_lag=5)
mse = multiscale_entropy(data, scale_factors=[1, 2, 3, 4, 5])
pe = permutation_entropy(data, n_perm=1000)
```

### 3. Entropy Analysis (`entropy_analysis.py`)
Additional entropy computation and validation tools.

**Key Features:**
- Sample Entropy (SampEn)
- Approximate Entropy (ApEn)
- Multiscale Entropy (MSE)
- Permutation Entropy (PE)
- Fuzzy Entropy
- Renyi Entropy
- Shannon Entropy
- Entropy bounds validation

**Usage:**
```python
from entropy_analysis import (
    sample_entropy,
    multiscale_entropy,
    compute_all_entropies,
)

entropy = sample_entropy(data, window_size=5, scaling_factor=1.5)
entropies = compute_all_entropies(data, window_size=5, scaling_factor=1.5)
```

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import numpy as np
from consciousness_metrics import ConsciousnessMetrics
from chaos_analysis import (
    wolf_lyapunov_exponent,
    multiscale_entropy,
    permutation_entropy,
)

# Generate sample data
data = np.random.randn(1000)

# Compute consciousness metrics
metrics = ConsciousnessMetrics()
print(f"Integrated Information: {metrics.integrated_information}")
print(f"Epiplexity Score: {metrics.epiplexity_score}")

# Compute chaos measures
le = wolf_lyapunov_exponent(data, max_lag=100, min_lag=5)
mse = multiscale_entropy(data, scale_factors=[1, 2, 3, 4, 5])
pe = permutation_entropy(data, n_perm=1000)

print(f"Lyapunov Exponent: {le}")
print(f"Multiscale Entropy: {mse}")
print(f"Permutation Entropy: {pe}")
```

## Testing

```bash
python -m pytest tests/
```

## Documentation

- [Consciousness Metrics](docs/consciousness_metrics.md)
- [Chaos Analysis](docs/chaos_analysis.md)
- [Entropy Analysis](docs/entropy_analysis.md)

## License

MIT License