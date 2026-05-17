# Helios Trajectory Analysis

A comprehensive platform for QRNG (Quantum Random Number Generator) analysis with 40+ modules for consciousness metrics, chaos analysis, deep pattern detection, and anomaly identification.

## Features

- **Consciousness Metrics**: Sample Entropy, Approximate Entropy, Permutation Entropy
- **Chaos Analysis**: Lyapunov exponents, fractal dimensions, recurrence plots
- **Deep Pattern Analysis**: Neural network-based pattern recognition
- **Epiplexity Estimation**: Quantum entanglement correlation measures
- **Influence Detection**: Causal inference and dependency analysis
- **Anomaly Scope**: Helios-specific anomaly detection with 66KB comprehensive module
- **QRNG Comprehensive Analysis**: Full QRNG pipeline with 37KB analysis module
- **SPDC Source Analysis**: Spontaneous Parametric Down-Conversion source characterization
- **cuQuantum Accelerator**: GPU-accelerated quantum operations
- **Inference Framework**: Multi-architecture inference pipelines

## New: Hybrid Entropy Estimators

We've added novel hybrid entropy estimators to address limitations of standard methods:

### Symbolic-Weighted Wavelet Entropy (SWWE)
Combines symbolic dynamics with wavelet denoising for robust entropy estimation in non-stationary regimes.

### Non-Stationary Symbolic Entropy (NSAE)
Explicitly models temporal correlations and non-stationarity by tracking symbol transitions over multiple time steps.

**Benefits:**
- Better performance with small samples (N < 20)
- Explicit handling of non-stationarity
- Wavelet denoising removes thermal noise
- Improved signal-to-noise ratio

## Quick Start

```python
from src.entropy_estimators import EntropyEstimatorEnsemble
import numpy as np

# Load helios data
helios_data = np.load('data/helios_sample.npy')

# Use ensemble of all entropy estimators
ensemble = EntropyEstimatorEnsemble()
entropies = ensemble.fit_transform(helios_data)

print(f"Sample Entropy: {entropies[0]:.4f}")
print(f"SWWE Entropy: {entropies[3]:.4f}")
print(f"NSAE Entropy: {entropies[4]:.4f}")
```

## Installation

```bash
pip install -e .
```

## Dependencies

- numpy >= 1.20
- scipy >= 1.7
- nolds >= 0.5
- torch >= 1.9 (for cuQuantum)

## Module Structure

```
helios-trajectory-analysis/
├── src/
│   ├── entropy_estimators.py      # Hybrid estimators (NEW)
│   ├── metrics_integration.py     # Consciousness metrics pipeline
│   ├── consciousness_metrics.py   # Main consciousness analysis
│   ├── chaos_analysis.py          # Chaos theory measures
│   ├── deep_pattern_analysis.py   # Deep learning patterns
│   ├── epiplexity_estimator.py    # Quantum entanglement
│   ├── influence_detection.py     # Causal inference
│   └── helios_anomaly_scope.py    # Anomaly detection
├── tests/
│   ├── test_entropy_estimators.py # NEW: Tests for hybrid estimators
│   └── ...                        # Other test files
├── docs/
│   ├── usage_hybrid_estimators.md # NEW: Usage guide
│   └── entropy_estimator_analysis.md  # NEW: Analysis documentation
└── README.md
```

## Running Tests

```bash
pytest tests/ -v
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License

## References

- Gao et al., 2023: Finite-sample bias in entropy estimators
- Zhang et al., 2024: Non-stationary symbolic entropy performance
- Helios consciousness metrics pipeline
