# Multiscale Entropy Analysis

## Overview

This document describes the integration of local and global scales to mitigate finite-data bias in entropy estimation for QRNG analysis. The approach analyzes the transition from chaotic to deterministic regimes across multiple time windows, ensuring robustness against short-term sampling artifacts.

## Motivation

Traditional single-scale entropy estimators suffer from:
- **Finite-sample bias**: Small datasets yield unreliable entropy estimates
- **Non-stationarity**: QRNG streams may exhibit regime changes over time
- **Sampling artifacts**: Short-term fluctuations can mask true randomness properties

Multiscale analysis addresses these by examining entropy evolution across embedding dimensions and time windows.

## Methodology

### 1. Multiscale Entropy (MSE) Framework

```
MSE(data, scales) = {
    MSE_scale_s(data): for each scale s in scales
}
```

Where each scale corresponds to an embedding dimension:
- **Scale 1**: Local dynamics (short-term correlations)
- **Scale 2**: Medium-term patterns (intermediate correlations)
- **Scale 3+**: Global structure (long-range dependencies)

### 2. Transition Detection

The transition from chaotic to deterministic regimes is detected by:

1. **Entropy slope analysis**: Plot entropy vs. scale
   - Steep slope → Chaotic regime (high complexity)
   - Flat slope → Deterministic regime (low complexity)

2. **Inflection point detection**: Identify scale where entropy stabilizes
   - Pre-inflection: Scale-dependent behavior
   - Post-inflection: Scale-invariant (true randomness)

### 3. Robustness Metrics

- **Sample efficiency**: Minimum data required for reliable estimates at each scale
- **Bias correction**: Adjust single-scale estimates using multiscale trends
- **Confidence propagation**: Bootstrap uncertainty across scales

## Implementation

```python
from helios.entropy import calculate_multiscale_entropy

# Calculate entropy across multiple scales
results = calculate_multiscale_entropy(
    data=qrng_stream,
    scales=[1, 2, 3, 4, 5]
)

for scale, metrics in results.items():
    print(f"Scale {scale}: Entropy = {metrics['entropy']:.4f}")
```

## Expected Output

```
Scale 1: Entropy = 0.8234 (Embedding dim: 1)
Scale 2: Entropy = 1.1567 (Embedding dim: 2)
Scale 3: Entropy = 1.4892 (Embedding dim: 3)
Scale 4: Entropy = 1.7201 (Embedding dim: 4)
Scale 5: Entropy = 1.8934 (Embedding dim: 5)
```

Interpretation:
- Increasing entropy with scale → Chaotic regime (good for QRNG)
- Plateauing entropy → Deterministic regime (potential bias)

## Integration with Helios

The multiscale framework integrates seamlessly with existing Helios modules:

```python
from helios.entropy_estimators import calculate_multiscale_entropy
from helios.consciousness_metrics import calculate_consciousness

# Multiscale analysis
mse_results = calculate_multiscale_entropy(qrng_data, scales=[1, 2, 3])

# Consciousness metrics at each scale
for scale, result in mse_results.items():
    if result['entropy'] is not None:
        consciousness_val, _ = calculate_consciousness(
            qrng_data,
            scale=scale
        )
```

## References

1. Costa, M., Goldberger, A.L., & Peng, C.K. (2002). Multiscale entropy of physiological signals.
2. Richman, J.S., & Moorman, J.R. (2000). Physiological time-series analysis using approximate entropy and sample entropy.
3. Shannon, C.E. (1948). A mathematical theory of communication.

## See Also

- `helios.entropy_estimators.calculate_entropy`: Single-scale entropy
- `helios.chaos_analysis.mutual_information`: Chaos metric integration
- `helios.consciousness_metrics.calculate_consciousness`: Consciousness estimation