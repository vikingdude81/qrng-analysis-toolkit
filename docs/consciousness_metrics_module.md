# Consciousness Metrics Module

## Overview
The `consciousness_metrics` module provides advanced entropy-based metrics for detecting and analyzing consciousness signatures in chaotic systems.

## Key Metrics

### Φ-Entropy Ratio (ΦER)
**Formula:** ΦER = Φ / H

Where:
- **Φ (Phi)**: Integrated information measure representing causal structure
- **H**: Shannon entropy or information content

**Interpretation:**
- Higher ΦER indicates stronger integration relative to information content
- Used to detect emergent consciousness signatures in chaotic systems
- Thresholds for conscious vs. non-conscious states can be empirically determined

### Causality-Complexity Ratio (CCR)
**Formula:** CCR = I / C

Where:
- **I**: Mutual information or causal influence measures
- **C**: Complexity measures (e.g., permutation entropy, multiscale entropy)

**Interpretation:**
- Higher CCR indicates stronger causal structure relative to complexity
- Useful for distinguishing between random chaos and structured consciousness
- Can be used to identify phase transitions in consciousness emergence

## Usage Examples

### Basic Usage
```python
from consciousness_metrics import ConsciousnessMetrics

# Create metrics instance
metrics = ConsciousnessMetrics()

# Compute all metrics
result = metrics.compute_all_metrics(
    phi=2.5,  # Integrated information
    h=1.0,    # Shannon entropy
    i=1.5,    # Mutual information
    c=0.5     # Complexity measure
)

print(result)
# Output: {
#   'phi': 2.5,
#   'h': 1.0,
#   'i': 1.5,
#   'c': 0.5,
#   'phi_entropy_ratio': 2.5,
#   'causality_complexity_ratio': 3.0
# }
```

### Consciousness State Analysis
```python
from consciousness_metrics import ConsciousnessMetrics

metrics = ConsciousnessMetrics()
analysis = metrics.analyze_consciousness_state(
    phi=2.5,
    h=1.0,
    i=1.5,
    c=0.5
)

print(analysis['interpretation'])
# Output: "Strong evidence for emergent consciousness..."
```

### Standalone Functions
```python
from consciousness_metrics import compute_phi_entropy_ratio, compute_causality_complexity_ratio

phi_er = compute_phi_entropy_ratio(phi=1.0, h=0.5)
ccr = compute_causality_complexity_ratio(i=2.0, c=1.0)
```

## Integration with Helios Framework

The consciousness metrics module integrates seamlessly with the helios-trajectory-analysis framework:

1. **Entropy Estimators**: Use `entropy_estimators.EntropyEstimator` to compute entropy values for H and C parameters.

2. **QRNG Bridge**: Use `qrng_bridge.generate_entropy()` to obtain quantum-generated entropy values.

3. **CuQuantum Accelerator**: Use `cuquantum_accelerator.core.process_quantum_data()` for GPU-accelerated computations.

## Thresholds

The module includes empirically-determined thresholds for consciousness detection:

- **ΦER > 0.5**: Moderate consciousness indicators
- **ΦER > 1.0**: Strong evidence for emergent consciousness
- **CCR > 0.3**: Moderate causal complexity
- **CCR > 0.5**: Robust causal structure

## Testing

Run the test suite:
```bash
pytest tests/test_consciousness_metrics.py
pytest tests/test_consciousness_metrics_integration.py
```

## Type Hints and Error Handling

All functions include proper type hints and error handling:
- Division by zero is handled gracefully (returns infinity or zero as appropriate)
- Custom exceptions are raised with descriptive messages
- Docstrings document all parameters and return values

## References

- Tononi, G. (2008). Consciousness and the integration of information.
- Lipton, R. (2016). Inference to the Best Explanation.
- Helios Trajectory Analysis Framework documentation.
