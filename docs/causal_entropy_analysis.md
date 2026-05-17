# Causal Entropy Analysis for Helios

## Overview
This module implements novel causal entropy estimation capabilities for the Helios trajectory analysis platform.

## Addressed Gaps
1. **Causal entropy estimation** - Helps distinguish correlation from causation in consciousness-related trajectory data
2. **Transfer entropy** - Measures directed information flow between variables
3. **Granger causality tests** - Statistical tests for causal relationships
4. **Recurrent network entropy** - Complexity measures for causal relationships

## Key Functions

### `calculate_conditional_entropy(target, source, lag=1)`
Calculates conditional entropy H(target | source). Measures uncertainty in target given knowledge of source. Lower values indicate stronger causal influence.

### `calculate_transfer_entropy(source, target, lag=1, embedding_dim=3, tau=1)`
Calculates transfer entropy from source to target. Measures directed information flow between variables.

### `calculate_granger_causality(source, target, max_lag=5, embedding_dim=2)`
Tests for Granger causality from source to target using conditional variance reduction.

### `calculate_recurrent_network_entropy(source, target, embedding_dim=3, tau=1)`
Calculates entropy of recurrent network between source and target. Measures complexity of causal relationships.

## Usage Example
```python
from cuquantum_accelerator.causal_entropy import calculate_transfer_entropy

# Analyze influence from neural activity to consciousness metrics
source = neural_activity_series  # Shape: (n_samples,)
target = consciousness_metric_series  # Shape: (n_samples,)

te = calculate_transfer_entropy(source, target, embedding_dim=3)
print(f"Transfer entropy: {te:.4f} bits")
```

## Integration with Helios
This module integrates seamlessly with existing entropy analysis:
- `consciousness_metrics.py` - Use for influence detection between consciousness components
- `chaos_analysis.py` - Combine with chaos metrics for comprehensive analysis
- `epiplexity_estimator.py` - Cross-validate epiplexity estimates

## Quantum Methods
Future integration with cuQuantum accelerator for:
- Quantum mutual information between detector channels
- Quantum causal discovery algorithms
- Entanglement-based influence detection