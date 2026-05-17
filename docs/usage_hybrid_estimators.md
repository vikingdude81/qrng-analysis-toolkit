# Hybrid Entropy Estimators Usage Guide

## Quick Start

```python
from src.entropy_estimators import (
    SymbolicWeightedWaveletEntropy,
    NonStationarySymbolicEntropy,
    EntropyEstimatorEnsemble,
)
import numpy as np

# Generate sample helios data
t = np.arange(500)
helios_data = np.sin(2 * np.pi * 0.1 * t) + np.random.randn(500) * 0.3

# Use the ensemble (recommended)
ensemble = EntropyEstimatorEnsemble(use_hybrid=True)
entropies = ensemble.fit_transform(helios_data)

print("Entropy values:")
for i, e in enumerate(entropies):
    print(f"  Estimator {i}: {e:.4f}")
```

## Basic Usage Examples

### Example 1: SWWE for Thermal Noise Reduction

```python
from src.entropy_estimators import SymbolicWeightedWaveletEntropy
import numpy as np

# Load helios data
helios_data = load_helios_data('data/helios_sample.npy')

# Initialize SWWE with default parameters
swwe = SymbolicWeightedWaveletEntropy()

# Calculate entropy
entropy = swwe.fit_transform(helios_data)
print(f"SWWE Entropy: {entropy:.4f}")
```

### Example 2: NSAE for Non-Stationary Data

```python
from src.entropy_estimators import NonStationarySymbolicEntropy
import numpy as np

# Load helios data with non-stationarity
helios_data = load_helios_data('data/helios_nonstationary.npy')

# Initialize NSAE
nsae = NonStationarySymbolicEntropy(
    embedding_dim=3,
    tau=1,
    transition_window=10,
    alphabet_size=8
)

# Calculate entropy
entropy = nsae.fit_transform(helios_data)
print(f"NSAE Entropy: {entropy:.4f}")
```

### Example 3: Comparing All Estimators

```python
from src.entropy_estimators import (
    SampleEntropy,
    ApproximateEntropy,
    PermutationEntropy,
    SymbolicWeightedWaveletEntropy,
    NonStationarySymbolicEntropy,
)
import numpy as np

helios_data = load_helios_data('data/helios_sample.npy')

# Standard estimators
se = SampleEntropy()
sa = ApproximateEntropy()
pe = PermutationEntropy()

# Hybrid estimators
swwe = SymbolicWeightedWaveletEntropy()
nsae = NonStationarySymbolicEntropy()

# Calculate all entropies
entropies = {
    'sample_entropy': se.fit_transform(helios_data),
    'approximate_entropy': sa.fit_transform(helios_data),
    'permutation_entropy': pe.fit_transform(helios_data),
    'swwe': swwe.fit_transform(helios_data),
    'nsae': nsae.fit_transform(helios_data),
}

print("Entropy Comparison:")
for name, value in entropies.items():
    print(f"  {name}: {value:.4f}")
```

## Advanced Usage

### Example 4: State Classification with Hybrid Estimators

```python
from src.metrics_integration import ConsciousnessMetricsPipeline
import numpy as np

# Load helios data
helios_data = load_helios_data('data/helios_sample.npy')

# Initialize pipeline with hybrid estimators
pipeline = ConsciousnessMetricsPipeline(use_hybrid=True)

# Classify consciousness state
state = pipeline.classify_state(helios_data)

print(f"State: {state.state_label}")
print(f"Confidence: {state.confidence:.4f}")
print(f"SWWE Entropy: {state.swwe_entropy:.4f}")
print(f"NSAE Entropy: {state.nsaes_entropy:.4f}")
```

### Example 5: Detecting State Transitions

```python
from src.metrics_integration import ConsciousnessMetricsPipeline
import numpy as np

helios_data = load_helios_data('data/helios_sample.npy')

pipeline = ConsciousnessMetricsPipeline(use_hybrid=True)

# Detect state transitions
transitions = pipeline.detect_state_transitions(
    helios_data,
    window_size=50,
    threshold=0.3
)

print(f"Detected {len(transitions)} state transitions:")
for i, transition in enumerate(transitions):
    print(f"  Transition {i}: t={transition['time']:.1f}, "
          f"from='{transition['from_state']}', to='{transition['to_state']}'")
```

### Example 6: Noise Sensitivity Analysis

```python
from src.entropy_estimators import EntropyComparisonAnalyzer
import numpy as np

helios_data = load_helios_data('data/helios_sample.npy')

# Create noisy version
noise_level = 0.1
noisy_data = helios_data + np.random.randn(len(helios_data)) * noise_level

analyzer = EntropyComparisonAnalyzer()

# Analyze noise sensitivity
sensitivity = analyzer.analyze_noise_sensitivity(
    clean_data=helios_data,
    noisy_data=noisy_data,
    noise_level=noise_level
)

print("Noise Sensitivity Analysis:")
for metric, value in sensitivity.items():
    print(f"  {metric}: {value:.4f}")
```

### Example 7: Parameter Tuning

```python
from src.entropy_estimators import SymbolicWeightedWaveletEntropy
import numpy as np

helios_data = load_helios_data('data/helios_sample.npy')

# Test different wavelet types
for wavelet_type in ['db4', 'sym8', 'coif5']:
    swwe = SymbolicWeightedWaveletEntropy(wavelet_type=wavelet_type)
    entropy = swwe.fit_transform(helios_data)
    print(f"SWWE ({wavelet_type}): {entropy:.4f}")

# Test different embedding dimensions
for dim in [2, 3, 4, 5]:
    swwe = SymbolicWeightedWaveletEntropy(embedding_dim=dim)
    entropy = swwe.fit_transform(helios_data)
    print(f"SWWE (dim={dim}): {entropy:.4f}")
```

## Integration with Other Modules

### Example 8: Using with Chaos Analysis

```python
from src.chaos_analysis import ChaosAnalysis
from src.entropy_estimators import EntropyEstimatorEnsemble
import numpy as np

helios_data = load_helios_data('data/helios_sample.npy')

# Calculate chaos metrics
chaos = ChaosAnalysis()
chaos_metrics = chaos.analyze(helios_data)

# Use hybrid entropy estimators for additional metrics
ensemble = EntropyEstimatorEnsemble(use_hybrid=True)
entropies = ensemble.fit_transform(helios_data)

print("Chaos Analysis with Hybrid Entropy:")
for key, value in chaos_metrics.items():
    print(f"  {key}: {value:.4f}")
print(f"\nHybrid Entropies: {entropies}")
```

### Example 9: Using with Deep Pattern Analysis

```python
from src.deep_pattern_analysis import DeepPatternAnalysis
from src.entropy_estimators import EntropyEstimatorEnsemble
import numpy as np

helios_data = load_helios_data('data/helios_sample.npy')

# Analyze deep patterns
patterns = DeepPatternAnalysis()
pattern_results = patterns.analyze(helios_data)

# Use hybrid entropy for pattern validation
ensemble = EntropyEstimatorEnsemble(use_hybrid=True)
entropies = ensemble.fit_transform(helios_data)

print("Deep Pattern Analysis with Hybrid Entropy:")
for key, value in pattern_results.items():
    print(f"  {key}: {value:.4f}")
print(f"\nHybrid Entropies: {entropies}")
```

## Batch Processing

### Example 10: Processing Multiple Datasets

```python
from src.entropy_estimators import EntropyEstimatorEnsemble
import numpy as np

# Load multiple datasets
datasets = [
    load_helios_data('data/helios_sample.npy'),
    load_helios_data('data/helios_nonstationary.npy'),
    load_helios_data('data/helios_small.npy'),
]

# Initialize ensemble
ensemble = EntropyEstimatorEnsemble(use_hybrid=True)

# Process all datasets
results = []
for i, data in enumerate(datasets):
    entropies = ensemble.fit_transform(data)
    results.append({
        'dataset': f'dataset_{i}',
        'entropies': entropies,
        'mean_entropy': np.mean(entropies),
    })

for result in results:
    print(f"{result['dataset']}: mean={result['mean_entropy']:.4f}")
```

## Error Handling

### Handling Small Datasets

```python
from src.entropy_estimators import EntropyEstimatorEnsemble
import numpy as np

# Small dataset
small_data = load_helios_data('data/helios_small.npy')  # N=20

try:
    ensemble = EntropyEstimatorEnsemble(use_hybrid=True)
    entropies = ensemble.fit_transform(small_data)
except ValueError as e:
    print(f"Error: {e}")
    print("Solution: Increase embedding_dim or reduce tau")
```

### Handling Noisy Data

```python
from src.entropy_estimators import SymbolicWeightedWaveletEntropy
import numpy as np

# Very noisy data
noisy_data = load_helios_data('data/helios_sample.npy') + np.random.randn(500) * 5.0

try:
    swwe = SymbolicWeightedWaveletEntropy()
    entropy = swwe.fit_transform(noisy_data)
except ValueError as e:
    print(f"Error: {e}")
    print("Solution: Use wavelet denoising or reduce noise level")
```

## Performance Tips

1. **Use ensemble for batch processing** - more efficient than individual estimators
2. **Choose appropriate embedding_dim** based on dataset size:
   - N > 500: dim=3-5
   - 100 < N < 500: dim=2-3
   - N < 100: dim=2, tau=1
3. **Use wavelet denoising** for thermal noise in helios data
4. **Monitor SWWE/NSAE ratio** as a robustness indicator

## Troubleshooting

### Issue: "need at least N data points"

**Solution:** Increase embedding_dim or reduce tau.

```python
swwe = SymbolicWeightedWaveletEntropy(embedding_dim=2, tau=1)
```

### Issue: "wavelet_type not supported"

**Solution:** Use one of: 'db4', 'sym8', 'coif5'

```python
swwe = SymbolicWeightedWaveletEntropy(wavelet_type='db4')
```

### Issue: Entropy values are NaN

**Solution:** Check data quality and increase sample size.

```python
# Ensure data is not constant
if np.std(helios_data) < 0.1:
    print("Warning: Data has very low variance")
```

## See Also

- `api_hybrid_estimators.md`: Complete API reference
- `entropy_estimator_analysis.md`: Statistical validity analysis
- `src/entropy_estimators.py`: Source code
