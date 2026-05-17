# Entropy Estimators Usage Guide

## Quick Start

```python
from helios.entropy_estimators import shannon_entropy, renyi_alpha
import numpy as np

# Analyze a QRNG sequence
sequence = np.random.randint(0, 2, size=10000)
print(f"Shannon entropy: {shannon_entropy(sequence):.4f}")
print(f"Rényi entropy (α=2): {renyi_alpha(sequence, alpha=2.0):.4f}")
```

## Installation

```bash
pip install numpy scipy nolds pyunicorn  # Optional packages
```

## Functions Reference

### `shannon_entropy(data)`

Calculate Shannon entropy (bits) for discrete data.

**Example:**
```python
from helios.entropy_estimators import shannon_entropy

# Binary sequence with 50/50 distribution
binary = [0, 1] * 5000
print(shannon_entropy(binary))  # ≈ 1.0 bits
```

### `renyi_alpha(data, alpha=1.0)`

Calculate Rényi entropy of order alpha.

**Example:**
```python
from helios.entropy_estimators import renyi_alpha

# Higher alpha emphasizes rare events
binary = [0, 1] * 5000
print(renyi_alpha(binary, alpha=2.0))  # ≈ 1.0 bits
```

### `kolmogorov_sinai(data)`

Placeholder for Kolmogorov-Sinai entropy.

**For real KS entropy:**
```python
from pyunicorn.inference.entropy import kolmogorov_sinai as ks_entropy
print(ks_entropy(binary))
```

## Integration with Helios Analysis

### In chaos_analysis.py:

```python
from helios.entropy_estimators import shannon_entropy, renyi_alpha
import numpy as np

class ChaosAnalyzer:
    def __init__(self):
        self.shannon = shannon_entropy
        self.renyi = renyi_alpha
    
    def analyze_sequence(self, sequence):
        """Analyze entropy properties of QRNG sequence."""
        return {
            'shannon': self.shannon(sequence),
            'renyi_2': self.renyi(sequence, alpha=2.0),
            'renyi_inf': self.renyi(sequence, alpha=float('inf')),
        }
```

### In consciousness_metrics.py:

```python
from helios.entropy_estimators import shannon_entropy
import numpy as np

def consciousness_metric(phi, entropy):
    """Calculate consciousness metric from chaos and entropy."""
    # Normalize entropy to [0, 1]
    max_entropy = np.log2(2**phi) if phi > 0 else 1.0
    normalized_entropy = min(entropy / max_entropy, 1.0)
    
    # Consciousness metric: deviation from maximum randomness
    return 1.0 - normalized_entropy
```

## Advanced Usage

### Batch Analysis:

```python
from helios.entropy_estimators import shannon_entropy
import numpy as np

def batch_entropy_analysis(sequences):
    """Analyze entropy for multiple sequences."""
    results = []
    for seq in sequences:
        results.append({
            'shannon': shannon_entropy(seq),
            'length': len(seq),
        })
    return results
```

### Entropy Rate Calculation:

```python
from helios.entropy_estimators import shannon_entropy
import numpy as np

def entropy_rate(sequence, window_size=100):
    """Calculate average entropy rate over sliding windows."""
    rates = []
    for i in range(0, len(sequence) - window_size, window_size // 2):
        window = sequence[i:i + window_size]
        rates.append(shannon_entropy(window))
    return np.mean(rates)
```

## Troubleshooting

### Import Errors:

If you get `ModuleNotFoundError` for `scipy`, `nolds`, or `pyunicorn`:
- The fallback implementations will be used automatically
- Install the packages: `pip install scipy nolds pyunicorn`

### Zero Entropy Warning:

```python
# All identical values give zero entropy
shannon_entropy([0, 0, 0, 0])  # Returns 0.0 (correct!)
```

### Log(0) Error:

The fallback implementation handles zero probabilities:
```python
shannon_entropy([0, 1, 0, 1])  # Works correctly
```

## Performance Notes

- For large sequences (>1M elements), use numpy arrays
- Fallback implementations are ~2x slower than scipy equivalents
- Consider caching entropy calculations for repeated analysis

## References

- [Shannon Entropy](https://en.wikipedia.org/wiki/Shannon_entropy)
- [Rényi Entropy](https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy)
- [Kolmogorov-Sinai Entropy](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Sinai_entropy)

## License

Part of helios-trajectory-analysis project.
