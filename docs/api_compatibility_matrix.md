# API Compatibility Matrix for Entropy/Chaos/Consciousness Modules

## Overview

This document provides a comprehensive compatibility matrix across `helios`, `testbed`, and `toolkit` modules for entropy, chaos, and consciousness analysis functions. Understanding these differences is critical for cross-project integration and data interchange.

## Compatibility Summary

| Module Pair | Status | Key Differences |
|-------------|--------|------------------|
| Helios ↔ Testbed | ✅ Compatible | Normalization: bits (base-2) vs nats (base-e) |
| Helios ↔ Toolkit | ⚠️ Partial | Scaling: 0-1 vs 0-100 for consciousness |
| Testbed ↔ Toolkit | ⚠️ Partial | Same as above |

## Entropy Functions

### API Comparison

| Helios API | Testbed API | Toolkit API | Compatibility Notes |
|------------|-------------|-------------|---------------------|
| `helios.entropy.shannon` | `testbed.measures.shannon_entropy` | `toolkit.entropy.shannon` | **Equivalent** in functionality but **differs in normalization**: Helios/Testbed use bits (base-2), Toolkit uses nats (base-e) |

### Usage Examples

```python
# Helios (bits)
from helios.entropy import shannon
helios_entropy = shannon(data)  # Returns value in bits

# Testbed (bits)
from testbed.measures import shannon_entropy
testbed_entropy = shannon_entropy(data)  # Returns value in bits

# Toolkit (nats)
from toolkit.entropy import shannon
toolkit_entropy = shannon(data)  # Returns value in nats
```

### Conversion Formula

```
1 nat = ln(2) ≈ 0.693 bits
1 bit = log₂(e) ≈ 1.4427 nats

# Convert Helios/Testbed (bits) to Toolkit (nats)
toolkit_value = helios_value * np.log(2)

# Convert Toolkit (nats) to Helios/Testbed (bits)
helios_value = toolkit_value / np.log(2)
```

## Chaos Functions

### API Comparison

| Helios API | Testbed API | Toolkit API | Compatibility Notes |
|------------|-------------|-------------|---------------------|
| `helios.chaos.mutual_information` | `testbed.measures.mutual_information` | `toolkit.chaos.mutual_information` | **Equivalent** in functionality but **differs in normalization**: Helios/Testbed use bits, Toolkit uses nats |

### Usage Examples

```python
# Helios (bits)
from helios.chaos import mutual_information
helios_mi = mutual_information(x, y)  # Returns value in bits

# Testbed (bits)
from testbed.measures import mutual_information
testbed_mi = mutual_information(x, y)  # Returns value in bits

# Toolkit (nats)
from toolkit.chaos import mutual_information
toolkit_mi = mutual_information(x, y)  # Returns value in nats
```

### Conversion Formula

Same as entropy functions:
```python
import numpy as np

# Bits to nats
toolkit_value = helios_value * np.log(2)

# Nats to bits
helios_value = toolkit_value / np.log(2)
```

## Consciousness Functions

### API Comparison

| Helios API | Testbed API | Toolkit API | Compatibility Notes |
|------------|-------------|-------------|---------------------|
| `helios.consciousness.sensitivity` | `testbed.measures.sensitivity` | `toolkit.consciousness.sensitivity` | **Equivalent** in functionality but **differs in normalization**: Helios/Testbed use 0-1 scale, Toolkit uses 0-100 scale |

### Usage Examples

```python
# Helios (0-1 scale)
from helios.consciousness import sensitivity
helios_sens = sensitivity(data)  # Returns value in [0, 1]

# Testbed (0-1 scale)
from testbed.measures import sensitivity
testbed_sens = sensitivity(data)  # Returns value in [0, 1]

# Toolkit (0-100 scale)
from toolkit.consciousness import sensitivity
toolkit_sens = sensitivity(data)  # Returns value in [0, 100]
```

### Conversion Formula

```
scale_100 = scale_01 * 100
scale_01 = scale_100 / 100

# Helios/Testbed to Toolkit
toolkit_value = helios_value * 100

# Toolkit to Helios/Testbed
helios_value = toolkit_value / 100
```

## Normalization Reference Table

| Metric | Helios Unit | Testbed Unit | Toolkit Unit | Conversion Factor |
|--------|-------------|--------------|--------------|-------------------|
| Entropy | bits (base-2) | bits (base-2) | nats (base-e) | ln(2) ≈ 0.693 |
| Mutual Information | bits (base-2) | bits (base-2) | nats (base-e) | ln(2) ≈ 0.693 |
| Consciousness | 0-1 scale | 0-1 scale | 0-100 scale | ×100 |

## Best Practices

### 1. Always Verify Units Before Comparison

```python
import numpy as np

def normalize_entropy(helios_entropy: float) -> float:
    """Convert Helios/Testbed entropy (bits) to Toolkit units (nats)."""
    return helios_entropy * np.log(2)

def normalize_consciousness(helios_sensitivity: float) -> float:
    """Convert Helios/Testbed sensitivity (0-1) to Toolkit units (0-100)."""
    return helios_sensitivity * 100
```

### 2. Use Standardized Interfaces for Cross-Project Work

```python
from typing import Union, Dict
import numpy as np

def normalize_metric(
    value: float,
    source: str,  # 'helios', 'testbed', or 'toolkit'
    target: str   # 'helios', 'testbed', or 'toolkit'
) -> float:
    """Normalize metric between different project APIs."""
    if source == target:
        return value
    
    conversions = {
        ('helios', 'toolkit'): lambda v: v * np.log(2),
        ('testbed', 'toolkit'): lambda v: v * np.log(2),
        ('toolkit', 'helios'): lambda v: v / np.log(2),
        ('toolkit', 'testbed'): lambda v: v / np.log(2),
        ('helios', 'consciousness_toolkit'): lambda v: v * 100,
        ('testbed', 'consciousness_toolkit'): lambda v: v * 100,
        ('toolkit', 'consciousness_helios'): lambda v: v / 100,
    }
    
    key = (source, target)
    if key in conversions:
        return conversions[key](value)
    raise ValueError(f"No conversion defined for {source} → {target}")
```

### 3. Document Source When Sharing Results

```python
result_dict = {
    'entropy': 1.5,
    'source': 'helios',  # Important for downstream consumers!
    'unit': 'bits'
}
```

## Migration Guide

### From Toolkit to Helios

```python
# Convert entropy from nats to bits
helios_entropy = toolkit_entropy / np.log(2)

# Convert consciousness from 0-100 to 0-1
helios_sensitivity = toolkit_sensitivity / 100
```

### From Helios to Toolkit

```python
# Convert entropy from bits to nats
toolkit_entropy = helios_entropy * np.log(2)

# Convert consciousness from 0-1 to 0-100
toolkit_sensitivity = helios_sensitivity * 100
```

## Testing Compatibility

```python
import numpy as np
from helios.entropy import shannon as helios_shannon
from testbed.measures import shannon_entropy as testbed_shannon
from toolkit.entropy import shannon as toolkit_shannon

test_data = np.random.randn(1000)

helios_val = helios_shannon(test_data)
testbed_val = testbed_shannon(test_data)
toolkit_val = toolkit_shannon(test_data)

# Verify compatibility (within tolerance)
assert abs(helios_val - testbed_val) < 1e-10, "Helios/Testbed mismatch"
assert abs(helios_val * np.log(2) - toolkit_val) < 1e-5, "Helios/Toolkit mismatch"
```

## See Also

- `helios.entropy_estimators`: Multiscale entropy implementations
- `helios.chaos_analysis`: Chaos metric calculations
- `helios.consciousness_metrics`: Consciousness estimation functions
- `docs/multiscale_entropy_analysis.md`: Detailed multiscale methodology