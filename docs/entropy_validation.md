# Entropy Validation Module

## Overview

This module provides robust entropy validation and estimation functions for the Helios Trajectory Analysis platform.

## Functions

### `validate_entropy_function(func_name, alpha=1.0)`

Validates entropy function usage to prevent TypeError exceptions.

**Parameters:**
- `func_name`: Must be 'Renyi' or 'Tsallis'
- `alpha`: Entropy parameter (must be non-negative float or int)

**Returns:**
- Tuple of (bool, str) indicating validation status and message

### `_kde_entropy(data)`

Fallback KDE-based entropy estimator with robust error handling.

**Parameters:**
- `data`: NumPy array of data points

**Returns:**
- Float: Estimated entropy value

## Usage Example

```python
from helios.entropy_validation import validate_entropy_function

# Validate Renyi entropy calculation
try:
    result = validate_entropy_function('Renyi', alpha=2.0)
except ValueError as e:
    print(f"Validation error: {e}")
```

## Notes

- All functions include proper exception handling
- Type validation prevents common TypeError issues
- Fallback implementations ensure numerical stability