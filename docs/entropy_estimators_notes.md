# Entropy Estimators Implementation Notes

## Overview
This document tracks the implementation status of various entropy estimators for QRNG analysis.

## Completed Implementations

### 1. Kozachenko-Leonenko (kNN)
- **File**: `metrics/entropy_estimators.py`
- **Status**: ✅ Implemented
- **Description**: Non-parametric entropy estimator using k-nearest neighbor distances
- **Parameters**: `data` (np.ndarray), `k` (int, default=2)
- **Returns**: float (estimated entropy in nats)

### 2. Bias-Corrected Miller-Madow
- **File**: `metrics/entropy_estimators.py`
- **Status**: ✅ Implemented
- **Description**: Histogram-based entropy with bias correction for small samples
- **Parameters**: `data` (np.ndarray), `n` (int, default=len(data))
- **Returns**: float (bias-corrected entropy in nats)

### 3. Rényi Entropy (α=0.5 and α=2)
- **File**: `metrics/entropy_estimators.py`
- **Status**: ✅ Implemented
- **Description**: Correlation integral-based estimators for Rényi entropy at specific alpha values
- **Parameters**: `data` (np.ndarray)
- **Returns**: tuple[float, float] (α=0.5, α=2 entropies in nats)

### 4. Tsallis Entropy (q=1.2)
- **File**: `metrics/entropy_estimators.py`
- **Status**: ✅ Implemented
- **Description**: Non-extensive entropy estimator with q parameter
- **Parameters**: `data` (np.ndarray)
- **Returns**: float (Tsallis entropy in nats)

### 5. GPU Histogram Estimator
- **File**: `metrics/entropy_estimators.py`
- **Status**: ✅ Implemented (with CuPy fallback)
- **Description**: GPU-accelerated histogram-based entropy estimation
- **Parameters**: `data` (np.ndarray)
- **Returns**: float (Shannon entropy in nats)

## Missing Estimators (To Be Implemented)

### 1. Lepetit Entropy
- **Status**: ❌ Not implemented
- **Description**: Alternative entropy estimator based on Lepetit's method
- **Parameters needed**: `data`, `window_size`
- **Implementation note**: Requires custom implementation

### 2. CRaM-Based Methods
- **Status**: ❌ Not implemented
- **Variants**:
  - CRaM (Classical Randomness Measurement)
  - CRaM2 (Enhanced version)
- **Parameters needed**: `data`, `method` ('CRaM' or 'CRaM2')
- **Implementation note**: Requires statistical framework

### 3. Other Custom Implementations
- **Status**: ❌ Not implemented
- **Potential candidates**:
  - Min-entropy estimator
  - Max-entropy estimator
  - Collision entropy (Rényi α=2 alternative)

## Test Coverage

### Unit Tests
- **File**: `tests/entropy_estimators.py`
- **Status**: ✅ Implemented with pytest_mock
- **Test cases**:
  - `test_constant_input`: Verifies zero entropy for constant data
  - `test_nan_input`: Verifies NaN propagation
  - `test_inf_input`: Verifies infinity propagation
  - `test_low_sample_bias`: Verifies bias correction at low sample sizes

## Recommendations

1. **Priority**: Implement Lepetit and CRaM estimators next
2. **Testing**: Add integration tests with real QRNG data sources
3. **Documentation**: Add docstrings with mathematical formulas for each estimator
4. **Performance**: Profile GPU vs CPU performance for large datasets

## References

- Kozachenko & Leonenko (1987): "Sample Estimate of the Entropy of a Random Vector"
- Miller (1955) & Madow (1942): Bias correction for histogram entropy
- Rényi (1961): "On Measures of Entropy and Information"
- Tsallis (1988): "Possible Generalization of Boltzmann-Gibbs Statistics"
