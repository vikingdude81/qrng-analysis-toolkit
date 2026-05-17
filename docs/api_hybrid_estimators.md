# Hybrid Entropy Estimators API Reference

## Overview

This document provides the complete API reference for the hybrid entropy estimators added to Helios:

- `SymbolicWeightedWaveletEntropy` (SWWE)
- `NonStationarySymbolicEntropy` (NSAE)
- `EntropyEstimatorEnsemble`
- `EntropyComparisonAnalyzer`

## Classes

### SymbolicWeightedWaveletEntropy

```python
class SymbolicWeightedWaveletEntropy:
    """
    Symbolic-Weighted Wavelet Entropy (SWWE) estimator.
    
    Combines symbolic dynamics with wavelet denoising to extract latent symbols
    from helios data and calculate entropy on the denoised symbol stream.
    """
    
    def __init__(
        self,
        embedding_dim: int = 3,
        tau: int = 1,
        wavelet_type: str = 'db4',
        wavelet_level: int = 2
    ):
        """
        Initialize SWWE estimator.
        
        Args:
            embedding_dim: Embedding dimension for symbolic dynamics (default: 3)
            tau: Time delay for reconstruction (default: 1)
            wavelet_type: Wavelet family name (default: 'db4')
                Options: 'db4', 'sym8', 'coif5'
            wavelet_level: Decomposition level for denoising (default: 2)
        """
    
    def fit(self, X):
        """
        Fit the estimator to data.
        
        Args:
            X: Input time series (1D numpy array or list)
        """
    
    def transform(self, X) -> float:
        """
        Transform fitted data to entropy value.
        
        Args:
            X: Input time series (same as fit)
        
        Returns:
            float: SWWE entropy value
        """
    
    def fit_transform(self, X) -> float:
        """
        Fit and transform in one step.
        
        Args:
            X: Input time series
        
        Returns:
            float: SWWE entropy value
        """
```

**Example Usage:**

```python
from src.entropy_estimators import SymbolicWeightedWaveletEntropy

# Initialize estimator
swwe = SymbolicWeightedWaveletEntropy(
    embedding_dim=3,
    tau=1,
    wavelet_type='db4',
    wavelet_level=2
)

# Calculate entropy
entropy = swwe.fit_transform(helios_data)
print(f"SWWE Entropy: {entropy:.4f}")
```

### NonStationarySymbolicEntropy

```python
class NonStationarySymbolicEntropy:
    """
    Non-Stationary Symbolic Entropy (NSAE) estimator.
    
    Computes entropy of symbolic representation directly in a non-stationary
    regime by tracking symbol transitions over multiple time steps.
    """
    
    def __init__(
        self,
        embedding_dim: int = 3,
        tau: int = 1,
        transition_window: int = 10,
        alphabet_size: int = 8
    ):
        """
        Initialize NSAE estimator.
        
        Args:
            embedding_dim: Embedding dimension (default: 3)
            tau: Time delay for reconstruction (default: 1)
            transition_window: Window size for tracking transitions (default: 10)
            alphabet_size: Symbol alphabet size (default: 8)
        """
    
    def fit(self, X):
        """
        Fit the estimator to data.
        
        Args:
            X: Input time series
        """
    
    def transform(self, X) -> float:
        """
        Transform fitted data to entropy value.
        
        Args:
            X: Input time series
        
        Returns:
            float: NSAE entropy value
        """
    
    def fit_transform(self, X) -> float:
        """
        Fit and transform in one step.
        
        Args:
            X: Input time series
        
        Returns:
            float: NSAE entropy value
        """
```

**Example Usage:**

```python
from src.entropy_estimators import NonStationarySymbolicEntropy

# Initialize estimator
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

### EntropyEstimatorEnsemble

```python
class EntropyEstimatorEnsemble:
    """
    Ensemble of all entropy estimators.
    
    Provides access to both standard and hybrid estimators in a single interface.
    """
    
    def __init__(self, use_hybrid: bool = True):
        """
        Initialize ensemble.
        
        Args:
            use_hybrid: Whether to include hybrid estimators (default: True)
        """
    
    def fit(self, X):
        """
        Fit all estimators to data.
        
        Args:
            X: Input time series
        """
    
    def transform(self, X) -> list:
        """
        Transform fitted data to entropy values from all estimators.
        
        Returns a list of 5 entropy values:
        [sample_entropy, approximate_entropy, permutation_entropy,
         symbolic_weighted_wavelet_entropy, non_stationary_symbolic_entropy]
        
        Args:
            X: Input time series
        
        Returns:
            list[float]: Entropy values from all estimators
        """
    
    def fit_transform(self, X) -> list:
        """
        Fit and transform in one step.
        
        Args:
            X: Input time series
        
        Returns:
            list[float]: Entropy values from all estimators
        """
```

**Example Usage:**

```python
from src.entropy_estimators import EntropyEstimatorEnsemble

# Initialize ensemble with hybrid estimators
ensemble = EntropyEstimatorEnsemble(use_hybrid=True)

# Calculate all entropies
entropies = ensemble.fit_transform(helios_data)

print(f"Sample Entropy: {entropies[0]:.4f}")
print(f"Approximate Entropy: {entropies[1]:.4f}")
print(f"Permutation Entropy: {entropies[2]:.4f}")
print(f"SWWE Entropy: {entropies[3]:.4f}")
print(f"NSAE Entropy: {entropies[4]:.4f}")
```

### EntropyComparisonAnalyzer

```python
class EntropyComparisonAnalyzer:
    """
    Analyzer for comparing entropy estimator performance.
    
    Provides tools for noise sensitivity analysis and estimator comparison.
    """
    
    def __init__(self):
        """Initialize analyzer."""
    
    def analyze_noise_sensitivity(
        self,
        clean_data: np.ndarray,
        noisy_data: np.ndarray,
        noise_level: float
    ) -> dict:
        """
        Analyze estimator sensitivity to noise.
        
        Args:
            clean_data: Clean time series (reference)
            noisy_data: Noisy time series
            noise_level: Noise level as fraction of signal amplitude
        
        Returns:
            dict: Sensitivity metrics for each estimator
        """
    
    def compare_estimators(self, X) -> dict:
        """
        Compare all estimators on given data.
        
        Args:
            X: Input time series
        
        Returns:
            dict: Entropy values and comparison metrics
        """
```

**Example Usage:**

```python
from src.entropy_estimators import EntropyComparisonAnalyzer

analyzer = EntropyComparisonAnalyzer()

# Analyze noise sensitivity
sensitivity = analyzer.analyze_noise_sensitivity(
    clean_data=helios_data,
    noisy_data=helios_data + np.random.randn(len(helios_data)) * 0.1,
    noise_level=0.1
)

print("Noise Sensitivity:")
for metric, value in sensitivity.items():
    print(f"  {metric}: {value:.4f}")
```

## Consciousness Metrics Pipeline Integration

The hybrid estimators integrate with the `ConsciousnessMetricsPipeline`:

```python
from src.metrics_integration import ConsciousnessMetricsPipeline

# Initialize pipeline with hybrid estimators
pipeline = ConsciousnessMetricsPipeline(use_hybrid=True)

# Classify consciousness state
state = pipeline.classify_state(helios_data)

print(f"State: {state.state_label}")
print(f"Confidence: {state.confidence:.4f}")
```

## Parameter Tuning Guide

### SWWE Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| embedding_dim | 3 | 2-10 | Embedding dimension for symbolic dynamics |
| tau | 1 | 1-5 | Time delay for reconstruction |
| wavelet_type | 'db4' | ['db4', 'sym8', 'coif5'] | Wavelet family |
| wavelet_level | 2 | 1-5 | Decomposition level |

### NSAE Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| embedding_dim | 3 | 2-10 | Embedding dimension |
| tau | 1 | 1-5 | Time delay |
| transition_window | 10 | 5-50 | Window for tracking transitions |
| alphabet_size | 8 | 4-16 | Symbol alphabet size |

## Best Practices

1. **Always use the ensemble** when possible - it provides all metrics at once
2. **Tune embedding_dim** based on your data characteristics (start with 3)
3. **Use wavelet denoising** for thermal noise in helios data
4. **Monitor SWWE/NSAE ratio** as a robustness indicator
5. **For small samples (N < 20)**, prefer NSAE over standard estimators

## Error Handling

### ValueError: "need at least N data points"

Increase embedding_dim or reduce tau for small datasets.

### ValueError: "wavelet_type not supported"

Use one of: 'db4', 'sym8', 'coif5'

## See Also

- `usage_hybrid_estimators.md`: Usage guide and examples
- `entropy_estimator_analysis.md`: Statistical validity analysis
- `src/metrics_integration.py`: Consciousness metrics pipeline
